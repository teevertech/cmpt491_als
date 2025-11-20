import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger
import typer

from cmpt491_als.modeling.sand_datasets import SANDDataset
from cmpt491_als.modeling.elastic_ast_wrapper import ElasticASTForAudioClassification
from cmpt491_als.modeling.collate import pad_mels
from cmpt491_als.config import (
    MODELS_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    get_training_config,
)

app = typer.Typer()


# -------------------------------------------------------------------------
# SpecAugment
# -------------------------------------------------------------------------
def spec_augment_batch(
    x: torch.Tensor,
    time_mask_param: int = 20,
    freq_mask_param: int = 8,
    num_masks: int = 1,
) -> torch.Tensor:
    """
    Simple in-batch SpecAugment on mel spectrograms.

    x: (B, T, F)
    """
    x = x.clone()
    B, T, F = x.shape

    for b in range(B):
        for _ in range(num_masks):
            # Time mask
            t = random.randint(0, time_mask_param)
            if t > 0 and T - t > 0:
                t0 = random.randint(0, T - t)
                x[b, t0:t0 + t, :] = 0.0

            # Freq mask
            f = random.randint(0, freq_mask_param)
            if f > 0 and F - f > 0:
                f0 = random.randint(0, F - f)
                x[b, :, f0:f0 + f] = 0.0

    return x


# -------------------------------------------------------------------------
# Mixup (with soft labels)
# -------------------------------------------------------------------------
def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float().to(device)


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    alpha: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup on inputs + labels.

    x: (B, T, F)
    y: (B,) int64 labels (0..num_classes-1)

    Returns:
      mixed_x: (B, T, F)
      mixed_y: (B, C) soft labels
    """
    if alpha <= 0.0:
        # Just return (optionally) smoothed one-hot labels
        y_oh = one_hot(y, num_classes, x.device)
        if label_smoothing > 0.0:
            y_oh = (1 - label_smoothing) * y_oh + label_smoothing / num_classes
        return x, y_oh

    lam = np.random.beta(alpha, alpha)
    B = x.size(0)
    index = torch.randperm(B, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]

    y1 = one_hot(y, num_classes, x.device)
    y2 = y1[index]

    if label_smoothing > 0.0:
        y1 = (1 - label_smoothing) * y1 + label_smoothing / num_classes
        y2 = (1 - label_smoothing) * y2 + label_smoothing / num_classes

    mixed_y = lam * y1 + (1.0 - lam) * y2
    return mixed_x, mixed_y


# -------------------------------------------------------------------------
# Focal Loss that supports hard or soft labels
# -------------------------------------------------------------------------
class FocalLoss(torch.nn.Module):
    """
    Class-weighted focal loss.

    - Supports hard labels: target shape (B,)
    - Supports soft labels (e.g. Mixup): target shape (B, C)
    - alpha: optional class weights (tensor of shape (C,))
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 1.5,
        reduction: str = "mean",
    ):
        super().__init__()
        # Store alpha as a buffer so it moves with the model
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)
        target:
          - (B,) int64
          - or (B, C) float (soft)
        """
        # Compute loss in float32 for stability
        logits = logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        B, C = logits.shape

        if target.dim() == 1:
            # Hard labels → one-hot
            target_oh = F.one_hot(target, num_classes=C).float()
        else:
            target_oh = target  # soft labels

        # p_t = sum over classes of (p * y)
        pt = (probs * target_oh).sum(dim=-1).clamp(min=1e-7, max=1.0)

        # CE = - sum(y * log p)
        ce = -(target_oh * log_probs).sum(dim=-1)

        # Focal modulation
        focal_factor = (1.0 - pt) ** self.gamma
        loss = focal_factor * ce

        # Class weights (alpha)
        if self.alpha is not None:
            alpha_vec = self.alpha.to(logits.device, dtype=torch.float32)
            alpha_t = (alpha_vec * target_oh).sum(dim=-1)  # per-sample weight
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# -------------------------------------------------------------------------
# Class weights from train.csv
# -------------------------------------------------------------------------
def compute_class_weights(train_csv: Path, num_classes: int = 5) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    raw_labels = df["label"].values
    labels = raw_labels - 1  # SAND labels 1..5 → 0..4

    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (num_classes * counts)

    logger.info(f"Class counts: {counts}, class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


# -------------------------------------------------------------------------
# Optionally freeze early ElasticAST layers
# -------------------------------------------------------------------------
def freeze_backbone_layers(
    model: ElasticASTForAudioClassification,
    num_blocks_to_freeze: int = 4,
) -> None:
    """
    Tries to freeze the first N transformer blocks of ElasticAST.
    Safe no-op if structure doesn't match expectations.
    """
    enc = getattr(model, "encoder", None)
    if enc is None:
        logger.warning("No encoder attribute on model; skipping freezing.")
        return

    blocks = getattr(enc, "blocks", None)
    if blocks is None:
        logger.warning("No `blocks` on encoder; skipping freezing.")
        return

    # ElasticAST uses CustomSequential(modules_list=...)
    modules_list = getattr(blocks, "modules_list", None)
    if modules_list is None:
        layers = list(blocks.children())
    else:
        layers = list(modules_list)

    if not layers:
        logger.warning("Encoder blocks empty; skipping freezing.")
        return

    n_freeze = min(num_blocks_to_freeze, len(layers))
    logger.info(f"Freezing first {n_freeze} ElasticAST blocks (backbone).")

    for layer in layers[:n_freeze]:
        for p in layer.parameters():
            p.requires_grad = False


# -------------------------------------------------------------------------
# CLI TRAIN ENTRYPOINT
# -------------------------------------------------------------------------
@app.command()
def fit(
    platform: str = typer.Option("auto", help="Hardware preset (unused, for future)."),
    use_specaugment: bool = typer.Option(True, help="Apply SpecAugment."),
    use_mixup: bool = typer.Option(True, help="Apply Mixup."),
    mixup_alpha: float = typer.Option(0.1, help="Beta alpha for Mixup."),
    label_smoothing: float = typer.Option(0.0, help="Label smoothing for hard labels."),
    freeze_blocks: int = typer.Option(0, help="Number of ElasticAST backbone blocks to freeze."),
):
    """
    Train ElasticAST on the SAND dataset with:

    - Class-weighted focal loss
    - SpecAugment
    - Mixup (soft labels)
    - Cosine LR with warmup
    - Gradient clipping
    - Optional backbone freezing
    """

    logger.info("========== TRAINING START ==========")

    # ---------------------------------------------------------------------
    # Device & AMP
    # ---------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")

    # ---------------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------------
    cfg = get_training_config(platform)
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["learning_rate"]
    warmup_ratio = cfg["warmup_ratio"]
    num_workers = cfg["num_workers"]

    logger.info(f"Training config: {cfg}")

    # ---------------------------------------------------------------------
    # Datasets & DataLoaders
    # ---------------------------------------------------------------------
    train_csv = INTERIM_DATA_DIR / "train.csv"
    val_csv = INTERIM_DATA_DIR / "val.csv"

    train_dataset = SANDDataset(
        audio_root=RAW_DATA_DIR,
        metadata_csv=train_csv,
    )
    val_dataset = SANDDataset(
        audio_root=RAW_DATA_DIR,
        metadata_csv=val_csv,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_mels,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_mels,
    )

    logger.info(f"Loaded dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    model = ElasticASTForAudioClassification(num_labels=5).to(device)

    # Lazy-init encoder using first batch (ElasticAST needs real shape)
    init_batch = next(iter(train_loader))
    with torch.no_grad():
        _ = model(init_batch["input_values"].to(device))
    logger.info("ElasticAST encoder initialized from first batch.")

    # Optionally freeze early backbone blocks
    if freeze_blocks > 0:
        freeze_backbone_layers(model, num_blocks_to_freeze=freeze_blocks)

    # ---------------------------------------------------------------------
    # Loss (Class-weighted Focal Loss)
    # ---------------------------------------------------------------------
    class_weights = compute_class_weights(train_csv, num_classes=5).to(device)
    focal_loss = FocalLoss(alpha=class_weights, gamma=1.5, reduction="mean")

    # ---------------------------------------------------------------------
    # Optimizer + Scheduler
    # ---------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    warmup_steps = max(warmup_steps, 1)  # avoid zero

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,  # tiny start (must be > 0)
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # AMP scaler
    scaler = GradScaler() if device.type == "cuda" else None

    # ---------------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------------
    model_dir = MODELS_DIR / "elasticast_sand"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0

    # =====================================================================
    # TRAINING LOOP
    # =====================================================================
    for epoch in range(1, num_epochs + 1):
        logger.info(f"----- EPOCH {epoch}/{num_epochs} -----")

        # ---------------------------
        # TRAIN
        # ---------------------------
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)  # (B,) ints 0..4

            # SpecAugment first (on spectrograms)
            if use_specaugment:
                x = spec_augment_batch(x)

            # Mixup (produces soft labels)
            if use_mixup:
                x, y_soft = mixup_batch(
                    x,
                    y,
                    num_classes=5,
                    alpha=mixup_alpha,
                    label_smoothing=label_smoothing,
                )
            else:
                if label_smoothing > 0.0:
                    y_soft = one_hot(y, num_classes=5, device=device)
                    y_soft = (1 - label_smoothing) * y_soft + label_smoothing / 5
                else:
                    y_soft = None

            optimizer.zero_grad()

            if scaler:
                # AMP forward
                with autocast():
                    outputs = model(x)
                    logits = outputs.logits  # may be fp16

                # Loss in fp32
                if y_soft is not None:
                    loss = focal_loss(logits, y_soft)
                else:
                    loss = focal_loss(logits, y)

                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x)
                logits = outputs.logits
                if y_soft is not None:
                    loss = focal_loss(logits, y_soft)
                else:
                    loss = focal_loss(logits, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))
        logger.info(f"[Train] Loss: {avg_train_loss:.4f}")

        # ---------------------------
        # VALIDATION
        # ---------------------------
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_values"].to(device)
                y = batch["labels"].to(device)

                outputs = model(x)
                logits = outputs.logits.float()  # ensure fp32 for metrics

                # Use hard labels in validation
                val_loss = focal_loss(logits, y)
                val_losses.append(val_loss.item())

                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y.cpu().numpy())

        avg_val_loss = float(np.mean(val_losses))
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="weighted")

        logger.info(
            f"[Val] Loss: {avg_val_loss:.4f}, "
            f"Acc: {val_acc:.4f}, F1 (weighted): {val_f1:.4f}"
        )

        # Save best model by F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            logger.info(f"✔ Saved BEST model (F1={val_f1:.4f}) → {save_path}")

    logger.info("========== TRAINING COMPLETE ==========")