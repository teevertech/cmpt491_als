import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import amp                      # NEW autocast API
from torch.cuda.amp import GradScaler
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


# -----------------------------------------------------------------------------
# SpecAugment
# -----------------------------------------------------------------------------
def spec_augment_batch(
    x: torch.Tensor,
    time_mask_param: int = 40,
    freq_mask_param: int = 15,
    num_masks: int = 2,
) -> torch.Tensor:
    """Simple in-batch SpecAugment."""
    x = x.clone()
    B, T, F = x.shape

    for b in range(B):
        for _ in range(num_masks):
            # time mask
            t = random.randint(0, time_mask_param)
            if t > 0 and T - t > 0:
                t0 = random.randint(0, T - t)
                x[b, t0:t0+t] = 0

            # freq mask
            f = random.randint(0, freq_mask_param)
            if f > 0 and F - f > 0:
                f0 = random.randint(0, F - f)
                x[b, :, f0:f0+f] = 0

    return x


# -----------------------------------------------------------------------------
# Class weights
# -----------------------------------------------------------------------------
def compute_class_weights(train_csv: Path, num_classes=5):
    df = pd.read_csv(train_csv)
    labels = df["label"].values - 1        # convert to 0–4

    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()

    weights = total / (num_classes * counts)
    logger.info(f"Class counts: {counts}, weights: {weights}")

    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------------------------------------------------------
# TRAINING ENTRYPOINT
# -----------------------------------------------------------------------------
@app.command()
def fit(
    use_specaugment: bool = typer.Option(True, help="Enable SpecAugment"),
    platform: str = typer.Option("auto")
):
    logger.info("======== TRAINING START ========")

    # -----------------------------------------------------------------------------
    # Device: BF16 on A100, else FP16
    # -----------------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        bf16_ok = torch.cuda.is_bf16_supported()

        if bf16_ok:
            amp_dtype = torch.bfloat16
            logger.info("Using BF16 autocast (A100 supported)")
        else:
            amp_dtype = torch.float16
            logger.info("Using FP16 autocast (BF16 unsupported)")
    else:
        device = torch.device("cpu")
        amp_dtype = None

    logger.info(f"Device: {device}")

    # -----------------------------------------------------------------------------
    # CONFIG
    # -----------------------------------------------------------------------------
    cfg = get_training_config(platform)
    batch_size    = cfg["batch_size"]
    num_epochs    = cfg["num_epochs"]
    lr            = cfg["learning_rate"]
    warmup_ratio   = cfg["warmup_ratio"]
    num_workers   = cfg["num_workers"]

    logger.info(f"Training config: {cfg}")

    # -----------------------------------------------------------------------------
    # CLASS WEIGHTS
    # -----------------------------------------------------------------------------
    train_csv = INTERIM_DATA_DIR / "train.csv"
    class_weights = compute_class_weights(train_csv).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # -----------------------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------------------
    model = ElasticASTForAudioClassification(num_labels=5).to(device)

    # -----------------------------------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------------------------------
    train_dataset = SANDDataset(
        audio_root=RAW_DATA_DIR,
        metadata_csv=train_csv,
    )

    val_csv = INTERIM_DATA_DIR / "val.csv"
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

    logger.info(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val samples.")

    # -----------------------------------------------------------------------------
    # LAZY INITIALIZATION (ElasticAST requirement)
    # -----------------------------------------------------------------------------
    init_batch = next(iter(train_loader))
    with torch.no_grad():
        _ = model(init_batch["input_values"].to(device))
    logger.info("ElasticAST encoder initialized.")

    # -----------------------------------------------------------------------------
    # OPTIMIZER + LR SCHEDULER
    # -----------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)

    # Linear warmup → Cosine decay
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8,        # tiny LR start
                end_factor=1.0,
                total_iters=warmup_steps,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps],
    )

    # -----------------------------------------------------------------------------
    # AMP
    # -----------------------------------------------------------------------------
    scaler = GradScaler() if device.type == "cuda" else None

    # -----------------------------------------------------------------------------
    # OUTPUT FOLDER
    # -----------------------------------------------------------------------------
    model_dir = MODELS_DIR / "elasticast_sand"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0

    # =============================================================================
    # TRAINING LOOP
    # =============================================================================
    for epoch in range(1, num_epochs + 1):
        logger.info(f"----- EPOCH {epoch}/{num_epochs} -----")

        # ----------------------------
        # TRAIN
        # ----------------------------
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)

            if use_specaugment:
                x = spec_augment_batch(x)

            optimizer.zero_grad()

            #
            # AMP forward — supports bfloat16 on A100
            #
            if scaler:
                with amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(x)
                    logits = outputs.logits.float()  # force FP32 for stable loss
                    loss = loss_fn(logits, y)

                scaler.scale(loss).backward()

                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

            else:
                outputs = model(x)
                logits = outputs.logits
                loss = loss_fn(logits, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            train_losses.append(loss.item())

        logger.info(f"[Train] Loss: {np.mean(train_losses):.4f}")

        # ----------------------------
        # VALIDATION
        # ----------------------------
        model.eval()
        val_losses, preds, targs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_values"].to(device)
                y = batch["labels"].to(device)

                outputs = model(x)
                logits = outputs.logits.float()

                loss = loss_fn(logits, y)
                val_losses.append(loss.item())

                pred = logits.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())
                targs.extend(y.cpu().numpy())

        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(targs, preds)
        val_f1  = f1_score(targs, preds, average="weighted")

        logger.info(f"[Val] Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), path)
            logger.info(f"✔ Saved BEST model → {path}")

    logger.info("======== TRAINING COMPLETE ========")