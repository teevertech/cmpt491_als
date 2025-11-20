import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
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


# --------------------------------------------------------------
# SpecAugment
# --------------------------------------------------------------
def spec_augment_batch(
    x: torch.Tensor,
    time_mask_param: int = 40,
    freq_mask_param: int = 15,
    num_masks: int = 2,
) -> torch.Tensor:

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


# --------------------------------------------------------------
# Class weights
# --------------------------------------------------------------
def compute_class_weights(train_csv: Path, num_classes: int = 5) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    raw_labels = df["label"].values
    labels = raw_labels - 1

    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()

    weights = total / (num_classes * counts)
    logger.info(f"Class counts: {counts}, class weights: {weights}")

    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------
# Warmup + Cosine scheduler
# --------------------------------------------------------------
def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))

        return max(min_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------
# FIT COMMAND
# --------------------------------------------------------------
@app.command()
def fit(
    platform: str = typer.Option(
        "auto", help="Hardware preset: auto, a100, m2, etc."
    ),
    use_specaugment: bool = typer.Option(
        True, help="Apply SpecAugment during training."
    ),
):
    """
    Train ElasticAST on the SAND dataset with warmup+cosine LR, SpecAugment,
    class-weighted loss, and lazy encoder initialization.
    """

    logger.info("========== TRAINING START ==========")

    # Device ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, cuda.is_available={torch.cuda.is_available()}")

    # Config ------------------------------------------------------
    cfg = get_training_config(platform)
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["learning_rate"]
    num_workers = cfg["num_workers"]

    logger.info(f"Training config: {cfg}")

    # Class weights ----------------------------------------------
    train_csv = INTERIM_DATA_DIR / "train.csv"
    class_weights = compute_class_weights(train_csv, num_classes=5).to(device)
    class_weights = class_weights.float()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Model -------------------------------------------------------
    model = ElasticASTForAudioClassification(num_labels=5).to(device)

    # Datasets ----------------------------------------------------
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

    logger.info(f"Loaded dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    # Lazy init ElasticAST encoder --------------------------------
    init_batch = next(iter(train_loader))
    with torch.no_grad():
        x_init = init_batch["input_values"].to(device)
        _ = model(x_init)
    logger.info("ElasticAST encoder initialized from first batch.")

    # Optimizer ---------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler() if device.type == "cuda" else None

    # Scheduler: warmup + cosine ---------------------------------
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    min_lr = lr / 20

    scheduler = build_warmup_cosine_scheduler(
        optimizer, warmup_steps, total_steps, min_lr
    )

    # Output dir --------------------------------------------------
    model_dir = MODELS_DIR / "elasticast_sand"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0

    # =================================================================
    # TRAINING LOOP
    # =================================================================
    for epoch in range(1, num_epochs + 1):
        logger.info(f"----- EPOCH {epoch}/{num_epochs} -----")

        # ----------------------------
        # TRAIN
        # ----------------------------
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device).long()

            if use_specaugment:
                x = spec_augment_batch(x)

            optimizer.zero_grad()

            # ----- PURE FP32 TRAINING -----
            outputs = model(x)
            logits = outputs.logits      # float32
            loss = loss_fn(logits, y)    # weights also float32

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))
        logger.info(f"[Train] Loss: {avg_train_loss:.4f}")

        # ----------------------------
        # VALIDATE
        # ----------------------------
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_values"].to(device)
                y = batch["labels"].to(device).long()

                outputs = model(x)
                logits = outputs.logits
                loss = loss_fn(logits, y)

                val_losses.append(loss.item())
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y.cpu().numpy())

        avg_val_loss = float(np.mean(val_losses))
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="weighted")

        logger.info(
            f"[Val] Loss: {avg_val_loss:.4f}, "
            f"Acc: {val_acc:.4f}, F1(weighted): {val_f1:.4f}"
        )

        # Save best model --------------------------------------------------------------------------------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            logger.info(f"✔ Saved BEST model (F1={val_f1:.4f}) → {save_path}")

    logger.info("========== TRAINING COMPLETE ==========")