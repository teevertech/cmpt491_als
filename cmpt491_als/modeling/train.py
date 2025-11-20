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


# ============================================================
# Simple SpecAugment
# ============================================================
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
            # time mask
            t = random.randint(0, time_mask_param)
            if t > 0 and T - t > 0:
                t0 = random.randint(0, T - t)
                x[b, t0:t0 + t, :] = 0.0

            # freq mask
            f = random.randint(0, freq_mask_param)
            if f > 0 and F - f > 0:
                f0 = random.randint(0, F - f)
                x[b, :, f0:f0 + f] = 0.0

    return x


# ============================================================
# Compute class weights
# ============================================================
def compute_class_weights(train_csv: Path, num_classes: int = 5) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    labels = df["label"].values - 1  # dataset converts to 0–4

    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()

    weights = total / (num_classes * counts)
    logger.info(f"Class counts: {counts}, weights: {weights}")

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# TRAIN COMMAND
# ============================================================
@app.command()
def fit(
    platform: str = typer.Option("auto"),
    use_specaugment: bool = typer.Option(True),
):

    logger.info("========== TRAINING START ==========")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Config
    cfg = get_training_config(platform)
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["learning_rate"]
    num_workers = cfg["num_workers"]
    warmup_ratio = cfg["warmup_ratio"]

    logger.info(f"Training config: {cfg}")

    # Class weights
    train_csv = INTERIM_DATA_DIR / "train.csv"
    class_weights = compute_class_weights(train_csv).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Model
    model = ElasticASTForAudioClassification(num_labels=5).to(device)

    # Data
    train_dataset = SANDDataset(RAW_DATA_DIR, train_csv)
    val_dataset = SANDDataset(
        RAW_DATA_DIR, INTERIM_DATA_DIR / "val.csv"
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

    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val")

    # Lazy init model forward
    init_batch = next(iter(train_loader))
    with torch.no_grad():
        _ = model(init_batch["input_values"].to(device))
    logger.info("ElasticAST encoder lazy initialization complete.")

    # Optimizer + AMP scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Scheduler (warmup + cosine)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.0,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
    )

    model_dir = MODELS_DIR / "elasticast_sand"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    global_step = 0

    # ============================================================
    # TRAIN LOOP
    # ============================================================
    for epoch in range(1, num_epochs + 1):
        logger.info(f"===== EPOCH {epoch}/{num_epochs} =====")
        model.train()

        train_losses = []

        for batch in train_loader:
            global_step += 1

            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)

            if use_specaugment:
                x = spec_augment_batch(x)

            optimizer.zero_grad()

            # AUTOCOMPAT — model internally uses autocast
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(x)
                logits = outputs.logits

            # Convert logits to float32 OUTSIDE autocast
            logits = logits.float()

            loss = loss_fn(logits, y)

            # AMP + clipping
            if scaler.is_enabled():
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses.append(loss.item())

            # Scheduler step
            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

        avg_train_loss = float(np.mean(train_losses))
        logger.info(f"[Train] Loss: {avg_train_loss:.4f}")

        # ============================================================
        # VALIDATION
        # ============================================================
        model.eval()
        val_losses = []
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_values"].to(device)
                y = batch["labels"].to(device)

                with autocast(device_type="cuda", enabled=False):
                    logits = model(x).logits.float()

                loss = loss_fn(logits, y)
                val_losses.append(loss.item())

                preds = logits.argmax(dim=-1).cpu().numpy()
                preds_list.extend(preds)
                targets_list.extend(y.cpu().numpy())

        avg_val_loss = float(np.mean(val_losses))
        val_acc = accuracy_score(targets_list, preds_list)
        val_f1 = f1_score(targets_list, preds_list, average="weighted")

        logger.info(
            f"[Val] Loss={avg_val_loss:.4f} "
            f"Acc={val_acc:.4f} F1={val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            logger.info(f"✔ Saved BEST model (F1={val_f1:.4f}) → {save_path}")

    logger.info("========== TRAINING COMPLETE ==========")