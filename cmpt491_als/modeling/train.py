import typer
from loguru import logger
import torch
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor

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
# Load Feature Extractor (HuggingFace)
# --------------------------------------------------------------
def load_feature_extractor(model_name: str):
    """
    Loads an ASTFeatureExtractor from HuggingFace.
    This defines the STFT, mel-spec, and normalization used.
    """
    logger.info(f"Loading ASTFeatureExtractor for: {model_name}")
    return ASTFeatureExtractor.from_pretrained(
        model_name,
        trust_remote_code=True
    )


# --------------------------------------------------------------
# Load Local ElasticAST Model
# --------------------------------------------------------------
def load_model(num_labels: int = 5):
    """
    Load the local ElasticAST model (NOT from HuggingFace).
    """
    logger.info("Initializing local ElasticAST model...")
    return ElasticASTForAudioClassification(num_labels=num_labels)


# --------------------------------------------------------------
# Create Dataloaders
# --------------------------------------------------------------
def create_dataloaders(
    feature_extractor,
    batch_size: int,
    num_workers: int,
):
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

    logger.info(f"Loaded dataset: {len(train_dataset)} train, {len(val_dataset)} val")

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

    return train_loader, val_loader


# --------------------------------------------------------------
# TRAIN COMMAND
# --------------------------------------------------------------
@app.command("fit")
def fit_command(
    model_name: str = typer.Option(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        help="HF model whose feature extractor defines mel-spec parameters.",
    ),
    platform: str = typer.Option(
        "auto",
        help="Hardware preset: auto, a100, m2, etc.",
    ),
):
    logger.info("========== TRAINING START ==========")

    # Device ------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Config ------------------------------------------------------
    cfg = get_training_config(platform)
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["learning_rate"]
    num_workers = cfg["num_workers"]

    logger.info(f"Training config: {cfg}")

    # Feature extractor -------------------------------------------
    feature_extractor = load_feature_extractor(model_name)

    # Model --------------------------------------------------------
    model = load_model(num_labels=5).to(device)

    # ==============================================================
    # Create Dataloaders BEFORE optimizer
    # ==============================================================
    train_loader, val_loader = create_dataloaders(
        feature_extractor=feature_extractor,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ==============================================================
    # FORCE-LAZY-INIT ElasticAST USING FIRST REAL BATCH
    # ==============================================================
    init_batch = next(iter(train_loader))

    with torch.no_grad():
        x = init_batch["input_values"].to(device)
        _ = model(x)  # triggers building ElasticAST encoder with real (T,F)

    logger.info("ElasticAST encoder initialized from real batch.")

    # ==============================================================
    # Now encoder exists — create the optimizer
    # ==============================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Output directory
    model_dir = MODELS_DIR / "elasticast_sand"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================
    # TRAINING LOOP
    # ==============================================================
    from torch.cuda.amp import autocast, GradScaler
    from sklearn.metrics import accuracy_score, f1_score

    scaler = GradScaler() if device.type == "cuda" else None

    # Scheduler (cosine decay with warmup)
    warmup_steps = 100
    total_steps = len(train_loader) * num_epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
    )

    best_val_f1 = 0.0
    global_step = 0

    logger.info("========== BEGIN TRAINING ==========")

    for epoch in range(1, num_epochs + 1):

        model.train()
        train_losses = []

        for batch in train_loader:

            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)

            # 1. Forward pass with AMP
            with autocast(enabled=(scaler is not None)):
                outputs = model(x, labels=y)
                loss = outputs.loss / cfg["gradient_accumulation_steps"]

            # 2. Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 3. Step optimizer every N steps (accumulation)
            if (global_step + 1) % cfg["gradient_accumulation_steps"] == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

                # LR warmup then cosine decay
                if global_step < warmup_steps:
                    lr_scale = float(global_step) / float(max(1, warmup_steps))
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr * lr_scale
                else:
                    scheduler.step()

            train_losses.append(loss.item())
            global_step += 1

        # ------------------------------------------
        # VALIDATION
        # ------------------------------------------
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_values"].to(device)
                y = batch["labels"].to(device)

                outputs = model(x, labels=y)
                val_losses.append(outputs.loss.item())

                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y.cpu().numpy())

        val_loss = sum(val_losses) / len(val_losses)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="weighted")

        logger.info(f"[Epoch {epoch}] Train Loss: {sum(train_losses)/len(train_losses):.4f}")
        logger.info(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # ------------------------------------------
        # SAVE BEST MODEL
        # ------------------------------------------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            logger.info(f"✔ Saved new BEST model (F1={val_f1:.4f}) → {save_path}")

    logger.info("========== TRAINING COMPLETE ==========")