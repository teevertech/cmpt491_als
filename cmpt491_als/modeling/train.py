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
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"----- EPOCH {epoch}/{num_epochs} -----")

        # -----------------------
        # TRAIN
        # -----------------------
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(x, labels=y)
                loss = outputs.loss

            if scaler:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"[Train] Loss: {avg_train_loss:.4f}")

        # -----------------------
        # VALIDATE
        # -----------------------
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_values"].to(device)
                y = batch["labels"].to(device)

                outputs = model(x, labels=y)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"[Val] Loss: {avg_val_loss:.4f}")

        # -----------------------
        # SAVE BEST MODEL
        # -----------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model → {save_path}")

    logger.info("========== TRAINING COMPLETE ==========")