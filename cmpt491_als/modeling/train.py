import typer
from loguru import logger
import torch
from torch.utils.data import DataLoader

from cmpt491_als.modeling.sand_datasets import SANDDataset
from cmpt491_als.modeling.elastic_ast_wrapper import ElasticASTForAudioClassification

from cmpt491_als.config import (
    MODELS_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    get_training_config,
)

app = typer.Typer(no_args_is_help=True)

def load_model(num_labels: int = 5):
    """
    Instantiate ElasticAST from the local GitHub repo via our wrapper.
    """
    logger.info("Initializing ElasticASTForAudioClassification (local repo)")
    return ElasticASTForAudioClassification(num_labels=num_labels)


def create_dataloaders(
    batch_size: int,
    num_workers: int,
    target_sr: int = 16000,
    n_mels: int = 128,
):
    """
    Create dataset + dataloaders for train/val using on-the-fly log-Mel features.
    """
    train_csv = INTERIM_DATA_DIR / "train.csv"
    val_csv = INTERIM_DATA_DIR / "val.csv"

    train_dataset = SANDDataset(
        audio_root=RAW_DATA_DIR,
        metadata_csv=train_csv,
        target_sr=target_sr,
        n_mels=n_mels,
    )

    val_dataset = SANDDataset(
        audio_root=RAW_DATA_DIR,
        metadata_csv=val_csv,
        target_sr=target_sr,
        n_mels=n_mels,
    )

    logger.info(
        f"Loaded dataset: {len(train_dataset)} train, {len(val_dataset)} val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


@app.command("fit")
def fit_command(
    platform: str = typer.Option(
        "auto",
        help="Hardware preset: auto, a100, m2, etc.",
    ),
):
    """
    Train ElasticAST on the SAND dataset.
    """
    logger.info("========== TRAINING START ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cfg = get_training_config(platform)
    num_epochs = cfg["num_epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["learning_rate"]
    num_workers = cfg["num_workers"]

    logger.info(f"Training config: {cfg}")

    # Model --------------------------------------------------------
    model = load_model(num_labels=5).to(device)
    with torch.no_grad():
        dummy = torch.randn(1, 128, 128).to(device)  # (B, T, F) where F = n_mels = 128
        model(dummy)

    # Dataloaders --------------------------------------------------
    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    model_dir = MODELS_DIR / "elasticast_local"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"----- EPOCH {epoch}/{num_epochs} -----")

        # ----------------------- Train ---------------------------
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            x = batch["input_values"].to(device)  # (B, T, F)
            y = batch["labels"].to(device)        # (B,)

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

        # ----------------------- Val -----------------------------
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_values"].to(device)
                y = batch["labels"].to(device)

                outputs = model(x, labels=y)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"[Val]   Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model → {save_path}")

    logger.info("========== TRAINING COMPLETE ==========")