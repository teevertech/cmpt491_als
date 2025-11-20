from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import ASTForAudioClassification, ASTFeatureExtractor
from .sand_datasets import SANDDataset
from cmpt491_als.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    get_training_config,
    MODEL_NAMES
)

app = typer.Typer()


def train_epoch(model, dataloader, optimizer, device, scaler=None, gradient_accumulation_steps=1):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move to device
        input_values = batch['input_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # Mixed precision forward pass
        if scaler is not None:  # CUDA with mixed precision
            with autocast():
                outputs = model(input_values=input_values, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps

            # Mixed precision backward pass
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:  # Regular precision (MPS/CPU)
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Track metrics
        total_loss += loss.item() * gradient_accumulation_steps
        preds = torch.argmax(outputs.logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Handle remaining gradients
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate_epoch(model, dataloader, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Collect for F1 calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    # Calculate F1-score (macro average for multi-class)
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, f1


@app.command()
def train(
    model_name: str = "ast",
    platform: str = "auto",  # "m2", "a100", or "auto"
    device: str = "auto",
    num_epochs: int = None,   # Override from config if needed
    batch_size: int = None,   # Override from config if needed
):
    """Fine-tune AST model with platform-optimized settings."""

    # Get platform-specific configuration
    config = get_training_config(platform)

    # Allow command-line overrides
    if num_epochs is not None:
        config = config.copy()
        config["num_epochs"] = num_epochs
    if batch_size is not None:
        config = config.copy()
        config["batch_size"] = batch_size

    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.warning("Using MPS - if crashes occur, try --platform m2")
        else:
            device = "cpu"

    logger.info(f"Platform: {platform}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

    # Get paths
    audio_root = RAW_DATA_DIR               # e.g. data/raw

    # Use stratified splits created by build_interim_csv.py
    train_metadata_path = INTERIM_DATA_DIR / "train.csv"
    val_metadata_path = INTERIM_DATA_DIR / "val.csv"

    model_output_dir = MODELS_DIR / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Get pretrained model name from config
    pretrained_model_name = MODEL_NAMES[model_name]

    # Load model with memory optimization
    logger.info(f"Loading pretrained model: {pretrained_model_name}")
    try:

        # ------------------------------------------------------------------
        # 🆕 ELASTIC AST BRANCH
        # ------------------------------------------------------------------
        if model_name == "elastic_ast":
            from cmpt491_als.modeling.elastic_ast_wrapper import ElasticASTForAudioClassification

            feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

            class_names = ['ALS-1', 'ALS-2', 'ALS-3', 'ALS-4', 'Healthy']
            num_labels = len(class_names)
            id2label = {i: name for i, name in enumerate(class_names)}
            label2id = {name: i for i, name in enumerate(class_names)}

            model = ElasticASTForAudioClassification(num_labels=num_labels).to(device)

            logger.info("ElasticAST loaded successfully with AST feature extractor.")

        # ------------------------------------------------------------------
        # EXISTING AST (HuggingFace) MODEL BRANCH
        # ------------------------------------------------------------------
        else:
            feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model_name)
            logger.info("Feature extractor loaded successfully")

            # Define proper class names
            class_names = ['ALS-1', 'ALS-2', 'ALS-3', 'ALS-4', 'Healthy']
            id2label = {i: name for i, name in enumerate(class_names)}
            label2id = {name: i for i, name in enumerate(class_names)}

            model = ASTForAudioClassification.from_pretrained(
                pretrained_model_name,
                num_labels=5,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.float32,  # Explicit dtype for stability
                id2label=id2label,
                label2id=label2id
            )
            logger.info("Model loaded successfully with proper class names")

            # Memory optimization for M2
            if platform == "m2" or device == "mps":
                if hasattr(model.config, 'use_memory_efficient_attention'):
                    model.config.use_memory_efficient_attention = True

            model = model.to(device)
            logger.info(f"Model moved to {device}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Create datasets
    # Create datasets
    try:
        train_dataset = SANDDataset(
            audio_root=audio_root,
            metadata_csv=train_metadata_path,
            feature_extractor=feature_extractor
        )

        val_dataset = SANDDataset(
            audio_root=audio_root,
            metadata_csv=val_metadata_path,
            feature_extractor=feature_extractor
        )

        logger.info(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset)}")

    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        return

    # Create data loaders with platform-specific settings
    dataloader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": (device == "cuda"),
        "persistent_workers": True if config["num_workers"] > 0 else False
    }

    # A100 optimizations
    if platform == "a100":
        dataloader_kwargs.update({
            "prefetch_factor": 2,  # Prefetch batches
            "drop_last": True,     # Consistent batch sizes
        })

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    # Setup training with optimizations
    use_amp = (device == "cuda")  # Use mixed precision on CUDA
    scaler = GradScaler() if use_amp else None

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
        eps=1e-8  # Better numerical stability
    )

    # Learning rate scheduler for A100
    if platform == "a100":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"], eta_min=1e-7
        )
    else:
        scheduler = None

    # Compile model for A100 (PyTorch 2.0+)
    if platform == "a100" and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("Model compiled for better performance")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")

    # Training loop
    best_val_f1 = 0
    logger.info(f"Starting training: {config['num_epochs']} epochs, batch_size={config['batch_size']}")
    logger.info(f"Mixed precision: {use_amp}, Gradient accumulation: {config['gradient_accumulation_steps']}")
    logger.info("Using F1-score for model selection (competition metric)")

    try:
        for epoch in range(config["num_epochs"]):
            logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

            # Train with optimizations
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device,
                scaler, config["gradient_accumulation_steps"]
            )

            # Validate (now returns F1-score too)
            val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, device)

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"Learning rate: {current_lr:.2e}")

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            # Save best model based on F1-score (competition metric)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model_path = model_output_dir / "best_model"

                # Save the original model (unwrap if compiled)
                model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
                model_to_save.save_pretrained(model_path)
                feature_extractor.save_pretrained(model_path)
                logger.info(f"New best F1-score: {best_val_f1:.4f}")

            # Memory cleanup
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

        logger.success(f"Training complete! Best F1-score: {best_val_f1:.4f}")

    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    app() # train.py
