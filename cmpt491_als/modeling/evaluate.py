import torch
from torch.utils.data import DataLoader
from loguru import logger
import typer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

from cmpt491_als.modeling.sand_datasets import SANDDataset
from cmpt491_als.modeling.elastic_ast_wrapper import ElasticASTForAudioClassification
from cmpt491_als.modeling.collate import pad_mels

from cmpt491_als.config import RAW_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR


app = typer.Typer()


def load_model(path: str, num_labels: int = 5, device="cuda"):
    model = ElasticASTForAudioClassification(num_labels=num_labels)
    logger.info(f"Loading model weights from: {path}")
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


@app.command("test")
def evaluate_command(
    checkpoint_name: str = typer.Option(
        "elasticast_sand/best_model.pt",
        help="Model checkpoint inside MODELS_DIR",
    )
):
    """
    Evaluate on test.csv and generate confusion matrix.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_path = MODELS_DIR / checkpoint_name
    model = load_model(str(ckpt_path), device=device)

    # Load test dataset
    test_csv = INTERIM_DATA_DIR / "test.csv"
    test_dataset = SANDDataset(
        audio_root=RAW_DATA_DIR,
        metadata_csv=test_csv
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,           # Evaluate one by one
        shuffle=False,
        num_workers=4,
        collate_fn=pad_mels,
    )

    all_preds = []
    all_labels = []

    logger.info("Running inference on test set...")

    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_values"].to(device)
            y = batch["labels"].cpu().numpy()[0]

            outputs = model(x)
            logits = outputs.logits.cpu()
            pred = torch.argmax(logits, dim=-1).numpy()[0]

            all_preds.append(pred)
            all_labels.append(y)

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4))

    labels = [0, 1, 2, 3, 4]  # Your class labels (0–4 after shifting)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    logger.info("Saved confusion matrix → confusion_matrix.png")
    logger.info("Evaluation complete.")