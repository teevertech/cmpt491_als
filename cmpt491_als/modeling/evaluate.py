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


def load_model(path: str, num_labels: int = 5, device="cuda", init_batch=None):
    """
    Loads ElasticAST model and forces encoder initialization before loading weights.
    """
    model = ElasticASTForAudioClassification(num_labels=num_labels).to(device)
    model.eval()

    # ---- FORCE encoder initialization using real sample shape ----
    if init_batch is None:
        raise ValueError("init_batch must be provided to initialize encoder.")

    with torch.no_grad():
        x = init_batch["input_values"].to(device)
        _ = model(x)   # builds encoder with correct sample_size=(F,T)

    # ---- Load weights now that encoder exists ----
    logger.info(f"Loading model weights from: {path}")
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd, strict=True)

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

    # 1. Paths ----------------------------------------------------
    ckpt_path = MODELS_DIR / checkpoint_name
    test_csv = INTERIM_DATA_DIR / "test.csv"

    # 2. Load test dataset ----------------------------------------
    test_dataset = SANDDataset(
        audio_root=RAW_DATA_DIR,
        metadata_csv=test_csv
    )

    # 3. Create test DataLoader BEFORE model initialization -------
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=pad_mels,
    )

    # 4. Grab first batch to initialize encoder -------------------
    init_batch = next(iter(test_loader))

    # 5. Load model with lazy encoder init ------------------------
    model = ElasticASTForAudioClassification(num_labels=5).to(device)
    model.eval()

    # ---- initialize encoder from real sample ----
    with torch.no_grad():
        x = init_batch["input_values"].to(device)
        _ = model(x)  # builds encoder with correct shape

    # ---- now load weights ----
    logger.info(f"Loading model weights from: {ckpt_path}")
    state_dict = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # 6. Run inference on all test samples ------------------------
    all_preds = []
    all_labels = []

    logger.info("Running inference on test set...")

    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_values"].to(device)
            y = batch["labels"].item()

            outputs = model(x)
            logits = outputs.logits.cpu()
            pred = torch.argmax(logits, dim=-1).item()

            all_preds.append(pred)
            all_labels.append(y)

    # 7. Confusion matrix + report -------------------------------
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_labels, all_preds)

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4))

    labels = [0, 1, 2, 3, 4]

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

     # ------------------------------------------------------------
    # 8. Generate results.csv submission file
    # ------------------------------------------------------------
    import pandas as pd
    from pathlib import Path

    # Load the original test metadata to get IDs
    test_df = pd.read_csv(test_csv)

    # Your predictions are 0–4 → convert to 1–5
    preds_1_to_5 = [p + 1 for p in all_preds]

    # Build submission DataFrame
    submission = pd.DataFrame({
        "ID": test_df["ID"],        # Must match exactly the test set order
        "CLASS": preds_1_to_5       # Must be ints 1–5
    })

    # Save results.csv
    out_path = Path("results.csv")
    submission.to_csv(out_path, index=False)

    logger.info(f"Saved submission CSV → {out_path.absolute()}")
    logger.info("Finished creating results.csv")