import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger

from cmpt491_als.modeling.sand_datasets import SANDDataset
from cmpt491_als.modeling.elastic_ast_wrapper import ElasticASTForAudioClassification
from cmpt491_als.modeling.collate import pad_mels
from cmpt491_als.config import RAW_DATA_DIR, MODELS_DIR


def make_submission(test_xlsx: str, output_csv: str = "results.csv"):
    """
    Generate SAND challenge submission file.
    Output format:
        ID,CLASS
        ID001,4
        ID002,1
        ...
    """

    logger.info("===== LOADING TEST XLSX =====")
    df = pd.read_excel(test_xlsx)

    required_cols = {"filepath", "ID"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Test XLSX must contain columns: {required_cols}")

    # Convert to temp CSV because SANDDataset expects CSV input
    temp_csv = "temp_test.csv"
    df.to_csv(temp_csv, index=False)

    logger.info("===== BUILDING TEST DATASET =====")
    test_ds = SANDDataset(
        audio_root=RAW_DATA_DIR,
        metadata_csv=temp_csv,
        is_test=True,     # <-- IMPORTANT: disable label loading
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=pad_mels,
    )

    logger.info("===== LOADING TRAINED MODEL =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ElasticASTForAudioClassification(num_labels=5).to(device)
    ckpt_path = MODELS_DIR / "elasticast_sand" / "best_model.pt"

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    logger.info(f"Loaded: {ckpt_path}")

    # Inference
    all_ids = []
    all_preds = []

    logger.info("===== RUNNING INFERENCE =====")
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_values"].to(device)
            ids = batch["ids"]

            outputs = model(x)
            logits = outputs.logits.float()

            preds = logits.argmax(dim=-1).cpu().tolist()  # 0–4

            all_ids.extend(ids)
            all_preds.extend(preds)

    # Convert predictions back to CLASS 1–5
    class_labels = [p + 1 for p in all_preds]

    # Build submission DataFrame
    submission = pd.DataFrame({
        "ID": all_ids,
        "CLASS": class_labels
    })

    # Ensure exactly the required column order
    submission = submission[["ID", "CLASS"]]

    submission.to_csv(output_csv, index=False)

    logger.info(f"===== SUBMISSION SAVED → {output_csv} =====")


if __name__ == "__main__":
    make_submission("sand_task_1_test.xlsx", "results.csv")