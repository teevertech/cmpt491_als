"""
Generate the final submission CSV for SAND Task 1.

Usage:
    python -m cmpt491_als.modeling.make_submission
"""

from pathlib import Path
import pandas as pd
import torch
from loguru import logger

from cmpt491_als.modeling.sand_datasets import SANDDataset
from cmpt491_als.modeling.elastic_ast_wrapper import ElasticASTForAudioClassification
from cmpt491_als.modeling.collate import pad_mels
from cmpt491_als.config import RAW_DATA_DIR, MODELS_DIR

from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# LOAD TEST DATA FROM XLSX AND BUILD DATASET
# ---------------------------------------------------------------------
def load_test_metadata(xlsx_path: Path) -> pd.DataFrame:
    """
    Reads the test XLSX provided by the competition and returns
    a DataFrame containing ID and filepath columns.
    """
    df = pd.read_excel(xlsx_path)

    if "ID" not in df.columns or "filepath" not in df.columns:
        raise ValueError(
            f"Test XLSX must contain ID and filepath columns; got: {df.columns.tolist()}"
        )

    logger.info(f"Loaded test metadata: {len(df)} samples")
    return df


# ---------------------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------------------
def predict(model, dataloader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_values"].to(device)
            outputs = model(x)
            logits = outputs.logits.float()
            cls = logits.argmax(dim=-1).cpu().numpy()  # 0–4
            preds.extend(cls)

    return preds


# ---------------------------------------------------------------------
# MAIN SUBMISSION FUNCTION
# ---------------------------------------------------------------------
def make_submission(test_xlsx: str, output_csv: str):
    test_xlsx = Path(test_xlsx)

    df = load_test_metadata(test_xlsx)

    # Build a temporary CSV so SANDDataset can load it
    temp_csv = test_xlsx.parent / "temp_test.csv"
    df.to_csv(temp_csv, index=False)

    # Build dataset (labels ignored)
    test_dataset = SANDDataset(
        audio_root=RAW_DATA_DIR,
        metadata_csv=temp_csv
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=pad_mels
    )

    # Load model
    model_path = MODELS_DIR / "elasticast_sand" / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained model at: {model_path}"
        )

    logger.info(f"Loading model from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ElasticASTForAudioClassification(num_labels=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Predict
    logger.info("Running inference...")
    preds = predict(model, test_loader, device)

    # Convert from 0–4 → 1–5
    preds = [p + 1 for p in preds]

    # Build final submission dataframe
    sub = pd.DataFrame({
        "ID": df["ID"].values,
        "CLASS": preds
    })

    # Save
    out_path = Path(output_csv)
    sub.to_csv(out_path, index=False)

    logger.info(f"Submission file saved to {out_path}")
    logger.info("Done!")


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # ABSOLUTE PATH TO YOUR XLSX
    TEST_XLSX = "/workspace/cmpt491_als/cmpt491_als/data/raw/sand_task1_test.xlsx"

    # OUTPUT NAME
    OUTPUT_CSV = "results.csv"

    make_submission(TEST_XLSX, OUTPUT_CSV)