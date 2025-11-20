import csv
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_CSV = DATA_DIR / "raw" / "sand_dataset.csv"
INTERIM_DIR = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def normalize_path(p: str):
    """ Convert Windows paths or relative paths → POSIX project paths """
    p = p.replace("\\", "/")
    return p

def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw CSV: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    if "filepath" not in df or "label" not in df:
        raise ValueError("CSV must have columns: filepath,label")

    # Normalize paths
    df["filepath"] = df["filepath"].apply(normalize_path)

    # Write cleaned interim CSV
    interim_csv = INTERIM_DIR / "sand_dataset.csv"
    df.to_csv(interim_csv, index=False)
    print(f"[OK] Wrote cleaned interim CSV → {interim_csv}")

    # Build train/val/test splits
    train_df, test_df = train_test_split(df, test_size=0.10, stratify=df["label"], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.10, stratify=train_df["label"], random_state=42)

    train_df.to_csv(INTERIM_DIR / "train.csv", index=False)
    val_df.to_csv(INTERIM_DIR / "val.csv", index=False)
    test_df.to_csv(INTERIM_DIR / "test.csv", index=False)

    print("[OK] Wrote: train.csv, val.csv, test.csv")

if __name__ == "__main__":
    main()