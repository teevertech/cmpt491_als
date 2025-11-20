import csv
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent
RAW_CSV = DATA_DIR / "raw" / "sand_dataset.csv"
INTERIM_DIR = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def normalize_path(p: str):
    """Convert Windows or relative paths to POSIX paths."""
    p = p.replace("\\", "/")
    return p

def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw CSV: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)

    # Required columns
    required_cols = {"filepath", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Normalize paths
    df["filepath"] = df["filepath"].apply(normalize_path)

    # Save full cleaned CSV (including ID, Age, Sex, Class)
    interim_csv = INTERIM_DIR / "sand_dataset.csv"
    df.to_csv(interim_csv, index=False)
    print(f"[OK] Wrote cleaned interim CSV → {interim_csv}")

    # Split based only on label (metadata is kept)
    train_df, test_df = train_test_split(
        df, test_size=0.10, stratify=df["label"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.10, stratify=train_df["label"], random_state=42
    )

    train_df.to_csv(INTERIM_DIR / "train.csv", index=False)
    val_df.to_csv(INTERIM_DIR / "val.csv", index=False)
    test_df.to_csv(INTERIM_DIR / "test.csv", index=False)

    print("[OK] Wrote: train.csv, val.csv, test.csv")

if __name__ == "__main__":
    main()