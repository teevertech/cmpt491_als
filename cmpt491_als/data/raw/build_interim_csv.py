import csv
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Current file: data/raw/build_interim_csv.py
RAW_DIR = Path(__file__).resolve().parent        # data/raw
DATA_DIR = RAW_DIR.parent                        # data/
INTERIM_DIR = DATA_DIR / "interim"               # data/interim

RAW_CSV = RAW_DIR / "sand_dataset.csv"
AUDIO_CSV = RAW_DIR / "sand_audio_map.csv"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def normalize_path(p: str):
    """Convert Windows or relative paths to POSIX paths."""
    return p.replace("\\", "/")


def main():
    # ---- Load demographic CSV ----
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw CSV: {RAW_CSV}")

    meta_df = pd.read_csv(RAW_CSV)
    required_meta = {"ID", "Age", "Sex", "Class"}
    if not required_meta.issubset(meta_df.columns):
        raise ValueError(f"Metadata CSV missing columns: {required_meta - set(meta_df.columns)}")

    # ---- Load audio file CSV ----
    if not AUDIO_CSV.exists():
        raise FileNotFoundError(f"Missing AUDIO CSV: {AUDIO_CSV}")

    audio_df = pd.read_csv(AUDIO_CSV)
    required_audio = {"filepath", "label"}
    if not required_audio.issubset(audio_df.columns):
        raise ValueError(f"AUDIO CSV missing columns: {required_audio - set(audio_df.columns)}")

    # Extract ID from filepath
    audio_df["ID"] = audio_df["filepath"].apply(
        lambda p: Path(p).stem.split("_")[0]  # e.g. ID024
    )

    # Normalize paths
    audio_df["filepath"] = audio_df["filepath"].apply(normalize_path)

    # ---- Merge audio data with demographics ----
    full_df = audio_df.merge(meta_df, on="ID", how="left")

    missing_meta = full_df[full_df["Age"].isna()]
    if len(missing_meta) > 0:
        print("[WARN] Some audio IDs have no demographic info:")
        print(missing_meta["ID"].unique())

    # ---- Save full dataset ----
    interim_csv = INTERIM_DIR / "sand_dataset.csv"
    full_df.to_csv(interim_csv, index=False)
    print(f"[OK] Wrote cleaned interim CSV → {interim_csv}")

    # ---- Stratified splits ----
    train_df, test_df = train_test_split(
        full_df, test_size=0.10, stratify=full_df["label"], random_state=42
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