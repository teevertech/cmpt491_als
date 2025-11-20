import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent
XLSX_FILE = RAW_DIR / "sand_task_1.xlsx"
AUDIO_DIR = RAW_DIR / "audio"     # will search recursively

OUTPUT_META = RAW_DIR / "sand_dataset.csv"
OUTPUT_AUDIO = RAW_DIR / "sand_audio_map.csv"

def main():
    # ------------------------------
    # 1) Build sand_dataset.csv
    # ------------------------------
    print("[1] Reading Excel metadata...")
    df = pd.read_excel(XLSX_FILE)
    df.columns = df.columns.str.strip()

    required = {"ID", "Age", "Sex", "Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing required columns: {missing}")

    df.to_csv(OUTPUT_META, index=False)
    print(f"[OK] Wrote metadata → {OUTPUT_META}")


    # ------------------------------
    # 2) Build sand_audio_map.csv
    # ------------------------------
    print("[2] Scanning audio directory (recursive)...")

    rows = []
    for wav in AUDIO_DIR.rglob("*.wav"):  # <—— recursive search!
        stem = wav.stem  # e.g. "ID059_phonationA"
        ID = stem.split("_")[0]  # "ID059"

        # Look up label from metadata
        class_row = df[df["ID"] == ID]
        if class_row.empty:
            print(f"[WARN] No metadata for audio file: {wav.name}")
            continue

        label = int(class_row["Class"].iloc[0])

        # Convert full path → path relative to raw/
        rel_path = wav.relative_to(RAW_DIR)

        rows.append({
            "filepath": str(rel_path).replace("\\", "/"),
            "label": label
        })

    audio_df = pd.DataFrame(rows)
    audio_df.to_csv(OUTPUT_AUDIO, index=False)
    print(f"[OK] Wrote audio map → {OUTPUT_AUDIO}")

    print("\n=== DONE ===")
    print("You can now run: python make_dataset.py")

if __name__ == "__main__":
    main()