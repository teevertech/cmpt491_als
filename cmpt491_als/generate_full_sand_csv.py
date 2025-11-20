import pandas as pd
from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw" / "SAND" / "task1" / "training"
XLSX_PATH = DATA_DIR / "raw" / "SAND" / "task1" / "sand_task_1.xlsx"
OUTPUT_CSV = DATA_DIR / "raw" / "sand_dataset.csv"

AUDIO_TASKS = [
    "phonationA", "phonationE", "phonationI", "phonationO", "phonationU",
    "rhythmKA", "rhythmPA", "rhythmTA"
]

def main():
    print("[INFO] Loading XLSX metadata...")
    meta = pd.read_excel(XLSX_PATH)

    # Ensure proper column names
    meta.columns = [str(c).strip() for c in meta.columns]

    rows = []

    # Create a lookup
    meta_dict = {}
    for _, row in meta.iterrows():
        subject_id = f"ID{int(row['ID']):03d}"
        meta_dict[subject_id] = {
            "Age": row.get("Age", None),
            "Sex": row.get("Sex", None),
            "Class": row.get("Class", None),
        }

    print("[INFO] Scanning audio directories...")

    for task in AUDIO_TASKS:
        task_dir = RAW_DIR / task
        if not task_dir.exists():
            print(f"[WARN] Missing task folder: {task_dir}")
            continue

        for wav in task_dir.glob("*.wav"):
            fname = wav.stem  # e.g., ID002_phonationA
            subject_id = fname.split("_")[0]

            if subject_id not in meta_dict:
                print(f"[WARN] Missing metadata for {subject_id}, skipping.")
                continue

            rows.append([
                str(wav.relative_to(DATA_DIR / "raw")),   # filepath
                subject_id,
                meta_dict[subject_id]["Age"],
                meta_dict[subject_id]["Sex"],
                meta_dict[subject_id]["Class"]
            ])

    print(f"[OK] Writing CSV with {len(rows)} rows → {OUTPUT_CSV}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "ID", "Age", "Sex", "Class"])
        writer.writerows(rows)

    print("[DONE] CSV complete.")

if __name__ == "__main__":
    main()