import csv
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_TASK1_DIR = DATA_DIR / "raw" / "SAND" / "task1" / "training"
METADATA_XLSX = DATA_DIR / "raw" / "SAND" / "task1" / "sand_task_1.xlsx"

AUDIO_TASKS = [
    "phonationA", "phonationE", "phonationI", "phonationO", "phonationU",
    "rhythmKA", "rhythmPA", "rhythmTA"
]

OUTPUT_CSV = DATA_DIR / "raw" / "sand_dataset.csv"


def main():
    print("[INFO] Loading metadata…")
    df = pd.read_excel(METADATA_XLSX)

    rows = []

    for _, row in df.iterrows():
        subject_id = row["ID"]     # e.g., ID000
        age = row["Age"]
        sex = row["Sex"]
        cls = row["Class"]         # 1–5

        subject_dir = RAW_TASK1_DIR

        for task in AUDIO_TASKS:
            wav_path = subject_dir / task / f"{subject_id}_{task}.wav"

            if wav_path.exists():
                relative = wav_path.relative_to(DATA_DIR / "raw")
                rows.append([
                    str(relative),
                    subject_id,
                    age,
                    sex,
                    cls
                ])
            else:
                print(f"[WARN] Missing: {wav_path}")

    print(f"[INFO] Writing {len(rows)} rows → {OUTPUT_CSV}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "ID", "Age", "Sex", "Class"])
        writer.writerows(rows)

    print("[DONE] CSV generated successfully.")


if __name__ == "__main__":
    main()