from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw" / "SAND" / "task1" / "training"
OUT_CSV = DATA_DIR / "raw" / "sand_dataset.csv"

def main():
    rows = [["ID", "Task", "Filepath"]]

    for task_dir in RAW_DIR.iterdir():
        if not task_dir.is_dir():
            continue

        task = task_dir.name

        for wav in task_dir.glob("*.wav"):
            name = wav.stem  # e.g. ID004_phonationA
            subject_id = name.split("_")[0]  # e.g. ID004

            rel_path = wav.relative_to(DATA_DIR / "raw")
            rows.append([subject_id, task, str(rel_path)])

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[OK] Metadata CSV created: {OUT_CSV}")
    print(f"[OK] Total samples: {len(rows)-1}")

if __name__ == "__main__":
    main()