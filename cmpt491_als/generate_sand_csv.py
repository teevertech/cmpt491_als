import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw" / "SAND" / "task1" / "training"
OUTPUT_CSV = DATA_DIR / "raw" / "sand_dataset.csv"

def main():
    rows = []

    # Loop through label folders
    for label_dir in RAW_DIR.iterdir():
        if not label_dir.is_dir():
            continue

        label = label_dir.name  # e.g., "phonationE"

        # Loop through WAV files inside each label folder
        for wav in label_dir.glob("*.wav"):
            relative_path = wav.relative_to(DATA_DIR / "raw")
            rows.append([str(relative_path), label])

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} entries to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()