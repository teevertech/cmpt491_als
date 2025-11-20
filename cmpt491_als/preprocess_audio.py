import torchaudio
import torch
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

RAW_AUDIO_DIR = DATA_DIR / "raw"
INTERIM_CSV = DATA_DIR / "interim" / "sand_dataset.csv"

PROCESSED_DIR = DATA_DIR / "processed" / "SAND"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(INTERIM_CSV)

    updated_rows = []

    for i, row in df.iterrows():
        raw_path = RAW_AUDIO_DIR / row["filepath"]

        if not raw_path.exists():
            print(f"[WARN] Missing raw audio: {raw_path}")
            continue

        # Load audio (waveform, sr)
        waveform, sr = torchaudio.load(str(raw_path))

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Create output folder: e.g., data/processed/SAND/ID001/
        id_name = raw_path.stem
        out_dir = PROCESSED_DIR / id_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{id_name}.pt"
        torch.save({"waveform": waveform, "sr": sr, "label": row["label"]}, out_path)

        updated_rows.append({
            "filepath": str(out_path.relative_to(DATA_DIR)),
            "label": row["label"]
        })

        print(f"[OK] {raw_path.name} → {out_path}")

    # Write updated interim CSV pointing to processed paths
    processed_csv = DATA_DIR / "interim" / "sand_dataset.csv"
    pd.DataFrame(updated_rows).to_csv(processed_csv, index=False)

    print(f"[DONE] Updated interim CSV with PROCESSED paths → {processed_csv}")

if __name__ == "__main__":
    main()