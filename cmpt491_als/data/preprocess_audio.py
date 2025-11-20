import torchaudio
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw" / "audio"
PROC_DIR = DATA_DIR / "processed"
TARGET_SR = 16000

def process_audio(input_path: Path, output_path: Path):
    try:
        # Load audio
        waveform, sr = torchaudio.load(str(input_path))

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)

        # Peak normalize
        peak = waveform.abs().max().item()
        if peak > 0:
            waveform = waveform / peak

        # Ensure output folder exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save WAV
        torchaudio.save(str(output_path), waveform, TARGET_SR)

    except Exception as e:
        print(f"[ERROR] Failed processing {input_path}: {e}")


def main():
    print("[Audio Preprocessing] Starting...")

    wav_files = list(RAW_DIR.rglob("*.wav"))
    print(f"[INFO] Found {len(wav_files)} audio files.")

    for wav in wav_files:
        rel = wav.relative_to(RAW_DIR)
        out_path = PROC_DIR / rel

        print(f"[PROCESS] {wav} → {out_path}")
        process_audio(wav, out_path)

    print("[DONE] Processed audio saved to:", PROC_DIR)


if __name__ == "__main__":
    main()