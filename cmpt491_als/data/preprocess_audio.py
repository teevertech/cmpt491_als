import librosa
import soundfile as sf
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"           # data/raw
PROC_DIR = Path(__file__).resolve().parent / "processed"     # data/processed

TARGET_SR = 16000   # ElasticAST recommended sample rate

def process_audio(input_path: Path, output_path: Path):
    """Load audio, resample, normalize, and save."""
    try:
        audio, sr = librosa.load(input_path, sr=TARGET_SR)

        # Normalize audio peak to ±1
        if audio.size > 0:
            audio = audio / max(1e-9, abs(audio).max())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, TARGET_SR)

    except Exception as e:
        print(f"[ERROR] Failed to process {input_path}: {e}")


def main():
    print("[Audio Preprocessing] Starting...")

    audio_root = RAW_DIR / "audio"
    if not audio_root.exists():
        raise FileNotFoundError(f"Missing RAW audio directory: {audio_root}")

    # Recursively process all wav files
    wav_files = list(audio_root.rglob("*.wav"))
    print(f"[INFO] Found {len(wav_files)} WAV files.")

    for wav in wav_files:
        relative = wav.relative_to(audio_root)
        out_path = PROC_DIR / relative
        out_path = out_path.with_suffix(".wav")

        print(f"[PROCESS] {wav} → {out_path}")
        process_audio(wav, out_path)

    print(f"[DONE] Processed audio written to: {PROC_DIR}")


if __name__ == "__main__":
    main()