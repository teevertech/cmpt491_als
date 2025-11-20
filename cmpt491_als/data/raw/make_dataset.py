import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent ##

def main():
    print("===== [1/2] Building INTERIM CSV =====")
    subprocess.run(["python", str(SCRIPT_DIR / "build_interim_csv.py")], check=True)

    print("===== [2/2] Preprocessing RAW audio → PROCESSED =====")
    PREPROCESS_SCRIPT = SCRIPT_DIR.parent / "preprocess_audio.py"
    subprocess.run(["python", str(PREPROCESS_SCRIPT)], check=True)

    print("===== Dataset pipeline COMPLETE =====")

if __name__ == "__main__":
    main()