from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import parselmouth
from parselmouth.praat import call

from cmpt491_als.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

def extract_audio_features(df):

    for row in df.iterrow():
        print(row)
        updated_subject_row = subject_row

    return df


@app.command()
def main(

    input_path: Path = INTERIM_DATA_DIR / "sand_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",

):
    logger.info("Generating features from dataset...")

    df = pd.read_csv(input_path)

    updated_df = extract_audio_features(df)

if __name__ == "__main__":
    app()
