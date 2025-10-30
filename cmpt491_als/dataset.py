from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import os
import shutil

from cmpt491_als.config import INTERIM_DATA_DIR, RAW_DATA_DIR, SAND_DATA_DIR

app = typer.Typer()

def copy_audio_files(df):
    """
    Make a copy of the audio data but sorted by subject_ID instead of sound type.
    """

    for index, row in df.iterrows():
        subject_ID = row['ID']

        # make subdirectory for a subject.
        subject_directory =  INTERIM_DATA_DIR / "SAND" / "task1" / "training" / subject_ID
        subject_directory.mkdir(parents=True, exist_ok=True)
        
        # iterate through .wav directories and match .wav files to subject ID.
        # copy them to the subject ID directory.
        for sound_dir in (SAND_DATA_DIR / "training").iterdir():
            if sound_dir.is_dir():
                for audio_file in sound_dir.iterdir():
                    if (audio_file.name[:5] == subject_ID):
                        shutil.copy2(audio_file, subject_directory)

    logger.success("Copy dataset complete.")

def parse_raw_data(file_path):
    """Read the SAND provided excel document to extract audio metadata"""

    training_data = []

    sheets = {
        '0': 'Training Baseline - Task 1',
        '1': 'Validation Baseline - Task 1'
    }

    for split_name, sheet_name in sheets.items():
        training_df = pd.read_excel(file_path, sheet_name=sheet_name)
        training_df['Dataset_split'] = split_name

        logger.info(f"Processing {len(training_df)} subjects from {sheet_name}")

        training_data.append(training_df)

    combined_df = pd.concat(training_data, ignore_index=True) if training_data else pd.DataFrame()

    return combined_df.sort_values(by=['ID'])

@app.command()
def main(
    input_path: Path = SAND_DATA_DIR / "sand_task_1.xlsx",
    output_path: Path = INTERIM_DATA_DIR / "sand_dataset.csv",
):

    logger.info("Processing dataset...")

    df = parse_raw_data(input_path)

    df.to_csv(output_path, index=False)

    copy_audio_files(df)

    logger.success("Processing dataset complete.")

if __name__ == "__main__":
    app()
