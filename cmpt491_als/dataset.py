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
        subject_directory = INTERIM_DATA_DIR / "SAND" / "task1" / "training" / subject_ID
        subject_directory.mkdir(parents=True, exist_ok=True)
        # iterate through .wav directories and match .wav files to subject ID.
        # copy them to the subject ID directory.
        for sound_dir in (SAND_DATA_DIR / "training").iterdir():
            if sound_dir.is_dir():
                for audio_file in sound_dir.iterdir():
                    if (audio_file.name[:5] == subject_ID):
                        shutil.copy2(audio_file, subject_directory)
    logger.success("Copy dataset complete.")


def copy_test_audio_files(df):
    """
    Copy test audio files organized by subject_ID instead of sound type.
    """
    for index, row in df.iterrows():
        subject_ID = f"ID{row['ID']:03d}"  # Format as ID004, ID011, etc.
        # make subdirectory for a subject.
        subject_directory = INTERIM_DATA_DIR / "SAND" / "task1" / "testing" / subject_ID
        subject_directory.mkdir(parents=True, exist_ok=True)
        # iterate through .wav directories and match .wav files to subject ID.
        # copy them to the subject ID directory.
        for sound_dir in (SAND_DATA_DIR / "test").iterdir():
            if sound_dir.is_dir():
                for audio_file in sound_dir.iterdir():
                    if (audio_file.name[:5] == subject_ID):
                        shutil.copy2(audio_file, subject_directory)
    logger.success("Copy test dataset complete.")


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


def parse_test_data(file_path):
    """Read the SAND test excel document to extract test metadata"""
    test_df = pd.read_excel(file_path, sheet_name='SAND - TESTING set - Task 1')
    test_df['Dataset_split'] = 2  # Test split
    # Leave Class column as-is (empty/NaN for test data)
    logger.info(f"Processing {len(test_df)} test subjects")
    return test_df.sort_values(by=['ID'])


def process_test_data(test_file_path, output_path):
    """Process test data."""
    logger.info("Processing test dataset...")
    test_df = parse_test_data(test_file_path)
    test_df.to_csv(output_path, index=False)
    copy_test_audio_files(test_df)
    logger.success("Processing test dataset complete.")


@app.command()
def main(
    train_val_path: Path = SAND_DATA_DIR / "sand_task_1.xlsx",
    test_path: Path = SAND_DATA_DIR / "sand_task1_test.xlsx",
    output_path: Path = INTERIM_DATA_DIR / "sand_dataset.csv",
    test_output_path: Path = INTERIM_DATA_DIR / "sand_test_dataset.csv",
):
    """Process training, validation, and test data."""
    # Process training/validation data
    logger.info("Processing training/validation dataset...")
    df = parse_raw_data(train_val_path)
    df.to_csv(output_path, index=False)
    copy_audio_files(df)
    logger.success("Processing training/validation dataset complete.")

    # Process test data
    process_test_data(test_path, test_output_path)

    logger.success("All data processing complete!")


if __name__ == "__main__":
    app()     # dataset.py
