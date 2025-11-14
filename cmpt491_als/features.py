"""
Audio feature engineering for SAND ALS classification.

This module handles preprocessing of raw audio files into standardized tensors
suitable for model training. Follows cookie cutter data science template.
"""

import torch
import torchaudio
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import shutil
from cmpt491_als.modeling.sand_datasets import RawAudioDataset
from cmpt491_als.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, AUDIO_CONFIG

app = typer.Typer()


def preprocess_audio_file(
    audio_path: Path,
    target_sample_rate: int = AUDIO_CONFIG["target_sample_rate"],
    max_length_seconds: float = AUDIO_CONFIG["max_length_seconds"],
    to_mono: bool = True
) -> torch.Tensor:
    """

        Returns Preprocessed audio tensor
    """
    try:
        # Check if file exists and has content
        if not audio_path.exists():
            logger.error(f"Audio file does not exist: {audio_path}")
            return None

        file_size = audio_path.stat().st_size
        if file_size == 0:
            logger.error(f"Audio file is empty (0 bytes): {audio_path}")
            return None

        if file_size < 100:  # Very small files are likely corrupted
            logger.warning(f"Audio file is very small ({file_size} bytes), might be corrupted: {audio_path}")

        # Method 1: Try using torchaudio with explicit backend
        waveform, sample_rate = None, None

        try:
            # Force use of soundfile backend
            waveform, sample_rate = torchaudio.load(str(audio_path), backend="soundfile")
            logger.debug(f"Loaded with soundfile backend: {audio_path}")
        except Exception as e:
            logger.debug(f"Soundfile backend failed: {e}")

        # Method 2: Use soundfile directly if torchaudio fails
        if waveform is None:
            try:
                import soundfile as sf
                waveform_np, sample_rate = sf.read(str(audio_path), dtype='float32')

                # Convert to torch tensor and add channel dimension if needed
                waveform = torch.tensor(waveform_np, dtype=torch.float32)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # Add channel dimension
                elif waveform.dim() == 2:
                    waveform = waveform.transpose(0, 1)  # (time, channels) -> (channels, time)

                logger.debug(f"Loaded with soundfile directly: {audio_path}")
            except ImportError:
                logger.error("soundfile not installed. Install with: pip install soundfile")
                return None
            except Exception as e:
                logger.error(f"soundfile loading failed: {e}")
                return None

        if waveform is None:
            logger.error(f"Could not load audio file with any method: {audio_path}")
            return None

        # Check if waveform is valid
        if waveform.numel() == 0:
            logger.error(f"Loaded empty waveform from: {audio_path}")
            return None

        logger.debug(f"Successfully loaded: {audio_path} - Shape: {waveform.shape}, SR: {sample_rate}")

        # Convert to mono if needed
        if to_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            )
            waveform = resampler(waveform)

        # Pad or truncate to fixed length
        target_length = int(max_length_seconds * target_sample_rate)
        current_length = waveform.shape[1]

        if current_length > target_length:
            # Truncate from center
            start_idx = (current_length - target_length) // 2
            waveform = waveform[:, start_idx:start_idx + target_length]
        elif current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)

        return waveform

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        logger.error(f"File size: {file_size} bytes")
        return None


def extract_audio_features(df, input_dir, output_dir):
    """
    Extract and preprocess audio features from raw audio files.
    Uses settings from config..
    """
    # Create raw audio dataset
    raw_dataset = RawAudioDataset(input_dir, INTERIM_DATA_DIR / "sand_dataset.csv")

    # Process each audio file using config settings
    processed_count = 0
    failed_count = 0
    failed_files = []

    for sample in tqdm(raw_dataset, desc="Processing audio files"):
        subject_id = sample['subject_id']
        audio_file = sample['audio_file']
        audio_task = sample['audio_task']

        # Create output directory for subject
        subject_output_dir = output_dir / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess audio (uses config defaults)
        processed_tensor = preprocess_audio_file(audio_file)

        if processed_tensor is not None:
            # Save as tensor
            output_file = subject_output_dir / f"{subject_id}_{audio_task}.pt"
            torch.save(processed_tensor, output_file)
            processed_count += 1
        else:
            failed_count += 1
            failed_files.append(str(audio_file))
            logger.warning(f"Failed to process {audio_file}")

    # Report statistics
    total_files = len(raw_dataset)
    success_rate = (processed_count / total_files) * 100 if total_files > 0 else 0

    logger.info(f"Processing complete!")
    logger.info(f"Successfully processed: {processed_count}/{total_files} files ({success_rate:.1f}%)")
    logger.info(f"Failed: {failed_count} files")
    logger.info(f"Used settings: {AUDIO_CONFIG}")

    if failed_files:
        logger.warning(f"Failed files: {failed_files[:5]}...")  # Show first 5
        if len(failed_files) > 5:
            logger.warning(f"... and {len(failed_files) - 5} more")

    return df


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "sand_dataset.csv",
    test_input_path: Path = INTERIM_DATA_DIR / "sand_test_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    """
    Main feature extraction pipeline for training/validation data.
    Also copies test data to processed directory to maintain flow.

    Uses audio processing settings from config.py:
    - Sample rate: {AUDIO_CONFIG['target_sample_rate']} Hz
    - Max length: {AUDIO_CONFIG['max_length_seconds']} seconds
    """
    logger.info("Running complete feature extraction pipeline...")
    logger.info(f"Audio processing config: {AUDIO_CONFIG}")

    # Paths for training/val
    input_dir = INTERIM_DATA_DIR / "SAND" / "task1" / "training"
    output_dir = PROCESSED_DATA_DIR / "SAND" / "task1" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training/val metadata
    if not input_path.exists():
        logger.error(f"Training/val metadata file not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Extract audio features for training/val
    updated_df = extract_audio_features(df, input_dir, output_dir)

    # Create train/val splits
    train_dir = PROCESSED_DATA_DIR / "SAND" / "task1" / "train"
    val_dir = PROCESSED_DATA_DIR / "SAND" / "task1" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Process each subject into splits
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating train/val splits"):
        subject_id = row['ID']
        split = row['Dataset_split']

        source_dir = output_dir / subject_id
        target_dir = train_dir if split == 0 else val_dir
        target_subject_dir = target_dir / subject_id

        if source_dir.exists():
            if target_subject_dir.exists():
                shutil.rmtree(target_subject_dir)
            shutil.copytree(source_dir, target_subject_dir)

    # Copy metadata to processed directory
    df.to_csv(output_path, index=False)

    logger.success("Training/validation feature extraction complete!")

    # Copy test data to processed directory (no processing, just maintain flow)
    if test_input_path.exists():
        logger.info("Copying test data to processed directory...")

        test_interim_dir = INTERIM_DATA_DIR / "SAND" / "task1" / "testing"
        test_processed_dir = PROCESSED_DATA_DIR / "SAND" / "task1" / "test"

        # Copy entire test directory structure
        if test_interim_dir.exists():
            if test_processed_dir.exists():
                shutil.rmtree(test_processed_dir)
            shutil.copytree(test_interim_dir, test_processed_dir)
            logger.success("Test data copied to processed directory!")
        else:
            logger.warning(f"Test interim directory not found: {test_interim_dir}")

        # Copy test metadata
        test_df = pd.read_csv(test_input_path)
        test_df.to_csv(PROCESSED_DATA_DIR / "test_features.csv", index=False)

    else:
        logger.warning(f"Test metadata not found: {test_input_path}")
        logger.info("Skipping test data copy")

    logger.success("Feature extraction pipeline complete!")
    logger.info(f"Train data: {train_dir}")
    logger.info(f"Validation data: {val_dir}")
    if test_input_path.exists():
        logger.info(f"Test data: {test_processed_dir} (raw audio files)")


if __name__ == "__main__":
    app() # features.py
