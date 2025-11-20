"""
SAND Dataset classes for ALS classification model training and evaluation.

This module contains PyTorch dataset classes for the SAND competition data:
- SANDDataset: For loading preprocessed tensors (used by train/predict scripts)
- RawAudioDataset: For loading raw audio files (used by features.py)
"""

#sand_datasets.py
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from loguru import logger
from torch.utils.data import Dataset


class SANDDataset(Dataset):
    """
    PyTorch Dataset for SAND Task 1 (ALS severity classification).

    Option A: load raw waveforms directly from .wav files using the
    "filepath" column in the metadata CSV, then apply ASTFeatureExtractor
    on-the-fly to get model-ready input_values.
    """

    def __init__(
        self,
        audio_root: Path,          # e.g. RAW_DATA_DIR from config
        metadata_csv: Path,        # e.g. data/interim/train.csv
        feature_extractor=None,
        target_sample_rate: int = 16000,
        max_length_seconds: float = 5.0,  # adjust if you want longer crops
    ):
        self.audio_root = Path(audio_root)
        self.metadata = pd.read_csv(metadata_csv)
        self.feature_extractor = feature_extractor
        self.target_sr = target_sample_rate
        self.max_length = int(max_length_seconds * target_sample_rate)

        self.samples = []
        self._create_samples()

        logger.info(f"Loaded {len(self.samples)} samples from {metadata_csv}")
        if len(self.samples) > 0:
            logger.info(f"Class distribution: {self._get_class_distribution()}")

    def _create_samples(self):
        """
        Build list of valid audio samples from the metadata CSV.

        Expects columns:
          - filepath  (e.g. 'audio/phonationA/ID059_phonationA.wav')
          - label     (1..5 from your pipeline – we convert to 0..4 here)
          - optional: ID, Age, Sex
        """
        for _, row in self.metadata.iterrows():
            rel_path = row["filepath"]
            raw_label = int(row["label"])

            audio_path = self.audio_root / rel_path  # audio_root = RAW_DATA_DIR

            if not audio_path.exists():
                logger.warning(f"[SANDDataset] Audio file missing, skipping: {audio_path}")
                continue

            # Convert labels from 1..5 → 0..4
            label = raw_label - 1 if raw_label != -1 else -1

            # Try to get subject_id from either CSV or filename
            subject_id = row.get("ID", None)
            if pd.isna(subject_id):
                fname = audio_path.stem  # ID059_phonationA
                subject_id = fname.split("_")[0]

            # Extract audio_task from filename (phonationA, etc.)
            fname = audio_path.stem
            parts = fname.split("_")
            audio_task = parts[1] if len(parts) > 1 else None

            self.samples.append({
                "audio_path": audio_path,
                "label": label,
                "subject_id": subject_id,
                "audio_task": audio_task,
                "age": row.get("Age", None),
                "sex": row.get("Sex", None),
            })

    def _get_class_distribution(self):
        """Get distribution of classes in the dataset."""
        labels = [s["label"] for s in self.samples if s["label"] != -1]
        if len(labels) == 0:
            return {}
        unique, counts = torch.unique(torch.tensor(labels), return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def __len__(self):
        return len(self.samples)

    def _load_and_standardize_waveform(self, audio_path: Path) -> torch.Tensor:
        """
        Load audio, convert to mono, resample, and center-crop / pad
        to fixed length (max_length).
        """
        waveform, sr = torchaudio.load(str(audio_path))

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sr
            )
            waveform = resampler(waveform)

        # Center crop or pad to max_length
        T = waveform.shape[1]
        if T > self.max_length:
            start = (T - self.max_length) // 2
            waveform = waveform[:, start:start + self.max_length]
        elif T < self.max_length:
            pad = self.max_length - T
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform  # (1, max_length)

    def __getitem__(self, idx):
        """Get a single sample for training/inference."""
        sample = self.samples[idx]

        # Load and standardize waveform
        waveform = self._load_and_standardize_waveform(sample["audio_path"])

        # Convert to numpy for ASTFeatureExtractor
        waveform_np = waveform.squeeze(0).numpy()

        # Use AST feature extractor (HuggingFace AST)
        if self.feature_extractor is not None:
            inputs = self.feature_extractor(
                waveform_np,
                sampling_rate=self.target_sr,
                return_tensors="pt"
            )
            # ASTFeatureExtractor returns (1, time, freq) or similar
            audio_features = inputs["input_values"].squeeze(0)

        # Fallback: return raw waveform (for debugging / non-AST models)
        else:
            audio_features = waveform.squeeze(0)

        # Make sure labels are valid LongTensors
        label = sample["label"]
        if label == -1:
            label = 0  # or whatever you want for "unknown"
        label_tensor = torch.tensor(label, dtype=torch.long)

        data_sample = {
            "input_values": audio_features,      # (time, freq) for AST / ElasticAST
            "labels": label_tensor,              # 0..4
            "subject_id": sample["subject_id"],
            "audio_task": sample["audio_task"],
            "metadata": {
                "age": sample["age"],
                "sex": sample["sex"],
            },
        }

        return data_sample


class RawAudioDataset(Dataset):
    """
    Dataset for loading raw audio files during feature preprocessing.
    Used by features.py to convert raw audio to preprocessed tensors.
    """

    def __init__(self, data_dir: Path, metadata_csv: Path):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_csv)
        self.audio_tasks = [
            "phonationA", "phonationE", "phonationI", "phonationO", "phonationU",
            "rhythmKA", "rhythmPA", "rhythmTA"
        ]

        self.samples = []
        self._create_samples()

        logger.info(f"Found {len(self.samples)} raw audio files to process")

    def _create_samples(self):
        """Create list of raw audio files to process"""
        for _, row in self.metadata.iterrows():
            subject_id = row['ID']
            subject_dir = self.data_dir / subject_id

            if not subject_dir.exists():
                logger.warning(f"Subject directory not found: {subject_dir}")
                continue

            for task in self.audio_tasks:
                audio_file = subject_dir / f"{subject_id}_{task}.wav"
                if audio_file.exists():
                    self.samples.append({
                        'subject_id': subject_id,
                        'audio_file': audio_file,
                        'audio_task': task,
                        'label': row['Class'] - 1,
                        'age': row['Age'],
                        'sex': row['Sex']
                    })
                else:
                    logger.warning(f"Raw audio file not found: {audio_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single raw audio sample for preprocessing."""
        return self.samples[idx]
