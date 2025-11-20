"""
SAND Dataset classes for ALS classification model training and evaluation.

This module contains PyTorch dataset classes for the SAND competition data:
- SANDDataset: For loading preprocessed tensors (used by train/predict scripts)
- RawAudioDataset: For loading raw audio files (used by features.py)
"""

#sand_datasets.py
import torch
import pandas as pd
from pathlib import Path
from loguru import logger
from torch.utils.data import Dataset


class SANDDataset(Dataset):
    """
    PyTorch Dataset for SAND Task 1 (ALS severity classification)
    This dataset assumes audio preprocessing has already been done in features.py
    """

    def __init__(
        self,
        data_dir: Path,
        metadata_csv: Path,
        feature_extractor=None
    ):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_csv)
        self.feature_extractor = feature_extractor
        self.audio_tasks = [
            "phonationA", "phonationE", "phonationI", "phonationO", "phonationU",
            "rhythmKA", "rhythmPA", "rhythmTA"
        ]

        self.samples = []
        self._create_samples()

        logger.info(f"Loaded {len(self.samples)} samples")
        if len(self.samples) > 0:
            logger.info(f"Class distribution: {self._get_class_distribution()}")

    def _create_samples(self):
        for _, row in self.metadata.iterrows():
            tensor_rel_path = row['filepath']
            label = row['label']

            tensor_file = self.data_dir / tensor_rel_path

            if not tensor_file.exists():
                logger.warning(f"Tensor file missing: {tensor_file}")
                continue

            # Extract subject_id from filename: e.g. "ID000_phonationA.pt"
            fname = Path(tensor_rel_path).name
            subject_id = fname.split("_")[0]  # "ID000"

            audio_task = fname.split("_")[1].replace(".pt", "")  # phonationA

            self.samples.append({
                'subject_id': subject_id,
                'tensor_file': tensor_file,
                'audio_task': audio_task,
                'label': label,
                'age': None,
                'sex': None
            })

    def _get_class_distribution(self):
        """Get distribution of classes in the dataset."""
        labels = [sample['label'] for sample in self.samples]
        unique, counts = torch.unique(torch.tensor(labels), return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample for training/inference."""
        sample = self.samples[idx]

        # Load preprocessed tensor (raw waveform)
        waveform = torch.load(sample['tensor_file'])

        # Convert to numpy for AST feature extractor
        waveform_np = waveform.squeeze().numpy()

        # Use AST feature extractor (HuggingFace AST)
        if self.feature_extractor is not None:
            inputs = self.feature_extractor(
                waveform_np,
                sampling_rate=16000,
                return_tensors="pt"
            )
            audio_features = inputs['input_values'].squeeze(0)

        # ElasticAST: return raw waveform (1D)
        else:
            # waveform is loaded as a torch tensor, may be (1, T)
            audio_features = waveform.squeeze()

        # Build the return data structure
        data_sample = {
            'input_values': audio_features,
            'labels': torch.tensor(sample['label'] if sample['label'] != -1 else 0),
            'subject_id': sample['subject_id'],
            'audio_task': sample['audio_task'],
            'metadata': {
                'age': sample['age'],
                'sex': sample['sex']
            }
        }

        return data_sample

    def get_class_weights(self):
        """Compute class weights for balanced training."""
        class_dist = self._get_class_distribution()
        num_classes = 5
        total_samples = len(self.samples)

        weights = []
        for class_idx in range(num_classes):
            if class_idx in class_dist:
                weight = total_samples / (num_classes * class_dist[class_idx])
            else:
                weight = 1.0
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


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
