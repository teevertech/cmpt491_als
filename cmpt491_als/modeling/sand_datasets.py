"""
SAND Dataset classes for ALS classification model training and evaluation.

This module contains PyTorch dataset classes for the SAND competition data:
- SANDDataset: For loading preprocessed tensors (used by train/predict scripts)
- RawAudioDataset: For loading raw audio files (used by features.py)
"""

from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger
import torchaudio


class SANDDataset(Dataset):
    """
    CSV formats supported:

    TRAIN / VAL CSV:
        filepath,label,ID,Age,Sex,...
        audio/xyz.wav,4,ID001,37,M,...

    TEST CSV (from XLSX):
        filepath,ID,...
        audio/xyz.wav,ID999,...

    __getitem__ returns:

    TRAIN / VAL:
        {
            "input_values": (T, F),
            "labels": int64,
            "ids": str,
            "age": int/float/None,
            "sex": str/None
        }

    TEST:
        {
            "input_values": (T, F),
            "ids": str
        }
    """

    def __init__(
        self,
        audio_root: Path,
        metadata_csv: Path,
        target_sr: int = 16000,
        n_mels: int = 128,
        max_width: int = 1024,
        is_test: bool = False,
    ) -> None:
        super().__init__()

        self.audio_root = Path(audio_root)
        self.df = pd.read_csv(metadata_csv)
        self.is_test = is_test
        self.target_sr = target_sr
        self.max_width = max_width

        # Validate columns
        if "filepath" not in self.df.columns:
            raise ValueError(f"'filepath' missing in CSV: {self.df.columns.tolist()}")

        if not is_test and "label" not in self.df.columns:
            raise ValueError(
                "Training/validation CSV must include 'label' column.\n"
                f"Got: {self.df.columns.tolist()}"
            )

        # Prebuild MelSpectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels,
            f_min=0.0,
            f_max=target_sr / 2,
        )

        logger.info(
            f"SANDDataset: {metadata_csv} | {len(self.df)} samples | test={self.is_test}"
        )

    # --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # --------------------------------------------------------------
    def _load_waveform(self, wav_path: Path) -> torch.Tensor:
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing audio file: {wav_path}")

        wav, sr = torchaudio.load(str(wav_path))

        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample
        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)

        return wav.float()  # (1, N)

    # --------------------------------------------------------------
    def _waveform_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(wav)        # (1, n_mels, T)
        mel = mel.squeeze(0)                # (n_mels, T)

        log_mel = torch.log(mel + 1e-6)     # (n_mels, T)
        log_mel = log_mel.transpose(0, 1)   # (T, n_mels)

        # Trim to ElasticAST window
        if log_mel.shape[0] > self.max_width:
            log_mel = log_mel[: self.max_width, :]

        return log_mel

    # --------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        wav_path = self.audio_root / str(row["filepath"])
        wav = self._load_waveform(wav_path)
        log_mel = self._waveform_to_logmel(wav)

        # --------------------------------
        # TEST MODE → no labels returned
        # --------------------------------
        if self.is_test:
            return {
                "ids": row["ID"],
                "input_values": log_mel,
            }

        # --------------------------------
        # TRAIN/VAL MODE
        # --------------------------------
        label_raw = int(row["label"])
        label = label_raw - 1  # convert 1–5 → 0–4

        if not (0 <= label < 5):
            raise ValueError(f"Invalid label {label_raw} → {label}")

        return {
            "ids": row.get("ID", None),
            "input_values": log_mel,
            "labels": torch.tensor(label, dtype=torch.long),
            "age": row.get("Age", None),
            "sex": row.get("Sex", None),
        }


class RawAudioDataset(Dataset):
    """
    Lightweight dataset used by your preprocessing / features.py pipeline.
    (You can keep your existing implementation if you already use it.)
    """

    def __init__(self, data_dir: Path, metadata_csv: Path) -> None:
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(metadata_csv)

        if "filepath" not in self.df.columns:
            raise ValueError(
                "RawAudioDataset expects a 'filepath' column in the CSV."
            )

        self.samples = []
        for _, row in self.df.iterrows():
            wav_path = self.data_dir / row["filepath"]
            if wav_path.exists():
                self.samples.append(
                    {
                        "path": wav_path,
                        "label": row.get("label", None),
                    }
                )
            else:
                logger.warning(f"Missing raw audio: {wav_path}")

        logger.info(f"RawAudioDataset built with {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
