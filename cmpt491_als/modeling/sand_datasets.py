"""
SAND Dataset classes for ALS classification model training and evaluation.

This module contains PyTorch dataset classes for the SAND competition data:
- SANDDataset: For loading preprocessed tensors (used by train/predict scripts)
- RawAudioDataset: For loading raw audio files (used by features.py)
"""

from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from loguru import logger
import torchaudio


class SANDDataset(Dataset):
    """
    Loads SAND CSVs of the form:

        filepath,label,ID,Age,Sex,Class
        audio/phonationU/ID255_phonationU.wav,4,ID255,37,M,4
        ...

    and returns log-Mel tensors ready for ElasticAST.

    __getitem__ returns:
        {
            "input_values": (T, F) float32 log-Mel,
            "labels": int64 scalar,
            "age": float or int,
            "sex": str,
            "id": str,
        }
    """

    def __init__(
        self,
        audio_root: Path,
        metadata_csv: Path,
        target_sr: int = 16000,
        n_mels: int = 128,
    ) -> None:
        super().__init__()

        self.audio_root = Path(audio_root)
        self.df = pd.read_csv(metadata_csv)

        if "filepath" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(
                "Expected at least 'filepath' and 'label' columns in "
                f"{metadata_csv}, got: {self.df.columns.tolist()}"
            )

        self.target_sr = target_sr

        # Mel-spec transform (AST defaults: 16kHz, 128 Mel bins)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels,
            f_min=0.0,
            f_max=target_sr / 2,
        )

        logger.info(
            f"SANDDataset from {metadata_csv} with {len(self.df)} rows, "
            f"audio_root={self.audio_root}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_waveform(self, wav_path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(wav_path))

        # mix to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)

        return wav  # shape: (1, N)

    def _waveform_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (1, N) float32 in [-1, 1]
        returns: (T, F) log-Mel
        """
        mel = self.mel_transform(wav)  # (1, n_mels, T)
        mel = mel.squeeze(0)           # (n_mels, T)

        # log-mel
        log_mel = torch.log(mel + 1e-6)  # (n_mels, T)
        log_mel = log_mel.transpose(0, 1)  # (T, n_mels)

        return log_mel

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        rel_path = str(row["filepath"])
        wav_path = self.audio_root / rel_path

        if not wav_path.exists():
            raise FileNotFoundError(f"Missing audio file: {wav_path}")

        wav = self._load_waveform(wav_path)
        log_mel = self._waveform_to_logmel(wav)

        MAX_WIDTH = 1024   # ElasticAST-pretrained window

        if log_mel.shape[0] > MAX_WIDTH:
            log_mel = log_mel[:MAX_WIDTH, :]
            
        raw_label = int(row["label"])

        # Convert from 1–5 → 0–4
        label = raw_label - 1

        if not (0 <= label < 5):
            raise ValueError(f"Label out of range after shift: {raw_label} -> {label}")

        sample = {
            "input_values": log_mel,  # (T, F)
            "labels": torch.tensor(label, dtype=torch.long),
            "id": row.get("ID", None),
            "age": row.get("Age", None),
            "sex": row.get("Sex", None),
        }

        return sample


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
