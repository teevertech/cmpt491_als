import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

# Correctly locate ElasticAST repo
ELASTIC_ROOT = Path(__file__).resolve().parents[3] / "ElasticAST"
sys.path.insert(0, str(ELASTIC_ROOT / "src"))

from models.elasticast import ElasticAST  # type: ignore


class ElasticASTForAudioClassification(nn.Module):
    """
    Wrapper for the GitHub ElasticAST model.
    Automatically infers sample_size from the input (mel-spec),
    and constructs the inner ElasticAST on first forward().
    """

    def __init__(
        self,
        num_labels: int = 5,
        n_mels: int = 128,
        patch_size: int = 16,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.n_mels = n_mels
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads

        # Model is created lazily because sample_width depends on input
        self.encoder = ElasticAST(
        sample_size=(self.n_mels, max_time_frames),
        patch_size=self.patch_size,
        num_classes=self.num_labels,
        dim=self.dim,
        depth=self.depth,
        heads=self.heads,
        channels=1
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def _build_encoder(self, sample_height: int, sample_width: int):
        """
        Build the ElasticAST model based on actual mel dimensions.
        """

        patch = self.patch_size

        # Ensure patch_size divides both dimensions
        if sample_height % patch != 0:
            raise ValueError(f"n_mels={sample_height} must be divisible by patch_size={patch}")
        if sample_width % patch != 0:
            raise ValueError(f"time_frames={sample_width} must be divisible by patch_size={patch}")

        self.encoder = ElasticAST(
            sample_size=(sample_height, sample_width),
            patch_size=patch,
            num_classes=self.num_labels,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            channels=1,  # spectrogram = 1 channel
        )

    def forward(self, input_values: torch.Tensor, labels=None) -> SequenceClassifierOutput:
        """
        input_values: (B, T, F) where F = mel bins (should equal n_mels)
        """

        # Standardize orientation: (B, F, T)
        x = input_values.transpose(1, 2)  # (B, F, T)

        B, F, T = x.shape

        # Lazy init ElasticAST using real sample size
        if self.encoder is None:
            self._build_encoder(sample_height=F, sample_width=T)

        # Prepare for ElasticAST: (B, 1, F, T)
        x = x.unsqueeze(1)

        logits = self.encoder(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(logits=logits, loss=loss)