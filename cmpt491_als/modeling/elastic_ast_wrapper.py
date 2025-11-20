import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

import sys
from pathlib import Path

# Path to ElasticAST repo (ElasticAST/src/)
ELASTIC_ROOT = Path(__file__).resolve().parents[2] / "ElasticAST"
sys.path.insert(0, str(ELASTIC_ROOT / "src"))

from models.elasticast import ElasticAST


class ElasticASTForAudioClassification(nn.Module):
    """
    Wraps ElasticAST so it accepts AST spectrograms and outputs logits
    in a HuggingFace-like format.
    """

    def __init__(self, num_labels: int):
        super().__init__()

        # ASTFeatureExtractor produces:
        # (batch, time_frames, freq_bins) = (batch, 1024, 128)
        #
        # ElasticAST expects (freq, time)
        # must be divisible by patch_size:
        # 128 % 16 = 0
        # 1024 % 16 = 0
        #
        sample_size = (128, 1024)  # (freq_bins, time_frames)
        patch_size = 16
        dim = 768
        depth = 12
        heads = 12

        self.encoder = ElasticAST(
            sample_size=sample_size,
            patch_size=patch_size,
            num_classes=num_labels,
            dim=dim,
            depth=depth,
            heads=heads,
            channels=1,          # spectrograms have 1 channel
            dropout=0.0,
            emb_dropout=0.0,
            imagenet_pretrain=False,
            AST_pretrain=False,
            SSAST_pretrain=False,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_values, labels=None):
        """
        input_values: (batch, time, freq) from ASTFeatureExtractor
        ElasticAST expects (batch, freq, time)
        """
        x = input_values.transpose(1, 2)  # (batch, freq, time)

        logits = self.encoder(x)  # (batch, num_labels)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss
        )