import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

import sys
from pathlib import Path

# Path to ElasticAST repo:
# project_root/
#   cmpt491_als/
#   ElasticAST/
#     src/
ELASTIC_ROOT = Path(__file__).resolve().parents[2] / "ElasticAST"
sys.path.insert(0, str(ELASTIC_ROOT / "src"))

from models.elasticast import ElasticAST


class ElasticASTForAudioClassification(nn.Module):
    """
    Wraps ElasticAST so it behaves like a HuggingFace-style classifier.
    """

    def __init__(self, num_labels: int):
        super().__init__()

        # 🔧 CONFIG: these are reasonable defaults based on the AST/ElasticAST paper
        # and code (10s audio, 128 mel bins, ViT-B/16-ish config).
        #
        # You can tweak these later, but this will get you past the constructor error.
        sample_size = (128, 1000)   # (frequency_bins, time_frames)
        patch_size = 16
        dim        = 768            # embedding dimension
        depth      = 12             # number of transformer blocks
        heads      = 12             # attention heads

        self.model = ElasticAST(
            sample_size=sample_size,
            patch_size=patch_size,
            num_classes=num_labels,
            dim=dim,
            depth=depth,
            heads=heads,
            channels=1,              # audio spectrogram is usually single-channel
            dropout=0.0,
            emb_dropout=0.0,
            token_dropout_prob=None,
            imagenet_pretrain=False,
            SSAST_pretrain=False,
            AST_pretrain=False,
            avg_pool_tk=False,
            random_token_dropout=0,
            eval_token_dropout=0,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_values, labels=None):
        """
        Args:
            input_values: model input batch
        Returns:
            SequenceClassifierOutput(logits=..., loss=optional)
        """
        # ElasticAST's forward returns logits (batch, num_classes)
        logits = self.model(input_values)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss,
        )