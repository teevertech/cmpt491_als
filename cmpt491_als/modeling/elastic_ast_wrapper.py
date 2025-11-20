import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

# ---------------------------------------------------------------------
# Wire in the local ElasticAST repo
#   Expecting: /workspace/ElasticAST/src/models/elasticast.py
#   Adjust ELASTIC_ROOT if your layout is different.
# ---------------------------------------------------------------------
ELASTIC_ROOT = Path(__file__).resolve().parents[2] / "ElasticAST"
sys.path.insert(0, str(ELASTIC_ROOT / "src"))

from models.elasticast import ElasticAST  # type: ignore


class ElasticASTForAudioClassification(nn.Module):
    """
    Thin wrapper that:
      * takes log-Mel spectrograms of shape (B, T, F),
      * reshapes to (B, 1, F, T) for ElasticAST,
      * returns a standard HuggingFace-style SequenceClassifierOutput.

    NOTE: The exact __init__ signature of ElasticAST is defined in
    ElasticAST/src/models/elasticast.py. The arguments below follow the
    AST convention (label_dim, input_fdim, etc). If you get a
    TypeError about unexpected keywords, open that file and tweak the
    call in __init__ accordingly.
    """

    def __init__(
        self,
        num_labels: int = 5,
        input_fdim: int = 128,
        imagenet_pretrain: bool = False,
        audioset_pretrain: bool = False,
        model_size: str = "base384",
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels

        # IMPORTANT:
        #   - check ElasticAST.__init__ in the repo and adjust kwargs
        #     if names differ (e.g. label_dim vs num_classes).
        self.encoder = ElasticAST(
            label_dim=num_labels,
            input_fdim=input_fdim,
            imagenet_pretrain=imagenet_pretrain,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size,
            **kwargs,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        """
        Args
        ----
        input_values:
            Float tensor of shape (batch, time, freq) containing
            log-Mel spectrograms.
        labels:
            Optional int64 tensor of shape (batch,) with class indices
            in [0, num_labels - 1].

        Returns
        -------
        SequenceClassifierOutput
            .logits: (batch, num_labels)
            .loss: scalar (if labels is not None)
        """
        # (B, T, F)  →  (B, F, T)  →  (B, 1, F, T)
        x = input_values.transpose(1, 2)
        x = x.unsqueeze(1)

        logits = self.encoder(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(logits=logits, loss=loss)