import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

# ElasticAST imports (repo must be cloned at project root /ElasticAST)
import sys
from pathlib import Path

ELASTIC_ROOT = Path(__file__).resolve().parents[2] / "ElasticAST"
sys.path.insert(0, str(ELASTIC_ROOT / "modeling"))

# Correct: ElasticAST encoder is named `ElasticAST`
from network import ElasticAST


class ElasticASTForAudioClassification(nn.Module):
    """
    Wraps ElasticAST so it behaves like HuggingFace ASTForAudioClassification.
    """

    def __init__(self, num_labels: int):
        super().__init__()

        # Create ElasticAST encoder
        self.encoder = ElasticAST()

        # Correct: embedding dimension stored here
        self.hidden_dim = self.encoder.embed_dim

        # Classification head
        self.classifier = nn.Linear(self.hidden_dim, num_labels)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_values, labels=None):
        """
        Args:
            input_values: (batch, waveform_length)
        Returns:
            SequenceClassifierOutput(logits=..., loss=optional)
        """
        # Encoder returns (batch, seq_len, hidden_dim)
        features = self.encoder(input_values)

        # Mean-pool over the patch dimension
        pooled = features.mean(dim=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss
        )
