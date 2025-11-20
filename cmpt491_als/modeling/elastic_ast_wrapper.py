import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

import sys
from pathlib import Path

# Path to ElasticAST repo:
# cmpt491-als/
#   ElasticAST/
#     src/
ELASTIC_ROOT = Path(__file__).resolve().parents[2] / "ElasticAST"
sys.path.insert(0, str(ELASTIC_ROOT / "src"))

# Import the actual ElasticAST model
from models.elasticast import ElasticAST


class ElasticASTForAudioClassification(nn.Module):
    """
    Wraps ElasticAST (raw waveform encoder) so it behaves like a HuggingFace classifier.
    """

    def __init__(self, num_labels: int):
        super().__init__()

        # Instantiate ElasticAST encoder
        self.encoder = ElasticAST()

        # ElasticAST exposes embedding dim as .embed_dim
        self.hidden_dim = self.encoder.embed_dim

        # Classification head
        self.classifier = nn.Linear(self.hidden_dim, num_labels)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_values, labels=None):
        """
        Args:
            input_values: (batch, waveform_length)
        """

        # Encoder output: (batch, seq_len, embed_dim)
        features = self.encoder(input_values)

        # Mean pool over time dimension
        pooled = features.mean(dim=1)  # -> (batch, embed_dim)

        # Classification logits
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss
        )