import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification


class ElasticASTForAudioClassification(nn.Module):
    """
    A thin wrapper around HuggingFace's ElasticAST or AST models.
    This wrapper ensures shape correctness for input_values coming
    from ASTFeatureExtractor.

    Supports:
    - MIT/ast-finetuned-audioset-10-10-0.4593 (standard AST)
    - elastic-audio/elastic-ast-large (ElasticAST)
    - any other HF AST-like audio classification model
    """

    def __init__(self, model_name: str, num_labels: int = 5):
        super().__init__()

        # Load HF model
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            trust_remote_code=True,   # Required for ElasticAST
            ignore_mismatched_sizes=True,
        )

    def forward(self, input_values, labels=None):
        """
        input_values: (batch, time, freq) or (time, freq)
        labels: (batch,)
        """

        # Add batch dimension if missing
        if input_values.dim() == 2:
            input_values = input_values.unsqueeze(0)

        # AST expects (batch, channels=1, time, freq)
        if input_values.dim() == 3:
            input_values = input_values.unsqueeze(1)

        # Forward through HF model
        outputs = self.model(
            input_values=input_values,
            labels=labels
        )

        return outputs