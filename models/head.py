"""Multi-label classification head.

Linear → LayerNorm → GELU → Dropout → Linear → 12 logits.

Sigmoid is NOT applied inside the model — use BCEWithLogitsLoss which
combines sigmoid + BCE in a numerically stable way.
"""

from __future__ import annotations

import torch.nn as nn


class ClassificationHead(nn.Module):
    """2-layer MLP classification head for 12-class multi-label output.

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden layer size (default 512)
        num_classes: Number of output classes (default 12)
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int = 384,
        hidden_features: int = 512,
        num_classes: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        """x: [B, in_features] → logits: [B, num_classes]."""
        return self.head(x)
