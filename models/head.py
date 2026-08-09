"""多标签分类头.

架构:
    Linear → LayerNorm → GELU → Dropout(0.3) → Linear → 12 logits

注意: 不在 model 内部做 sigmoid, 使用 BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """2 层 MLP 分类头, 输出 12 类 logits.

    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层维度 (默认 512)
        num_classes: 输出类别数 (默认 12)
        dropout: dropout 率
    """

    def __init__(
        self,
        in_features: int,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_features] → logits: [B, num_classes]."""
        return self.head(x)
