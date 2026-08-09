"""切片注意力池化模块.

将同一 series 的多个切片特征聚合为单个 series 特征:
    F = Σ α_i · f_i
    α = softmax(Linear(f_i, 1))

可微, 替代硬 Top-K 聚合.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceAttention(nn.Module):
    """Learnable weighted pooling over slice features.

    Args:
        feature_dim: 每个切片特征的维度
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.scorer = nn.Linear(feature_dim, 1)

    def forward(
        self,
        features: torch.Tensor,       # [N_slices, D] or [B, N_slices, D]
        mask: torch.Tensor | None = None,  # [N_slices] bool (True = valid)
    ) -> torch.Tensor:
        """加权聚合切片特征.

        Args:
            features: 切片特征
            mask: 有效切片 mask (用于变长序列 padding)

        Returns:
            pooled: [D] or [B, D] 聚合后的 series 特征
        """
        squeeze_batch = features.dim() == 2

        if squeeze_batch:
            features = features.unsqueeze(0)  # [1, N, D]

        scores = self.scorer(features).squeeze(-1)  # [B, N]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)          # [B, N]
        pooled = (features * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        if squeeze_batch:
            pooled = pooled.squeeze(0)
        return pooled
