"""BCEWithLogitsLoss + Label Smoothing + pos_weight 动态计算.

起步基线损失, 直接优化每个二分类目标.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLossSmooth(nn.Module):
    """带 label smoothing 和 pos_weight 的 BCEWithLogitsLoss.

    Args:
        label_smoothing: 标签平滑系数 (0 表示不平滑)
        pos_weight: 正样本权重, shape [num_classes] 或 None (自动计算)
    """

    def __init__(self, label_smoothing: float = 0.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
