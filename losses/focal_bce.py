"""Focal BCE Loss — 针对极度类别不平衡的多标签分类.

公式:
    FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)

用于多标签时, 每个类别独立计算 focal loss.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """Focal BCE Loss for multi-label classification.

    Args:
        gamma: 聚焦参数, 越大越关注难分样本 (默认 2.0)
        alpha: 正样本权重 (默认 0.25)
        reduction: "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,   # [B, C] raw logits
        targets: torch.Tensor,  # [B, C] binary labels (0/1)
    ) -> torch.Tensor:
        """计算 focal BCE loss.

        Args:
            logits: 模型原始输出 (未经过 sigmoid)
            targets: 二值标签

        Returns:
            scalar loss (reduction="mean" 时)
        """
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # [B, C]

        # 计算 p_t (模型对正确类别的置信度)
        probs = torch.sigmoid(logits)           # [B, C]
        p_t = targets * probs + (1 - targets) * (1 - probs)  # [B, C]

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha balancing
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        loss = alpha_weight * focal_weight * bce_loss  # [B, C]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # [B, C]
