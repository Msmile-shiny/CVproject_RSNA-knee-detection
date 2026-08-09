"""Focal BCE Loss for multi-label classification.

Addresses extreme class imbalance (90%+ of labels are negative for most classes).

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    γ=2: Focus on hard examples (misclassified or low-confidence)
    α=0.25: Down-weight easy negatives

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """Focal BCE Loss for 12-class multi-label classification.

    Uses BCEWithLogitsLoss internally (numerically stable sigmoid + BCE).

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
        alpha: Positive class weight (1-α for negatives).
        reduction: "mean" (default), "sum", or "none"
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
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal BCE loss.

        Args:
            logits:  [B, C] raw model outputs (before sigmoid)
            targets: [B, C] binary labels (0 or 1)

        Returns:
            Scalar loss (reduction="mean") or per-element loss.
        """
        # Standard BCE with logits (stable)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # [B, C]

        # p_t: model confidence for the correct class
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha balancing
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        loss = alpha_weight * focal_weight * bce  # [B, C]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
