"""多标签分类损失函数.

build_loss() 从配置构建损失:
- "bce_with_logits": BCEWithLogitsLoss + 可选的 label_smoothing
- "focal": Focal Loss
- "asl": Asymmetric Loss (ICCV 2021)
- "smooth_auc": SmoothAUC Loss
"""

from __future__ import annotations

import torch.nn as nn


def build_loss(name: str, **kwargs) -> nn.Module:
    """根据名称和参数构建损失函数.

    Args:
        name: 损失函数名 ("bce_with_logits" | "focal" | "asl" | "smooth_auc")
        **kwargs: 传递给具体损失构造函数的参数

    Returns:
        nn.Module 损失函数实例
    """
    name = name.lower()

    if name == "bce_with_logits":
        from .bce_smoothing import BCEWithLogitsLossSmooth
        return BCEWithLogitsLossSmooth(**kwargs)

    if name == "focal":
        from .focal_loss import FocalLoss
        return FocalLoss(**kwargs)

    if name == "asl":
        from .asymmetric_loss import AsymmetricLoss
        return AsymmetricLoss(**kwargs)

    if name == "smooth_auc":
        from .smooth_auc import SmoothAUCLoss
        return SmoothAUCLoss(**kwargs)

    raise ValueError(f"Unknown loss: {name}. Available: bce_with_logits, focal, asl, smooth_auc")
