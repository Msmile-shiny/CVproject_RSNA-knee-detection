"""多标签分类损失函数工厂.

用法:
    # 从 config 的 loss 段直接构建
    loss_cfg = {"name": "bce_with_logits", "label_smoothing": 0.05, "pos_weight": "auto"}
    criterion = build_loss(**loss_cfg)

    # pos_weight="auto" 会在训练开始时从训练集统计后设为 None (先用不加权的 BCE)
    # 后续可通过 criterion.pos_weight = computed_tensor 动态设置

支持的损失:
  - "bce_with_logits": BCEWithLogitsLoss + Label Smoothing (★ 当前主力)
  - "asl":             Asymmetric Loss (γ_neg=4, γ_pos=1) — 后续升级
  - "focal":           Focal Loss — 备选
  - "smooth_auc":      SmoothAUC Loss — 第三阶段尝试
"""

from __future__ import annotations

import torch.nn as nn


def build_loss(name: str, **kwargs) -> nn.Module:
    """根据名称构建损失函数.

    Args:
        name: "bce_with_logits" | "asl" | "focal" | "smooth_auc"
        **kwargs: 传给具体损失类的参数
                  注意: pos_weight="auto" 会被过滤, 后续从数据动态计算

    Returns:
        nn.Module 实例
    """
    # 过滤掉 "auto" 标记的参数 (需要训练时从数据动态计算)
    clean_kwargs = {k: v for k, v in kwargs.items() if v != "auto" and v is not None}

    name = name.lower()

    if name == "bce_with_logits":
        from .bce_smoothing import BCEWithLogitsLossSmooth
        return BCEWithLogitsLossSmooth(**clean_kwargs)

    if name == "asl":
        from .asymmetric_loss import AsymmetricLoss
        return AsymmetricLoss(**clean_kwargs)

    if name == "focal":
        from .focal_loss import FocalLoss
        return FocalLoss(**clean_kwargs)

    if name == "smooth_auc":
        from .smooth_auc import SmoothAUCLoss
        return SmoothAUCLoss(**clean_kwargs)

    raise ValueError(f"未知损失: {name}. 可用: bce_with_logits, asl, focal, smooth_auc")
