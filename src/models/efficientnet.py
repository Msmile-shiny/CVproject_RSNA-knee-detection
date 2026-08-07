"""EfficientNet backbone 工厂 — 基于 timm.

用于 2.5D 三平面模型的特征提取, 支持:
- tf_efficientnet_b0 ~ b4
- tf_efficientnetv2_s / m
- ImageNet / ImageNet-21k 预训练权重

所有 backbone 输出特征图, 供 triplane 融合模块使用.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def create_backbone(
    arch: str = "tf_efficientnet_b0",
    pretrained: bool = True,
    in_channels: int = 3,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    """创建 timm EfficientNet backbone, 返回特征提取器 (去掉分类头)."""
    backbone = timm.create_model(
        arch,
        pretrained=pretrained,
        in_chans=in_channels,
        drop_path_rate=drop_path_rate,
        features_only=True,          # 返回多尺度特征图
        num_classes=0,                # 去掉分类头
    )
    return backbone


def get_feature_dim(arch: str = "tf_efficientnet_b0") -> int:
    """查询 backbone 输出特征维度 (最后一层)."""
    m = timm.create_model(arch, pretrained=False, num_classes=0)
    return m.num_features
