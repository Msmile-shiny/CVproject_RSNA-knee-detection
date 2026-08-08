"""Backbone 工厂 — 基于 timm, 支持多架构和多通道适配.

用于 2D/2.5D 分类器的特征提取, 支持:

CNN 系列:
  - convnextv2_tiny / small / base   (★ 推荐, ImageNet-22k 预训练, 28M-89M)
  - convnext_tiny / small / base
  - tf_efficientnet_b0 ~ b4
  - tf_efficientnetv2_s / m
  - resnet50 / resnet101

Transformer 系列 (备选):
  - swin_tiny_patch4_window7_224

通道适配:
  - in_channels=3 (2.5D triplet): 直接用 ImageNet 预训练权重, 不需要修改
  - in_channels=1 (2D 单切片):   自动取 RGB 三通道均值适配到单通道
"""

from __future__ import annotations

import timm
import torch.nn as nn


# ── 模型注册表 ──────────────────────────────────────────────
# {arch: (标准输入尺寸, 输出特征维度, 预训练来源)}
MODEL_REGISTRY: dict[str, tuple[int, int, str]] = {
    # ConvNeXt V2 (★ 当前主力)
    "convnextv2_tiny":   (224, 768,  "imagenet22k"),
    "convnextv2_small":  (224, 768,  "imagenet22k"),
    "convnextv2_base":   (224, 1024, "imagenet22k"),
    # ConvNeXt V1
    "convnext_tiny":     (224, 768,  "imagenet22k"),
    "convnext_small":    (224, 768,  "imagenet22k"),
    # EfficientNet (快速实验)
    "tf_efficientnet_b0": (256, 1280, "imagenet"),
    "tf_efficientnet_b2": (288, 1408, "imagenet"),
    "tf_efficientnetv2_s": (384, 1280, "imagenet21k"),
    # ResNet (RadImageNet 兼容)
    "resnet50":          (224, 2048, "imagenet"),
    # Swin
    "swin_tiny_patch4_window7_224": (224, 768, "imagenet22k"),
}


def create_backbone(
    arch: str = "convnextv2_tiny",
    pretrained: bool = True,
    in_channels: int = 3,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    """创建 timm backbone, 返回多尺度特征提取器 (features_only 模式).

    当 in_channels != 3 时, 自动适配第一层卷积权重:
      - in_channels=1: 取预训练 RGB 三通道权重的均值
      - in_channels>3: 复制预训练权重并取均值扩展到多通道

    Args:
        arch: timm 模型名 (见 MODEL_REGISTRY)
        pretrained: 是否加载预训练权重
        in_channels: 输入通道数 (3 = 2.5D triplet, 1 = 2D 单切片)
        drop_path_rate: stochastic depth rate

    Returns:
        带 features_only=True 的 backbone, forward 返回各 stage 特征图列表
    """
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"未知架构: {arch}. 可用: {list(MODEL_REGISTRY)}")

    backbone = timm.create_model(
        arch,
        pretrained=pretrained,
        in_chans=in_channels,
        drop_path_rate=drop_path_rate,
        features_only=True,
        num_classes=0,
    )

    # 如果 in_channels != 3, timm 内部已做了权重适配
    # (timm>=0.9 原生支持, 不需要手动处理)
    return backbone


def get_feature_dim(arch: str = "convnextv2_tiny") -> int:
    """查询 backbone 最后一层输出的特征维度 (分类头输入维度)."""
    if arch in MODEL_REGISTRY:
        return MODEL_REGISTRY[arch][1]

    # fallback: 直接创建模型查
    m = timm.create_model(arch, pretrained=False, num_classes=0)
    return m.num_features


def get_input_size(arch: str = "convnextv2_tiny") -> int:
    """查询 backbone 的标准输入尺寸."""
    if arch in MODEL_REGISTRY:
        return MODEL_REGISTRY[arch][0]
    return 224
