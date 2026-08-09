"""ConvNeXt-S 2.5D 适配器 — 用于 Phase 4 Ensemble.

复用 EfficientNetV2S25D 的架构模式:
  - timm backbone → GAP → ClassificationHead
  - extract_features() 返回池化特征 (不含 head)

ConvNeXt-S 特点:
  - 现代 CNN (depthwise conv + inverted bottleneck)
  - 输入 224×224 原生, 这里用 384×384
  - feature_dim = 768
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from .head import ClassificationHead


class ConvNeXt25D(nn.Module):
    """ConvNeXt-S 2.5D 模型.

    Args:
        in_channels: 输入通道数 (默认 5)
        num_classes: 输出类别数 (默认 12)
        pretrained: 是否加载 ImageNet-22K 预训练权重
        dropout: 分类头 dropout
        feature_dim: backbone 输出特征维度 (ConvNeXt-S = 768)
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 12,
        pretrained: bool = True,
        dropout: float = 0.3,
        feature_dim: int = 768,
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm 未安装. pip install timm")

        self.feature_dim = feature_dim

        # ConvNeXt-S: 现代 CNN, 原生 224×224, 但支持任意尺寸
        self.backbone = timm.create_model(
            "convnext_small",
            pretrained=pretrained,
            in_chans=in_channels,       # 5 通道输入
            num_classes=0,              # 去掉原始分类器
            features_only=False,
        )

        # 读取实际特征维度
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features

        self.head = ClassificationHead(
            in_features=self.feature_dim,
            hidden_features=self.feature_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 5, H, W] → logits: [B, 12]."""
        features = self.backbone.forward_features(x)   # [B, D, H', W']
        features = features.mean(dim=[2, 3])            # [B, D] GAP
        return self.head(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取池化后特征 (不含分类头)."""
        features = self.backbone.forward_features(x)
        return features.mean(dim=[2, 3])
