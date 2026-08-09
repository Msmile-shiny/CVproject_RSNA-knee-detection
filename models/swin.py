"""Swin Transformer 2.5D 适配器 — 用于 Phase 4 Ensemble.

Swin-T 特点:
  - 层次化 Vision Transformer (shifted windows)
  - 原生 224×224, 这里用 384×384
  - feature_dim = 768
  - 相比 CNN, 提供不同的 inductive bias (ensemble diversity)
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


class Swin25D(nn.Module):
    """Swin-T 2.5D 模型.

    Args:
        in_channels: 输入通道数 (默认 5)
        num_classes: 输出类别数 (默认 12)
        pretrained: 是否加载 ImageNet-22K 预训练权重
        dropout: 分类头 dropout
        feature_dim: backbone 输出特征维度 (Swin-T = 768)
        window_size: Swin window size (默认 7)
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 12,
        pretrained: bool = True,
        dropout: float = 0.3,
        feature_dim: int = 768,
        window_size: int = 7,
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm 未安装. pip install timm")

        self.feature_dim = feature_dim

        # Swin-T: 层次化 Transformer
        # 注意: Swin 使用 patch_embed, 5ch 输入需要适配
        # timm 的 in_chans 参数会自动处理
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            img_size=384,               # override 默认 224
            features_only=False,
        )

        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features

        self.head = ClassificationHead(
            in_features=self.feature_dim,
            hidden_features=self.feature_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 5, H, W] → logits: [B, 12].

        Swin forward_features 返回 [B, N, D] (序列) 或 [B, D, H', W'].
        需要根据实际输出格式做池化.
        """
        features = self.backbone.forward_features(x)

        if features.dim() == 4:
            # [B, D, H', W'] → GAP → [B, D]
            features = features.mean(dim=[2, 3])
        elif features.dim() == 3:
            # [B, N, D] → mean over tokens → [B, D]
            features = features.mean(dim=1)

        return self.head(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取池化后特征 (不含分类头)."""
        features = self.backbone.forward_features(x)
        if features.dim() == 4:
            return features.mean(dim=[2, 3])
        elif features.dim() == 3:
            return features.mean(dim=1)
        return features
