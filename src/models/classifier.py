"""2D/2.5D 切片级多标签分类器.

支持:
- 2D:   单张切片 → [1, H, W] → ConvNeXtV2-Tiny → 12 logits
- 2.5D: 3 邻切片 → [3, H, W] → ConvNeXtV2-Tiny → 12 logits (原生 ImageNet 预训练)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import create_backbone, get_feature_dim


class KneeClassifier2D(nn.Module):
    """膝关节异常多标签分类器 (兼容 2D / 2.5D).

    Args:
        arch: timm backbone 名称 (默认 convnextv2_tiny)
        pretrained: 是否加载预训练权重
        in_channels: 1 = 2D 灰度, 3 = 2.5D triplet
        num_classes: 输出类别数 (12)
        dropout: 分类头 dropout
        drop_path_rate: stochastic depth rate
    """

    def __init__(
        self,
        arch: str = "convnextv2_tiny",
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 12,
        dropout: float = 0.3,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.arch = arch
        self.in_channels = in_channels

        self.backbone = create_backbone(
            arch,
            pretrained=pretrained,
            in_channels=in_channels,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = get_feature_dim(arch)

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → logits: [B, num_classes]."""
        features = self.backbone(x)               # list of feature maps
        x = features[-1]                          # 最后一个 stage
        x = x.mean(dim=[2, 3])                    # global avg pool
        return self.head(x)
