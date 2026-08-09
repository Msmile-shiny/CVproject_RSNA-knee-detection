"""EfficientNetV2-S 2.5D 模型.

输入: [B, 5, 384, 384] 5 张相邻切片堆叠
骨干: EfficientNetV2-S (timm)
      - 第一层 conv 输入通道 3→5
      - 去掉原始分类器
      - 输出 feature_dim=1280
分类头: Linear → LayerNorm → GELU → Dropout → Linear → 12 logits
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


class EfficientNetV2S25D(nn.Module):
    """EfficientNetV2-S 适配 2.5D 5 通道输入.

    Args:
        in_channels: 输入通道数 (默认 5)
        num_classes: 输出类别数 (默认 12)
        pretrained: 是否加载 ImageNet 预训练权重
        dropout: 分类头 dropout
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 12,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm 未安装. pip install timm")

        # 创建 EfficientNetV2-S backbone
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,         # 去掉原始分类器
            features_only=False,   # 只取最终特征
        )

        # 特征维度
        self.feature_dim = self.backbone.num_features  # 1280

        # 分类头
        self.head = ClassificationHead(
            in_features=self.feature_dim,
            hidden_features=512,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 5, H, W] → logits: [B, 12]."""
        features = self.backbone.forward_features(x)   # [B, 1280]
        return self.head(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征 (不做分类), 用于 slice attention 等下游."""
        return self.backbone.forward_features(x)        # [B, 1280]
