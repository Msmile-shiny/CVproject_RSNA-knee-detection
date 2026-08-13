"""ConvNeXtV2-Base 2.5D 适配器 — 用于 Multi-Arch Teacher Ensemble.

ConvNeXtV2-B 相比 ConvNeXt-S 的升级:
  - GRN (Global Response Normalization): 增强通道间特征竞争, 减少死特征
  - FCMAE (Fully Convolutional Masked Autoencoder) 预训练: 类似 MAE 但纯卷积
  - 更大参数量: 88M (ConvNeXtV2-B) vs 50M (ConvNeXt-S)
  - feature_dim = 1024

输入格式: [B, 5, H, W] — 5 张相邻切片堆叠为通道维度.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from .head import ClassificationHead

logger = logging.getLogger(__name__)

# ── ConvNeXtV2-Base 模型名优先级 ──────────────────────────────
_CONVNEXTV2_VARIANTS = [
    "convnextv2_base.fcmae_ft_in22k_in1k_384",   # FCMAE pretrained, IN22K fine-tuned @384
    "convnextv2_base.fcmae_ft_in22k_in1k",         # FCMAE pretrained, IN22K fine-tuned @224
    "convnextv2_base.fcmae_ft_in1k",               # FCMAE pretrained, IN1K fine-tuned
    "convnextv2_base",                              # 任意基础 variant
]


def _load_convnextv2_backbone(pretrained: bool = True):
    """加载 ConvNeXtV2-Base backbone, 按优先级尝试多个模型名."""
    if not HAS_TIMM:
        raise ImportError("timm 未安装. pip install timm")

    for variant in _CONVNEXTV2_VARIANTS:
        try:
            backbone = timm.create_model(
                variant,
                pretrained=pretrained,
                num_classes=0,
            )
            logger.info("ConvNeXtV2 backbone 加载成功: %s", variant)
            return backbone, variant
        except (RuntimeError, AttributeError, KeyError) as e:
            logger.debug("ConvNeXtV2 variant '%s' 不可用: %s", variant, e)
            continue

    raise RuntimeError(
        f"无法加载 ConvNeXtV2-Base backbone. 已尝试: {_CONVNEXTV2_VARIANTS}\n"
        "请升级 timm: pip install timm>=0.9.8"
    )


class ConvNeXtV2Base25D(nn.Module):
    """ConvNeXtV2-Base 2.5D 模型.

    Args:
        in_channels: 输入通道数 (默认 5)
        num_classes: 输出类别数 (默认 12)
        pretrained: 是否加载 FCMAE/ImageNet-22K 预训练权重
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

        # ── 加载 ConvNeXtV2-Base backbone ────────────────────
        self.backbone, self.model_name = _load_convnextv2_backbone(pretrained=pretrained)

        # ── 适配 5ch 输入 ────────────────────────────────────
        # ConvNeXtV2 stem: Conv2d(3→128, kernel=4, stride=4)
        # 5ch 输入需要替换第一层
        if in_channels != 3:
            old_stem = self.backbone.stem
            # 尝试定位 stem 中的第一个 conv
            if hasattr(old_stem, '0'):
                old_conv = old_stem[0]   # Sequential 结构
            elif hasattr(old_stem, 'conv'):
                old_conv = old_stem.conv
            else:
                # fallback: stem 本身是 Conv2d
                old_conv = old_stem

            if isinstance(old_conv, nn.Conv2d) and old_conv.in_channels != in_channels:
                new_conv = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                # Scaled-replicate init
                with torch.no_grad():
                    old_weight = old_conv.weight  # [128, 3, 4, 4]
                    repeats = (in_channels + 2) // 3
                    new_weight = old_weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                    scale = 3.0 / in_channels
                    new_weight *= scale
                    new_conv.weight.copy_(new_weight)
                    if old_conv.bias is not None:
                        new_conv.bias.copy_(old_conv.bias)

                if hasattr(old_stem, '0'):
                    old_stem[0] = new_conv
                elif hasattr(old_stem, 'conv'):
                    old_stem.conv = new_conv
                else:
                    self.backbone.stem = new_conv

                logger.info(
                    "ConvNeXtV2 stem 通道适配: %d → %d (scale=%.3f)",
                    3, in_channels, scale,
                )

        # ── 特征维度 ─────────────────────────────────────────
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features
        else:
            self.feature_dim = 1024  # ConvNeXtV2-Base

        # ── 分类头 ───────────────────────────────────────────
        self.head = ClassificationHead(
            in_features=self.feature_dim,
            hidden_features=self.feature_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

        logger.info(
            "ConvNeXtV2Base25D: variant=%s feature_dim=%d in_ch=%d",
            self.model_name, self.feature_dim, in_channels,
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
