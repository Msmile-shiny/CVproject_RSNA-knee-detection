"""DINOv2-B 2.5D 适配器 — 用于 Multi-Arch Teacher Ensemble.

DINOv2-B 特点:
  - Plain ViT with global self-attention (strong semantic features)
  - 预训练数据: LVD-142M (142M curated images, 远超 ImageNet)
  - Registers (4 tokens) 版本 — 对 dense/pixel-level 特征更友好
  - feature_dim = 768
  - 对膝关节 OA 分级、整体结构等全局判断特别强

输入适配:
  - 原生 patch_embed: Conv2d(3→768, kernel=14, stride=14)
  - 替换为: Conv2d(5→768, kernel=14, stride=14), 权重做 scaled-replicate init
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from .head import ClassificationHead

logger = logging.getLogger(__name__)

# ── 可用的 DINOv2 模型名 (按优先级排列) ───────────────────────
# timm >= 0.9.x 支持的 DINOv2 变体:
_DINOV2_VARIANTS = [
    "vit_base_patch14_dinov2.lvd142m",    # DINOv2-B + registers, LVD-142M
    "vit_base_patch14_reg4_dinov2.lvd142m",  # 显式 registers 4
    "dinov2_vitb14_reg",                    # 备用名
]


def _load_dinov2_backbone(pretrained: bool = True, img_size: int = 384):
    """加载 DINOv2-B backbone, 按优先级尝试多个模型名.

    Returns:
        (backbone, model_name)
    """
    if not HAS_TIMM:
        raise ImportError("timm 未安装. pip install timm")

    for variant in _DINOV2_VARIANTS:
        try:
            backbone = timm.create_model(
                variant,
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
            )
            logger.info("DINOv2 backbone 加载成功: %s", variant)
            return backbone, variant
        except (RuntimeError, AttributeError, KeyError) as e:
            logger.debug("DINOv2 variant '%s' 不可用: %s", variant, e)
            continue

    raise RuntimeError(
        f"无法加载 DINOv2 backbone. 已尝试: {_DINOV2_VARIANTS}\n"
        "请升级 timm: pip install timm>=0.9.8"
    )


def _adapt_patch_embed_5ch(backbone, in_channels: int = 5):
    """将 DINOv2 的 patch_embed 从 3ch 适配到 in_channels ch.

    DINOv2 patch_embed 结构: Conv2d(3, 768, kernel=14, stride=14)
    权重初始化策略: 将 3ch 预训练权重沿通道维度 replicate-scaled.
    """
    patch_embed = backbone.patch_embed
    old_conv = patch_embed.proj  # Conv2d

    if old_conv.in_channels == in_channels:
        return  # 已经匹配

    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    # Scaled-replicate: 将 3ch 权重重复到 5ch, 然后除以缩放因子
    with torch.no_grad():
        old_weight = old_conv.weight  # [768, 3, 14, 14]
        repeats = math.ceil(in_channels / 3)
        new_weight = old_weight.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
        # 缩放: 使输出激活的方差不变
        scale = 3.0 / in_channels
        new_weight *= scale
        new_conv.weight.copy_(new_weight)

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    patch_embed.proj = new_conv
    logger.info(
        "DINOv2 patch_embed 通道适配: %d → %d (scale=%.3f)",
        3, in_channels, scale,
    )


class DinoV225D(nn.Module):
    """DINOv2-B 2.5D 模型.

    将相邻的 in_channels 张切片堆叠为通道维度, 送入 DINOv2-B ViT.
    输出通过 CLS token → ClassificationHead → 12 logits.

    Args:
        in_channels: 堆叠切片数 (默认 5)
        num_classes: 输出类别数 (默认 12)
        pretrained: 是否加载 DINOv2 预训练权重
        dropout: 分类头 dropout
        img_size: 输入图像尺寸 (默认 384)
        use_cls_token: True → CLS token, False → mean pool over patches
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 12,
        pretrained: bool = True,
        dropout: float = 0.3,
        img_size: int = 384,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.use_cls_token = use_cls_token

        # ── 加载 DINOv2-B backbone ──────────────────────────
        self.backbone, self.model_name = _load_dinov2_backbone(
            pretrained=pretrained, img_size=img_size,
        )

        # ── 适配 5ch 输入 ────────────────────────────────────
        _adapt_patch_embed_5ch(self.backbone, in_channels)

        # ── 特征维度 ─────────────────────────────────────────
        self.feature_dim = self.backbone.embed_dim  # 768 for DINOv2-B

        # ── 分类头 ───────────────────────────────────────────
        self.head = ClassificationHead(
            in_features=self.feature_dim,
            hidden_features=self.feature_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

        logger.info(
            "DinoV225D: variant=%s feature_dim=%d in_ch=%d img_size=%d",
            self.model_name, self.feature_dim, in_channels, img_size,
        )

    def _extract_token_features(self, x: torch.Tensor) -> torch.Tensor:
        """从 DINOv2 backbone 提取特征.

        DINOv2 forward_features 在不同 timm 版本可能返回:
          - [B, N+1, D] — patch tokens + CLS (常见)
          - [B, N+5, D] — patch tokens + CLS + 4 registers
          - dict-like 对象

        这里统一处理为 [B, D].
        """
        output = self.backbone.forward_features(x)

        if isinstance(output, dict):
            # 某些 timm 版本返回 dict
            if "x_norm_patchtokens" in output:
                # DINOv2 风格的 dense output
                tokens = output["x_norm_patchtokens"]      # [B, N, D]
                if self.use_cls_token and "x_norm_clstoken" in output:
                    token = output["x_norm_clstoken"][:, 0]  # [B, D]
                else:
                    token = tokens.mean(dim=1)
                return token
            elif "x" in output:
                tokens = output["x"]
            elif "features" in output:
                tokens = output["features"]
            else:
                raise ValueError(f"未知 DINOv2 输出格式: {list(output.keys())}")
        elif isinstance(output, (list, tuple)):
            tokens = output[-1]  # 取最后一层
        else:
            tokens = output  # tensor

        # tokens: [B, N_tokens, D]
        if tokens.dim() == 4:
            # [B, D, H', W'] → GAP → [B, D]
            return tokens.mean(dim=[2, 3])

        if self.use_cls_token and tokens.shape[1] > 1:
            # 第 0 个 token 是 CLS
            return tokens[:, 0]    # [B, D]
        else:
            return tokens.mean(dim=1)   # [B, D] mean pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 5, H, W] → logits: [B, 12]."""
        features = self._extract_token_features(x)
        return self.head(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取池化后特征 (不含分类头), 用于 ensemble."""
        return self._extract_token_features(x)
