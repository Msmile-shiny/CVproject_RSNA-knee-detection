"""3D ResNet 模型 — Phase 3 轻量 3D 管线.

用于验证 3D 方法可跑通 (8GB VRAM), 为云端全量 3D 训练做准备.

输入: [B, 1, 32, 128, 128]  单通道 3D volume (32 个相邻切片)
骨干: ResNet3D-18 (timm)
输出: 12 logits

关键优化 (适配 8GB VRAM):
- Gradient Checkpointing: 用计算换显存, 只存激活检查点
- AMP fp16: 减半显存 + 加速
- batch_size=1 + 梯度累积
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from .head import ClassificationHead


class ResNet3DModel(nn.Module):
    """3D ResNet 用于膝关节 MRI volume 多标签分类.

    Args:
        in_channels: 输入通道数 (默认 1, 灰度 MRI)
        num_classes: 输出类别数 (默认 12)
        pretrained: 是否加载预训练权重
        dropout: 分类头 dropout
        feature_dim: backbone 输出特征维度 (ResNet3D-18 = 512)
        use_grad_checkpoint: True → 启用梯度检查点 (省 VRAM)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 12,
        pretrained: bool = True,
        dropout: float = 0.3,
        feature_dim: int = 512,
        use_grad_checkpoint: bool = True,
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("timm 未安装. pip install timm")

        self.feature_dim = feature_dim
        self.use_grad_checkpoint = use_grad_checkpoint

        # ── 3D Backbone ───────────────────────────────────────
        # ResNet3D-18: 输入 [B, C, D, H, W]
        try:
            self.backbone = timm.create_model(
                "resnet3d_18",
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=0,          # 去掉原始分类器
            )
        except Exception:
            # 备选: 某些 timm 版本可能用不同命名
            logger = __import__("logging").getLogger(__name__)
            logger.warning("resnet3d_18 不可用, 尝试 resnet18_3d...")
            self.backbone = timm.create_model(
                "resnet18_3d",
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=0,
            )

        # 获取实际 feature_dim (可能和预期不同)
        if hasattr(self.backbone, "num_features"):
            self.feature_dim = self.backbone.num_features

        # ── 3D → 1D 池化 ──────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # ── 分类头 ────────────────────────────────────────────
        self.head = ClassificationHead(
            in_features=self.feature_dim,
            hidden_features=self.feature_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取 backbone 特征 + 3D 池化.

        x: [B, C, D, H, W] → [B, feature_dim]
        """
        if self.use_grad_checkpoint and self.training:
            # 梯度检查点: 不存中间激活, 反向时重新计算
            features = checkpoint(
                self.backbone.forward_features, x,
                use_reentrant=False,
            )
        else:
            features = self.backbone.forward_features(x)

        # features: [B, D_feat, D', H', W'] → pool → [B, D_feat, 1, 1, 1]
        pooled = self.pool(features)
        return pooled.flatten(1)  # [B, D_feat]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            x: [B, 1, 32, 128, 128] 3D MRI volume

        Returns:
            logits: [B, 12]
        """
        features = self._forward_features(x)
        return self.head(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取池化后特征 (不做分类), 用于 ensemble."""
        return self._forward_features(x)
