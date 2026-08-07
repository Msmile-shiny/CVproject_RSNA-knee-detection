"""2.5D 三平面融合模型.

架构:
  MRI Volume
    ├─ Axial plane   (3 相邻切片) → Shared Backbone → feat_a  [B, D]
    ├─ Coronal plane  (3 相邻切片) → Shared Backbone → feat_c  [B, D]
    └─ Sagittal plane (3 相邻切片) → Shared Backbone → feat_s  [B, D]
                        ↓
              Fusion (Concat / Cross-Attention)
                        ↓
                 12-class Head

三个平面**共享同一个 backbone 权重**, 减少参数量, 方便 8GB 训练.

融合策略:
- concat: 拼接 [feat_a, feat_c, feat_s] → Linear → Head (推荐起步)
- cross_attn: feat_a 作为 query, feat_c/feat_s 作为 key/value (可选升级)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import create_backbone, get_feature_dim


class TriplaneModel(nn.Module):
    """2.5D 三平面多标签分类模型.

    Args:
        arch: timm backbone 名称
        pretrained: 是否使用预训练权重
        num_classes: 输出类别数 (默认 12)
        fusion: 融合方式 ("concat" | "cross_attn")
        dropout: 分类头 dropout 率
    """

    def __init__(
        self,
        arch: str = "convnextv2_tiny",
        pretrained: bool = True,
        num_classes: int = 12,
        fusion: str = "concat",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.arch = arch
        self.fusion_type = fusion

        # 三个平面共享同一个 backbone
        self.backbone = create_backbone(arch, pretrained=pretrained, in_channels=3)

        feat_dim = get_feature_dim(arch)

        if fusion == "concat":
            fusion_dim = feat_dim * 3
            self.fusion = nn.Identity()
        elif fusion == "cross_attn":
            fusion_dim = feat_dim
            self.fusion = CrossAttentionFusion(feat_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        axial: torch.Tensor,       # [B, 3, H, W]
        coronal: torch.Tensor,     # [B, 3, H, W]
        sagittal: torch.Tensor,    # [B, 3, H, W]
    ) -> torch.Tensor:
        """返回 logits [B, num_classes]."""
        f_a = self.backbone(axial)[-1]       # 取最后一层特征
        f_c = self.backbone(coronal)[-1]
        f_s = self.backbone(sagittal)[-1]

        # 全局平均池化特征 → [B, D]
        f_a = F.adaptive_avg_pool2d(f_a, 1).flatten(1)
        f_c = F.adaptive_avg_pool2d(f_c, 1).flatten(1)
        f_s = F.adaptive_avg_pool2d(f_s, 1).flatten(1)

        if self.fusion_type == "concat":
            fused = torch.cat([f_a, f_c, f_s], dim=1)   # [B, 3D]
        else:
            fused = self.fusion(f_a, f_c, f_s)           # [B, D]

        return self.head(fused)


class CrossAttentionFusion(nn.Module):
    """以 Axial 为 query, Coronal 和 Sagittal 为 key/value 做交叉注意力.

    Args:
        dim: 特征维度
        num_heads: 注意力头数
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn_c = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_s = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        f_a: torch.Tensor,   # [B, D]
        f_c: torch.Tensor,   # [B, D]
        f_s: torch.Tensor,   # [B, D]
    ) -> torch.Tensor:
        # 扩展维度 [B, D] → [B, 1, D]
        q = f_a.unsqueeze(1)
        k_c = v_c = f_c.unsqueeze(1)
        k_s = v_s = f_s.unsqueeze(1)

        out_c, _ = self.attn_c(q, k_c, v_c)
        out_s, _ = self.attn_s(q, k_s, v_s)

        fused = f_a + out_c.squeeze(1) + out_s.squeeze(1)
        return self.norm(fused)
