"""Tri-Plane 2.5D 模型 — 三平面独立编码 + 融合 + 分类.

架构:
  Sagittal 5-slice → Backbone → f_sag [B, D]
  Coronal  5-slice → Backbone → f_cor [B, D]
  Axial    5-slice → Backbone → f_ax  [B, D]
                          ↓
              MultiPlaneFusion (concat/transformer)
                          ↓
              ClassificationHead → 12 logits

支持:
- 共享 backbone (默认, 节省 VRAM)
- 独立 backbone (每平面独立权重)
- 缺失平面 mask (某些 study 可能缺少某个平面)
- 各平面独立 slice attention (可选)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .efficientnet25d import EfficientNetV2S25D
from .fusion import MultiPlaneFusion
from .head import ClassificationHead
from .attention import SliceAttention


class TriPlaneModel(nn.Module):
    """三平面 2.5D 模型.

    Args:
        in_channels: 每平面输入通道数 (默认 5)
        num_classes: 输出类别数 (默认 12)
        feature_dim: backbone 输出特征维度 (默认 1280)
        pretrained: backbone 是否加载预训练权重
        dropout: 分类头 dropout 率
        fusion: 融合方式 "concat" | "transformer"
        fusion_heads: transformer 头数
        fusion_layers: transformer 层数
        shared_backbone: True→三平面共享一个 backbone
        use_slice_attention: True→各平面前加 slice attention
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 12,
        feature_dim: int = 1280,
        pretrained: bool = True,
        dropout: float = 0.3,
        fusion: str = "concat",
        fusion_heads: int = 8,
        fusion_layers: int = 2,
        shared_backbone: bool = True,
        use_slice_attention: bool = False,
    ):
        super().__init__()
        self.shared_backbone = shared_backbone
        self.feature_dim = feature_dim

        # ── Backbone(s) ──────────────────────────────────────
        if shared_backbone:
            self.backbone = EfficientNetV2S25D(
                in_channels=in_channels,
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
            )
            # 去掉分类头 — 只用 extract_features
            self.backbone.head = nn.Identity()
        else:
            self.backbone_sag = EfficientNetV2S25D(
                in_channels=in_channels, num_classes=num_classes,
                pretrained=pretrained, dropout=dropout,
            )
            self.backbone_sag.head = nn.Identity()
            self.backbone_cor = EfficientNetV2S25D(
                in_channels=in_channels, num_classes=num_classes,
                pretrained=pretrained, dropout=dropout,
            )
            self.backbone_cor.head = nn.Identity()
            self.backbone_ax = EfficientNetV2S25D(
                in_channels=in_channels, num_classes=num_classes,
                pretrained=pretrained, dropout=dropout,
            )
            self.backbone_ax.head = nn.Identity()

        # ── Slice Attention (可选) ────────────────────────────
        self.use_slice_attention = use_slice_attention
        if use_slice_attention:
            self.sag_attn = SliceAttention(feature_dim)
            self.cor_attn = SliceAttention(feature_dim)
            self.ax_attn = SliceAttention(feature_dim)

        # ── 融合 ──────────────────────────────────────────────
        self.fusion = MultiPlaneFusion(
            feature_dim=feature_dim,
            fusion=fusion,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
        )

        # ── 分类头 ────────────────────────────────────────────
        self.head = ClassificationHead(
            in_features=feature_dim,
            hidden_features=feature_dim // 2,  # 512
            num_classes=num_classes,
            dropout=dropout,
        )

        # ── 缺失平面 embedding (learnable fallback) ────────────
        self.missing_emb = nn.Parameter(torch.zeros(1, feature_dim))

    def _encode_plane(
        self,
        x: torch.Tensor,
        backbone: EfficientNetV2S25D,
        attn: SliceAttention | None = None,
    ) -> torch.Tensor:
        """编码单个平面的 5-slice stack → [B, feature_dim].

        Args:
            x: [B, 5, H, W] 或全零 (缺失平面)
            backbone: EfficientNetV2S25D backbone
            attn: 可选的 slice attention

        Returns:
            [B, feature_dim]
        """
        # 检测缺失平面 (全零 tensor)
        batch_has_signal = x.reshape(x.shape[0], -1).abs().sum(dim=1) > 0  # [B]

        # 全部缺失 → 返回 learnable embedding
        if not batch_has_signal.any():
            return self.missing_emb.expand(x.shape[0], -1)

        # 有信号的 batch 正常前向
        features = backbone.extract_features(x)  # [B, D]

        if attn is not None:
            # SliceAttention 需要 [B, N_slices, D] 格式
            # 当前 extract_features 返回 [B, D], 已做 GAP
            # 如果要 slice-level attention, 需要修改 backbone
            pass

        # 缺失样本用 learnable embedding 替代
        if not batch_has_signal.all():
            missing_mask = ~batch_has_signal
            features[missing_mask] = self.missing_emb.expand(
                missing_mask.sum(), -1
            ).to(features.dtype)

        return features

    def forward(
        self,
        x_sag: torch.Tensor,   # [B, 5, H, W]  Sagittal 5-slice stack
        x_cor: torch.Tensor,   # [B, 5, H, W]  Coronal
        x_ax: torch.Tensor,    # [B, 5, H, W]  Axial
    ) -> torch.Tensor:
        """前向传播.

        Returns:
            logits: [B, num_classes]
        """
        # ── 编码各平面 ────────────────────────────────────────
        if self.shared_backbone:
            f_sag = self._encode_plane(x_sag, self.backbone)
            f_cor = self._encode_plane(x_cor, self.backbone)
            f_ax = self._encode_plane(x_ax, self.backbone)
        else:
            f_sag = self._encode_plane(
                x_sag, self.backbone_sag,
                self.sag_attn if self.use_slice_attention else None,
            )
            f_cor = self._encode_plane(
                x_cor, self.backbone_cor,
                self.cor_attn if self.use_slice_attention else None,
            )
            f_ax = self._encode_plane(
                x_ax, self.backbone_ax,
                self.ax_attn if self.use_slice_attention else None,
            )

        # ── 融合 ──────────────────────────────────────────────
        fused = self.fusion(f_sag, f_cor, f_ax)  # [B, D]

        # ── 分类 ──────────────────────────────────────────────
        return self.head(fused)

    def get_plane_features(
        self,
        x_sag: torch.Tensor,
        x_cor: torch.Tensor,
        x_ax: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """提取各平面特征 (用于分析).

        Returns:
            {"sag": [B,D], "cor": [B,D], "ax": [B,D], "fused": [B,D]}
        """
        if self.shared_backbone:
            f_sag = self._encode_plane(x_sag, self.backbone)
            f_cor = self._encode_plane(x_cor, self.backbone)
            f_ax = self._encode_plane(x_ax, self.backbone)
        else:
            f_sag = self._encode_plane(x_sag, self.backbone_sag)
            f_cor = self._encode_plane(x_cor, self.backbone_cor)
            f_ax = self._encode_plane(x_ax, self.backbone_ax)

        fused = self.fusion(f_sag, f_cor, f_ax)
        return {"sag": f_sag, "cor": f_cor, "ax": f_ax, "fused": fused}
