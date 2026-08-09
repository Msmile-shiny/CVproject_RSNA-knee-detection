"""多平面特征融合模块.

将 Sagittal / Coronal / Axial 三个平面的 series 特征融合为 study 特征.

Phase 1: 简单 Concat + Linear
Phase 2: TransformerEncoder 交叉注意力融合
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiPlaneFusion(nn.Module):
    """三平面特征融合.

    Args:
        feature_dim: 单平面特征维度
        fusion: "concat" (Phase 1) 或 "transformer" (Phase 2)
        num_heads: transformer 头数 (仅 fusion="transformer")
        num_layers: transformer 层数 (仅 fusion="transformer")
    """

    def __init__(
        self,
        feature_dim: int,
        fusion: str = "concat",
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.fusion_type = fusion
        self.feature_dim = feature_dim

        if fusion == "concat":
            self.proj = nn.Linear(feature_dim * 3, feature_dim)

        elif fusion == "transformer":
            self.plane_embed = nn.Parameter(torch.randn(3, feature_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

    def forward(
        self,
        f_sag: torch.Tensor,       # [B, D]   Sagittal 特征
        f_cor: torch.Tensor,       # [B, D]   Coronal 特征
        f_ax: torch.Tensor,        # [B, D]   Axial 特征
    ) -> torch.Tensor:
        """融合三平面特征 → study 特征 [B, D]."""
        if self.fusion_type == "concat":
            fused = torch.cat([f_sag, f_cor, f_ax], dim=-1)  # [B, 3D]
            return self.proj(fused)                            # [B, D]

        # Transformer fusion
        # 3 个 token (每个平面一个), 加上可学习的位置编码
        tokens = torch.stack([f_sag, f_cor, f_ax], dim=1)    # [B, 3, D]
        tokens = tokens + self.plane_embed.unsqueeze(0)       # [B, 3, D]
        tokens = self.transformer(tokens)                      # [B, 3, D]
        return tokens.mean(dim=1)                              # [B, D] mean pooling
