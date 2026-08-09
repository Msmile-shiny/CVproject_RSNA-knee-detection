"""Slice Transformer.

A lightweight transformer that models inter-slice relationships across
the 5 adjacent MRI slices. Each slice is represented by a fused feature
vector (DINOv2 [CLS] + CNN spatial via CrossModalFusion). Self-attention
over these 5 tokens learns which slices carry the strongest signal for
each abnormality — then aggregates them into a single study-level feature.

Design:
    - 5 slice tokens + learnable positional encoding
    - 2 layers of Multi-Head Self-Attention + FFN
    - Mean pooling of output tokens → study feature

References:
    MST (Sci Rep 2025, doi:10.1038/s41598-025-09041-8):
        Slice Transformer for 3D medical image classification
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SliceTransformer(nn.Module):
    """Self-attention over 5 adjacent MRI slices.

    Args:
        dim: Feature dimension per slice (384 for ViT-S [CLS])
        num_heads: Attention heads
        num_layers: Transformer layers (2 is sufficient for 5 tokens)
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_slices = 5

        # Learnable positional encoding for each slice position
        # Slices represent z-2, z-1, z, z+1, z+2 relative to center
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_slices, dim) * 0.02)

        # Transformer encoder over slices
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,    # 384 → 1536
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,            # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: mean-pooled tokens → final feature
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        slice_features: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Aggregate 5 slice features into a single study feature.

        Args:
            slice_features: [B, 5, D] — one feature vector per slice
            return_attention: if True, also return attention weights

        Returns:
            study_feature: [B, D] aggregated feature, or
            (study_feature, attention_weights) if return_attention=True
        """
        B = slice_features.shape[0]

        # Add positional encoding
        tokens = slice_features + self.pos_embed  # [B, 5, D]

        # Self-attention over slices
        tokens = self.transformer(tokens)  # [B, 5, D]

        # Mean pooling → study feature
        # (we could also use [CLS]-style first-token pooling, but mean is
        #  more robust for 5 nearly-symmetric positions)
        study_feature = tokens.mean(dim=1)  # [B, D]
        study_feature = self.norm(study_feature)

        if return_attention:
            # Extract attention weights from the last layer for interpretability
            # This shows which slices the model focuses on
            last_attn = None
            # NOTE: nn.TransformerEncoder doesn't expose attention weights directly.
            # For interpretability, use a custom implementation or hooks.
            return study_feature, last_attn

        return study_feature
