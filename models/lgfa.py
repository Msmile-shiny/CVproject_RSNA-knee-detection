"""Cross-Modal Fusion (LGFA-style).

Fuses DINOv2 semantic features with CNN spatial features via cross-attention.
The DINOv2 [CLS] token "queries" the CNN spatial feature map, letting the
model attend to fine-grained spatial locations that 14×14 patch tokenization
may have averaged away.

Design:
    - Q = DINOv2 [CLS] token (semantic: "what am I looking at?")
    - K,V = CNN spatial features  (spatial:  "where exactly is the signal?")
    - Cross-attention output + original [CLS] → LayerNorm → enhanced feature

References:
    U-DFA (arXiv:2510.00585): Local-Global Fusion Adapter
    MST (Sci Rep 2025): cross-modal attention for medical imaging
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """Cross-attention fusion between a [CLS] token and CNN spatial features.

    Args:
        cls_dim: DINOv2 [CLS] token dimension (384 for ViT-S)
        cnn_dim: CNN feature channel dimension (256 from SPA s8)
        num_heads: attention heads (default 4)
        dropout: attention dropout
    """

    def __init__(
        self,
        cls_dim: int = 384,
        cnn_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = cls_dim // num_heads
        assert cls_dim % num_heads == 0, f"cls_dim ({cls_dim}) must be divisible by num_heads ({num_heads})"

        # Project CNN features to same dimension as [CLS]
        self.cnn_proj = nn.Linear(cnn_dim, cls_dim)

        # Cross-attention: Q from [CLS], K,V from CNN spatial features
        self.q_proj = nn.Linear(cls_dim, cls_dim)    # [CLS] → Q
        self.k_proj = nn.Linear(cls_dim, cls_dim)    # CNN → K
        self.v_proj = nn.Linear(cls_dim, cls_dim)    # CNN → V

        self.out_proj = nn.Linear(cls_dim, cls_dim)
        self.dropout = nn.Dropout(dropout)

        # Final fusion: original [CLS] + attention output
        self.norm = nn.LayerNorm(cls_dim)
        self.gate = nn.Parameter(torch.zeros(1))  # Learnable gate: start near 0

    def forward(
        self,
        cls_token: torch.Tensor,
        cnn_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse [CLS] token with CNN spatial features.

        Args:
            cls_token:     [B, D]      DINOv2 [CLS] token (D=384)
            cnn_features:  [B, C, H, W] CNN spatial feature map (C=256, H=W=48)

        Returns:
            Enhanced feature [B, D] combining semantics + spatial detail.
        """
        B, C, H, W = cnn_features.shape
        D = cls_token.shape[-1]

        # ── Reshape CNN features to sequence ──────────────────
        # [B, C, H, W] → [B, H*W, C]
        cnn_seq = cnn_features.flatten(2).transpose(1, 2)  # [B, 2304, 256]
        cnn_seq = self.cnn_proj(cnn_seq)                    # [B, 2304, 384]

        # ── Cross-attention: [CLS] queries CNN ────────────────
        # Q: single [CLS] token — "what spatial details matter?"
        # K,V: CNN spatial grid — "here's what's at each location"
        q = self.q_proj(cls_token)        # [B, D]
        k = self.k_proj(cnn_seq)          # [B, N, D]
        v = self.v_proj(cnn_seq)          # [B, N, D]

        # Multi-head reshape
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, 1, d]
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale     # [B, H, 1, N]
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ v                         # [B, H, 1, d]
        attn_output = attn_output.transpose(1, 2).contiguous() # [B, 1, H, d]
        attn_output = attn_output.view(B, D)                   # [B, D]
        attn_output = self.out_proj(attn_output)

        # ── Gated fusion with original [CLS] ──────────────────
        gate = self.gate.tanh()  # [-1, 1], starts near 0 → initially passes [CLS] mostly
        enhanced = self.norm(cls_token + gate * attn_output)

        return enhanced  # [B, D]
