"""DINOv2 + CNN Refiner — Full Model Architecture.

                    5 adjacent MRI slices [B, 5, 384, 384]
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
    ┌──────────────┐  ┌──────────┐  ┌──────────────┐
    │ DINOv2 ViT-S │  │ CNN SPA  │  │ DINOv2 ViT-S │ ... (×5, frozen)
    │  (frozen)    │  │(trainable)│  │  (frozen)    │
    │              │  │          │  │              │
    │ [CLS] token  │  │ {s2,s4,  │  │ [CLS] token  │
    │ [384]        │  │   s8}    │  │ [384]        │
    └──────┬───────┘  └────┬─────┘  └──────┬───────┘
           │               │               │
           │      ┌────────┘               │
           │      │ CrossModalFusion       │
           │      │ (×5, one per slice)    │
           │      │                        │
           ▼      ▼                        ▼
       [CLS'₁]  [CLS'₂]  [CLS'₃]  [CLS'₄]  [CLS'₅]
           │       │        │        │        │
           └───────┴────────┴────────┴────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   SliceTransformer     │  (trainable)
              │   Self-attention over  │
              │   5 slice tokens       │
              └───────────┬────────────┘
                          │
                          ▼  [B, 384]
              ┌────────────────────────┐
              │  ClassificationHead    │  (trainable)
              │  384→512→12 logits     │
              └───────────┬────────────┘
                          │
                          ▼
                   12-class logits

Total params: ~22M (DINOv2 frozen) + ~1M (trainable)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .spa import SPAModule
from .lgfa import CrossModalFusion
from .slice_transformer import SliceTransformer
from .head import ClassificationHead


class DINOv2Refiner(nn.Module):
    """DINOv2 feature extractor + CNN Refiner for 2.5D knee MRI classification.

    DINOv2 processes each of the 5 adjacent slices independently (frozen).
    A CNN SPA preserves fine-grained spatial detail. Cross-modal fusion
    combines them per slice. A SliceTransformer models inter-slice
    relationships. Finally, a classification head outputs 12-class logits.

    Args:
        dinov2_model: Pre-loaded timm DINOv2 ViT-S/14 model (will be frozen)
        spa_channels: Base channel count for CNN SPA
        cls_dim: DINOv2 feature dimension (384 for ViT-S)
        num_slices: Number of adjacent slices (default 5)
        num_classes: Number of output classes (default 12)
        num_heads: Attention heads for cross-modal fusion + slice transformer
        slice_transformer_layers: Number of transformer layers in SliceTransformer
        dropout: Dropout rate
        freeze_dinov2: Whether to freeze DINOv2 (default True)
    """

    def __init__(
        self,
        dinov2_model: nn.Module,
        spa_channels: int = 64,
        cls_dim: int = 384,
        num_slices: int = 5,
        num_classes: int = 12,
        num_heads: int = 4,
        slice_transformer_layers: int = 2,
        dropout: float = 0.1,
        freeze_dinov2: bool = True,
    ):
        super().__init__()

        self.num_slices = num_slices
        self.cls_dim = cls_dim

        # ── DINOv2 backbone (frozen feature extractor) ────────
        self.dinov2 = dinov2_model
        if freeze_dinov2:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()

        # ── CNN Spatial Pattern Adapter ────────────────────────
        self.spa = SPAModule(in_channels=num_slices, base_channels=spa_channels)
        # SPA s8 output: [B, 256, 48, 48] (256 = spa_channels * 4)

        # ── Cross-modal fusion (one per slice) ─────────────────
        self.fusion = CrossModalFusion(
            cls_dim=cls_dim,
            cnn_dim=spa_channels * 4,  # 256
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── Slice Transformer ──────────────────────────────────
        self.slice_transformer = SliceTransformer(
            dim=cls_dim,
            num_heads=num_heads,
            num_layers=slice_transformer_layers,
            dropout=dropout,
        )

        # ── Classification Head ────────────────────────────────
        self.head = ClassificationHead(
            in_features=cls_dim,
            hidden_features=512,
            num_classes=num_classes,
            dropout=0.3,
        )

        # ── Count parameters ───────────────────────────────────
        self._log_param_counts()

    def _log_param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"[DINOv2Refiner] Total: {total:,} | "
              f"Trainable: {trainable:,} | Frozen: {frozen:,} "
              f"({frozen/total*100:.0f}% frozen)")

    def _extract_dinov2_features(
        self, slices_3ch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract [CLS] token from a batch of single slices.

        Args:
            slices_3ch: [B_slices, 3, H, W] — grayscale MRI repeated to 3 channels

        Returns:
            cls_token: [B_slices, D] — [CLS] token for each slice
        """
        # DINOv2 forward_features returns [CLS] + patch tokens
        # timm ViT: forward_features → [B, 785, 384] (1 CLS + 784 patches)
        with torch.no_grad():
            features = self.dinov2.forward_features(slices_3ch)

        # [CLS] token is the first token
        cls_token = features[:, 0, :]  # [B_slices, 384]
        return cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 5, 384, 384] — 5 adjacent grayscale MRI slices as 5 channels

        Returns:
            logits: [B, 12] — raw logits (no sigmoid)
        """
        B = x.shape[0]

        # ── 1. CNN SPA: spatial features from the 5-channel stack ─
        spa_features = self.spa(x)  # {"s2": [B,64,192,192], "s4": [B,128,96,96], "s8": [B,256,48,48]}

        # ── 2. DINOv2: per-slice semantic features ──────────────
        # Each slice is 1-channel grayscale → repeat to 3-channel "RGB"
        # x: [B, 5, 384, 384] → process each slice independently
        slice_cls_list: list[torch.Tensor] = []

        for i in range(self.num_slices):
            # Extract slice i: [B, 1, 384, 384] → [B, 3, 384, 384]
            slice_i = x[:, i:i+1, :, :]           # [B, 1, 384, 384]
            slice_3ch = slice_i.expand(-1, 3, -1, -1)  # [B, 3, 384, 384]

            cls_i = self._extract_dinov2_features(slice_3ch)  # [B, 384]
            slice_cls_list.append(cls_i)

        # ── 3. Cross-modal fusion per slice ────────────────────
        fused_cls_list: list[torch.Tensor] = []

        for cls_i in slice_cls_list:
            # Fuse [CLS] token with SPA spatial features
            enhanced = self.fusion(cls_i, spa_features["s8"])  # [B, 384]
            fused_cls_list.append(enhanced)

        # ── 4. Slice Transformer ───────────────────────────────
        # Stack: [B, 5, 384]
        slice_tokens = torch.stack(fused_cls_list, dim=1)
        study_feature = self.slice_transformer(slice_tokens)  # [B, 384]

        # ── 5. Classification Head ─────────────────────────────
        logits = self.head(study_feature)  # [B, 12]

        return logits

    def train(self, mode: bool = True):
        """Override train() to keep DINOv2 in eval mode."""
        super().train(mode)
        # Always keep DINOv2 in eval mode
        self.dinov2.eval()
        return self
