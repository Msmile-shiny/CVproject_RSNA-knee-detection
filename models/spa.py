"""CNN Spatial Pattern Adapter (SPA).

A lightweight CNN that preserves fine-grained spatial details lost during
DINOv2's 14×14 patch tokenization. Produces multi-scale feature maps at
1/2, 1/4, and 1/8 resolutions of the input.

Design:
    - 3 stages of Conv → BN → GELU blocks
    - Each stage downsamples by 2× via stride-2 conv
    - Output: dict of feature maps at {1/2, 1/4, 1/8} resolution
    - Total trainable params: ~100K (intentionally small — DINOv2 handles semantics)

References:
    U-DFA (arXiv:2510.00585): Spatial Pattern Adapter design
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → GELU, with optional stride for downsampling."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SPAModule(nn.Module):
    """Multi-scale spatial pattern adapter.

    Input:  [B, 5, 384, 384]  — 5 adjacent MRI slices as channels
    Output: {
        "s2": [B, 64,  192, 192],   — 1/2 resolution
        "s4": [B, 128, 96,  96 ],   — 1/4 resolution
        "s8": [B, 256, 48,  48 ],   — 1/8 resolution
    }

    The output resolution "s8" (48×48) matches DINOv2's patch grid (28×28)
    closely enough for cross-attention, while "s2" and "s4" retain finer
    spatial detail for skip-connection style fusion.
    """

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 64,
    ):
        super().__init__()

        # Stage 0: initial projection (no downsampling)
        self.stem = ConvBlock(in_channels, base_channels, stride=1)

        # Stage 1: 384 → 192 (1/2)
        self.stage1 = nn.Sequential(
            ConvBlock(base_channels, base_channels, stride=1),
            ConvBlock(base_channels, base_channels, stride=2),  # downsample
        )

        # Stage 2: 192 → 96 (1/4)
        self.stage2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, stride=1),
            ConvBlock(base_channels * 2, base_channels * 2, stride=2),
        )

        # Stage 3: 96 → 48 (1/8)
        self.stage3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, stride=1),
            ConvBlock(base_channels * 4, base_channels * 4, stride=2),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-scale spatial features.

        Args:
            x: [B, 5, 384, 384] 5-channel input

        Returns:
            Dict with keys "s2", "s4", "s8" at decreasing resolutions.
        """
        x = self.stem(x)       # [B, 64,  384, 384]
        s2 = self.stage1(x)    # [B, 64,  192, 192]
        s4 = self.stage2(s2)   # [B, 128, 96,  96 ]
        s8 = self.stage3(s4)   # [B, 256, 48,  48 ]

        return {"s2": s2, "s4": s4, "s8": s8}
