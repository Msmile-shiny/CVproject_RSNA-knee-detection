"""DINOv2 + CNN Refiner — RSNA 2026 Knee Abnormality Detection.

Components:
    SPAModule            — CNN Spatial Pattern Adapter (multi-scale spatial features)
    CrossModalFusion     — Cross-attention between DINOv2 tokens and CNN features
    SliceTransformer     — Multi-head self-attention over 5 adjacent slices
    ClassificationHead   — 2-layer MLP head → 12-class logits
    DINOv2Refiner        — Full model: DINOv2 (frozen) + SPA + Fusion + SliceTransformer + Head
"""

from .spa import SPAModule
from .lgfa import CrossModalFusion
from .slice_transformer import SliceTransformer
from .head import ClassificationHead
from .dinov2_refiner import DINOv2Refiner

__all__ = [
    "SPAModule",
    "CrossModalFusion",
    "SliceTransformer",
    "ClassificationHead",
    "DINOv2Refiner",
]
