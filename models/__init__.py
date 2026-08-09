"""RSNA Knee MRI — 模型定义.

Phase 1: EfficientNetV2-S 2.5D baseline + Slice Attention + Head.
Phase 2: Tri-plane fusion, ConvNeXt/Swin/DenseNet ensemble.
"""

from .efficientnet25d import EfficientNetV2S25D
from .attention import SliceAttention
from .fusion import MultiPlaneFusion
from .head import ClassificationHead
from .triplane import TriPlaneModel
