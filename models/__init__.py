"""RSNA Knee MRI — 模型定义.

Phase 1: EfficientNetV2-S 2.5D baseline + Slice Attention + Head.
Phase 2: Tri-plane fusion, ConvNeXt/Swin/DenseNet ensemble.
Phase 3: ResNet3D-18 3D volumetric model.
Phase 4: Multi-model ensemble (EnsembleModel + EnsembleInference).
"""

from .efficientnet25d import EfficientNetV2S25D
from .attention import SliceAttention
from .fusion import MultiPlaneFusion
from .head import ClassificationHead
from .triplane import TriPlaneModel
from .resnet3d import ResNet3DModel
from .convnext import ConvNeXt25D
from .swin import Swin25D
from .ensemble import EnsembleModel, EnsembleInference, ensemble_submissions
