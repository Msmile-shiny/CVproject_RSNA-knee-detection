"""RSNA Knee MRI — 模型定义.

Phase 1: EfficientNetV2-S 2.5D baseline + Slice Attention + Head.
Phase 2: Tri-plane fusion, ConvNeXt/Swin/DenseNet ensemble (DenseNet added for multi-arch Teacher diversity).
Phase 3: ResNet3D-18 3D volumetric model.
Phase 4: Multi-model ensemble (EnsembleModel + EnsembleInference).

v2 Upgrade (Multi-Arch Teacher):
  - DINOv2-B (5ch 2.5D): global-attention ViT, LVD-142M pretrained, dim=768
  - ConvNeXtV2-Base (5ch 2.5D): GRN + FCMAE pretrained, dim=1024
  - Swin-Base (5ch 2.5D): hierarchical ViT, dim=1024
  - EfficientNetV2-S (5ch 2.5D): multi-scale conv, dim=1280 (保留)
"""

from .efficientnet25d import EfficientNetV2S25D
from .attention import SliceAttention
from .fusion import MultiPlaneFusion
from .head import ClassificationHead
from .triplane import TriPlaneModel
from .resnet3d import ResNet3DModel
from .convnext import ConvNeXt25D
from .swin import Swin25D, SwinBase25D
from .densenet import DenseNet25D
from .dinov2_25d import DinoV225D
from .convnextv2 import ConvNeXtV2Base25D
from .ensemble import EnsembleModel, EnsembleInference, ensemble_submissions
from .teacher import ImageTeacher, PerClassFusion, MultiArchImageTeacher
