"""图像模型定义.

当前主力:
- KneeClassifier2D: 2D 单切片分类器 (ConvNeXtV2-Tiny + 12-class head)

后续:
- TriplaneModel: 2.5D 三平面融合 (需多平面数据)
- ResNet3D-18: 轻量 3D (阶段三)
"""

from .classifier import KneeClassifier2D
from .backbone import create_backbone, get_feature_dim
from .triplane import TriplaneModel       # 保留, 后续使用
