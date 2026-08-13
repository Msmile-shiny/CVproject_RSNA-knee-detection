"""损失函数模块."""

from .focal_bce import FocalBCELoss
from .asl import AsymmetricLoss, compute_asl_class_gamma, compute_confidence
from .distillation import (
    TeacherDistillationLoss,
    CompositeDistillationLoss,
    ConfidenceWeightedDistillationLoss,
)
