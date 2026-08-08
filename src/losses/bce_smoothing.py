"""BCEWithLogitsLoss + Label Smoothing.

多标签分类的基线损失:
  - 12 个类别各自独立计算 binary cross-entropy
  - logits 不需要手动 sigmoid (BCEWithLogitsLoss 内部做了, 数值更稳定)
  - label smoothing 把硬标签 [0,1] 软化为 [ε, 1-ε], 减轻过拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLossSmooth(nn.Module):
    """BCEWithLogitsLoss + 可选的 label smoothing 和 pos_weight.

    Args:
        label_smoothing: 0 = 不平滑, 0.05 = 标签软化 5%
                         (标签从 [0,1] 变成 [0.025, 0.975])
        pos_weight: Tensor[12] 或 None.
                    用于处理类别不均衡 (mini 集 58:542)
                    公式: 正样本的 loss 乘以 pos_weight
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor([]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算 loss.

        Args:
            logits:  [B, 12] 模型原始输出 (logits, 未 sigmoid)
            targets: [B, 12] 二值标签 (0/1)

        Returns:
            scalar loss
        """
        if self.label_smoothing > 0:
            # 硬标签 [0,1] → 软标签 [smooth/2, 1-smooth/2]
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        pw = self.pos_weight if self.pos_weight.numel() > 0 else None

        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pw,
        )
