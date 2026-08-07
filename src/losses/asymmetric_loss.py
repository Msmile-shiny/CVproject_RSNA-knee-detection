"""Asymmetric Loss (ASL) — 多标签分类主力损失.

参考: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)

关键参数:
- γ_neg=4: 强力压制简单负样本, 聚焦难负样本
- γ_pos=1: 轻度聚焦正样本 (正样本本身就少, 不需要强力压制)
- clip=0.05: 防止梯度被极端概率值主导
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Args:
        gamma_neg: 负样本聚焦参数 (默认 4)
        gamma_pos: 正样本聚焦参数 (默认 1)
        clip: 概率裁剪阈值 (默认 0.05)
        disable_torch_grad_focal_loss: 是否用 PyTorch 内置 focal 实现
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        disable_torch_grad_focal_loss: bool = False,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 计算概率
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.clip, 1.0 - self.clip)

        # 正样本项
        pos_term = targets * torch.pow(1 - probs, self.gamma_pos) * torch.log(probs)

        # 负样本项 — γ_neg 通常较大 (4), 强力抑制简单负样本
        neg_term = (1 - targets) * torch.pow(probs, self.gamma_neg) * torch.log(1 - probs)

        loss = -(pos_term + neg_term)
        return loss.mean()
