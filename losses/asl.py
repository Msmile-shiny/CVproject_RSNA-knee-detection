"""Asymmetric Loss (ASL) for multi-label classification.

专为多标签场景设计, 解决正负样本严重不平衡问题。

核心机制:
  1. Asymmetric focusing: γ- > γ+ (负样本聚焦更强)
  2. Probability shifting (m): p < m 的负样本完全忽略 (硬阈值)
  3. Per-class gamma: 罕见类用更小的 γ+ (珍惜正样本信号)

公式:
  L+ = (1-p)^(γ+)  * log(p)          正样本
  L- = (p_m)^(γ-)   * log(1-p_m)     负样本, p_m = max(p - m, 0)

Reference:
  Ridnik et al., "Asymmetric Loss for Multi-Label Classification", ICCV 2021.
  https://arxiv.org/abs/2009.14119
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    对比 FocalBCE:
      - FL: -(1-p_t)^γ * log(p_t), 正负样本同一 γ, 无概率偏移
      - ASL: γ+/γ- 独立, m 硬阈值忽略简单负样本

    参数选择指南 (膝关节 12 标签):
      - 常见类 (Effusion ~25%):  gamma_pos=1, gamma_neg=4
      - 中等类 (ACL ~15%):       gamma_pos=1, gamma_neg=3
      - 罕见类 (Fracture ~2%):   gamma_pos=0, gamma_neg=4  (珍惜正样本)
      - 通用默认:                gamma_pos=1, gamma_neg=4, clip=0.05

    Args:
        gamma_pos: 正样本聚焦参数, 标量或 per-class 列表.
                   0 = 不做聚焦 (罕见类), 1 = 标准聚焦 (常见类).
        gamma_neg: 负样本聚焦参数, 标量或 per-class 列表.
                   通常 > gamma_pos, 因为负样本占绝对多数.
        clip: 概率偏移 m ∈ [0, 1).
              预测概率 < m 的负样本完全忽略 (不贡献梯度).
              默认 0.05 适合大多数场景.
        reduction: "mean" | "sum" | "none"
        eps: 数值稳定
    """

    def __init__(
        self,
        gamma_pos: float | list[float] = 1.0,
        gamma_neg: float | list[float] = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = float(clip)
        self.reduction = str(reduction)
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,    # [B, C] raw logits
        targets: torch.Tensor,   # [B, C] binary labels (0/1)
        weights: torch.Tensor | None = None,  # [B, C] 逐样本逐类置信度权重
    ) -> torch.Tensor:
        """计算 ASL.

        Args:
            logits: 模型原始 logits (未经过 sigmoid)
            targets: 二值标签 (0 or 1), 支持软标签 (但 ASL 设计用于硬标签)
            weights: 可选逐样本权重 (如 confidence). None = 全 1.

        Returns:
            scalar loss (reduction="mean")
        """
        probs = torch.sigmoid(logits)                    # [B, C]
        n_classes = probs.shape[1]

        # ── 解析 per-class gamma ──────────────────────────────
        gp = self._to_tensor(self.gamma_pos, n_classes, logits)
        gn = self._to_tensor(self.gamma_neg, n_classes, logits)

        # ── 正样本 loss: (1-p)^(γ+) * log(p) ─────────────────
        pos_term = (1.0 - probs).clamp_min(self.eps).pow(gp)  # [B, C]
        pos_loss = -pos_term * torch.log(probs.clamp_min(self.eps))

        # ── 负样本 loss: p_m^(γ-) * log(1-p_m) ───────────────
        # Probability shifting: p_m = max(p - m, 0)
        p_m = (probs - self.clip).clamp_min(0.0)               # [B, C]
        neg_term = p_m.clamp_min(self.eps).pow(gn)
        neg_loss = -neg_term * torch.log((1.0 - probs).clamp_min(self.eps))

        # ── 合并 ──────────────────────────────────────────────
        # targets * pos_loss + (1-targets) * neg_loss
        # 注意: 当 p_m=0 时 neg_loss 自动为 0 (硬阈值生效)
        loss = targets * pos_loss + (1.0 - targets) * neg_loss  # [B, C]

        # ── 权重 ──────────────────────────────────────────────
        if weights is not None:
            weights = weights.to(device=loss.device, dtype=loss.dtype)
            if weights.dim() == 1:
                weights = weights.unsqueeze(1).expand_as(loss)
            loss = loss * weights.clamp_min(0.0)

        # ── Reduction ─────────────────────────────────────────
        if self.reduction == "mean":
            if weights is not None:
                return loss.sum() / weights.sum().clamp_min(1.0)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # [B, C]

    def _to_tensor(
        self,
        value: float | list[float],
        n_classes: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """将标量或列表转为 [1, C] tensor."""
        if isinstance(value, (int, float)):
            return torch.full((1, n_classes), float(value), device=ref.device)
        if len(value) != n_classes:
            raise ValueError(
                f"Per-class gamma length {len(value)} != n_classes {n_classes}"
            )
        return torch.tensor(value, device=ref.device, dtype=torch.float32).unsqueeze(0)

    def _validate_gamma(self, value: float | list[float], label: str) -> np.ndarray:
        """验证并返回 numpy array."""
        if isinstance(value, (int, float)):
            return np.full(12, float(value), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (12,):
            raise ValueError(f"{label} 必须是标量或长度为 12 的列表")
        return arr


def compute_asl_class_gamma(
    labels_df,
    target_columns: list[str],
    prevalence_threshold_rare: float = 0.05,
    prevalence_threshold_medium: float = 0.10,
    gamma_pos_common: float = 1.0,
    gamma_pos_medium: float = 0.5,
    gamma_pos_rare: float = 0.0,
    gamma_neg_common: float = 4.0,
    gamma_neg_medium: float = 3.0,
    gamma_neg_rare: float = 4.0,
    clip: float = 0.05,
    smoothing: float = 0.5,
) -> dict:
    """基于训练数据的类别 prevalence 计算 per-class ASL 参数.

    逻辑:
      - prevalence < 5%  → 罕见类: γ+ = 0 (不做聚焦), γ- = 4
      - prevalence 5-10% → 中等类: γ+ = 0.5, γ- = 3
      - prevalence > 10% → 常见类: γ+ = 1, γ- = 4

    Args:
        labels_df: StudyInstanceUID-indexed 标签 DataFrame
        target_columns: 12 个目标列名
        prevalence_threshold_rare: 罕见类阈值 (default 0.05)
        prevalence_threshold_medium: 中等类阈值 (default 0.10)
        gamma_pos_* : 各类别 γ+ 值
        gamma_neg_* : 各类别 γ- 值
        clip: ASL 概率偏移
        smoothing: Laplace smoothing factor (default 0.5)

    Returns:
        {class_name: {"gamma_pos": float, "gamma_neg": float, "prevalence": float}}
    """
    n = len(labels_df)
    if n == 0:
        # Fallback: 全部用默认值
        return {
            col: {"gamma_pos": gamma_pos_common, "gamma_neg": gamma_neg_common,
                  "prevalence": 0.1, "clip": clip}
            for col in target_columns
        }

    result = {}
    for col in target_columns:
        if col not in labels_df.columns:
            result[col] = {"gamma_pos": gamma_pos_common, "gamma_neg": gamma_neg_common,
                           "prevalence": 0.1, "clip": clip}
            continue

        # 计算 prevalence (with Laplace smoothing)
        pos_count = labels_df[col].astype(float).sum() + smoothing
        prevalence = float(pos_count / (n + 2 * smoothing))

        # 依据 prevalence 分类
        if prevalence < prevalence_threshold_rare:
            gp, gn = gamma_pos_rare, gamma_neg_rare
            tier = "rare"
        elif prevalence < prevalence_threshold_medium:
            gp, gn = gamma_pos_medium, gamma_neg_medium
            tier = "medium"
        else:
            gp, gn = gamma_pos_common, gamma_neg_common
            tier = "common"

        result[col] = {
            "gamma_pos": gp,
            "gamma_neg": gn,
            "prevalence": prevalence,
            "clip": clip,
            "tier": tier,
        }

    return result


def compute_confidence(
    teacher_probs: "np.ndarray",   # [N, C]
    method: str = "distance",
) -> "np.ndarray":
    """计算 Teacher 软标签的可信度.

    Args:
        teacher_probs: [N, C] Teacher 融合概率 ∈ [0, 1]
        method: "distance" — 距 0.5 的距离 (越远越确定)
                "entropy"  — 1 - normalized entropy

    Returns:
        confidence: [N, C] float32 ∈ [clamp_min, 1]
    """
    import numpy as np

    if method == "distance":
        confidence = 2.0 * np.abs(teacher_probs - 0.5)
    elif method == "entropy":
        eps = 1e-8
        p = np.clip(teacher_probs, eps, 1.0 - eps)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        max_entropy = -np.log(0.5)  # ≈ 0.693
        confidence = 1.0 - entropy / max_entropy
    else:
        raise ValueError(f"Unknown confidence method: {method}")

    # 保底最小值, 避免完全忽略某些样本
    confidence = np.clip(confidence, 0.05, 1.0)
    return confidence.astype(np.float32)
