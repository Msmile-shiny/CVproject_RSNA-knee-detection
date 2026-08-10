"""Teacher-Student Distillation Loss.

基于置信度 (confidence) 和一致性 (agreement) 加权的 Soft BCE,
可选 KL divergence 项.

核心直觉:
  - NLP 和 Image Teacher 都确定且一致 → 高权重 (可靠样本)
  - NLP 和 Image Teacher 都不确定或分歧 → 低权重 (不可靠, 不强力拟合)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherDistillationLoss(nn.Module):
    """Confidence × Agreement 加权的蒸馏损失.

    L = Σ (confidence × agreement × SoftBCE) + λ × KL(Student, Teacher)

    其中:
      confidence = 2 × |teacher_prob - 0.5|           ∈ [0, 1]
      agreement  = 1 - |nlp_prob - image_prob|         ∈ [0, 1]
      SoftBCE    = BCEWithLogits(student_logits, teacher_prob)

    Args:
        use_kl: 是否附加 KL divergence 项
        kl_weight: KL 项权重 (默认 0.15)
        temperature: KL 温度 (默认 2.0)
        reduction: "mean" | "sum"
    """

    def __init__(
        self,
        use_kl: bool = True,
        kl_weight: float = 0.15,
        temperature: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.use_kl = bool(use_kl)
        self.kl_weight = float(kl_weight)
        self.temperature = float(temperature)
        self.reduction = str(reduction)

    def forward(
        self,
        student_logits: torch.Tensor,   # [B, 12]
        teacher_probs: torch.Tensor,    # [B, 12]  融合后 teacher 软标签
        nlp_probs: torch.Tensor,        # [B, 12]  NLP 校准概率
        image_probs: torch.Tensor,      # [B, 12]  Image Teacher OOF 概率
        sample_weights: torch.Tensor | None = None,  # [B, 12] 额外逐样本权重
    ) -> torch.Tensor:
        """计算蒸馏损失.

        Args:
            student_logits: Student 模型原始 logits
            teacher_probs: Teacher 融合软标签 (PerClassFusion.fuse 输出)
            nlp_probs: NLP 校准概率 (用于计算 agreement)
            image_probs: Image Teacher 概率 (sigmoid 后, 用于计算 agreement)
            sample_weights: 可选逐样本逐类权重 (如 NLP weight_*)

        Returns:
            scalar loss
        """
        # ── 1. Confidence weight ──────────────────────────────
        confidence = 2.0 * torch.abs(teacher_probs - 0.5)  # [B, 12]
        confidence = confidence.clamp(min=0.05)              # 保底, 避免完全忽略

        # ── 2. Agreement weight ──────────────────────────────
        agreement = 1.0 - torch.abs(nlp_probs - image_probs)  # [B, 12]
        agreement = agreement.clamp(min=0.1, max=1.0)

        # ── 3. Combined weight ───────────────────────────────
        combined_weight = confidence * agreement  # [B, 12]

        if sample_weights is not None:
            combined_weight = combined_weight * sample_weights.clamp(min=0.0)

        # ── 4. Soft BCE ──────────────────────────────────────
        soft_bce = F.binary_cross_entropy_with_logits(
            student_logits, teacher_probs, reduction="none"
        )  # [B, 12]

        weighted_bce = soft_bce * combined_weight

        if self.reduction == "mean":
            loss = weighted_bce.sum() / combined_weight.sum().clamp_min(1.0)
        elif self.reduction == "sum":
            loss = weighted_bce.sum()
        else:
            loss = weighted_bce  # [B, 12]

        # ── 5. Optional KL ───────────────────────────────────
        if self.use_kl:
            T = self.temperature
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_probs / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            loss = loss + self.kl_weight * kl_loss

        return loss


class CompositeDistillationLoss(nn.Module):
    """组合损失: Gold hard label + NLP soft label + Teacher distillation.

    用于 Student 训练, 同时接收多个监督信号::

        L = L_gold + L_nlp + λ_distill × L_distill

    其中:
      L_gold:    gold hard label + 高权重
      L_nlp:     NLP 校准软标签 + NLP weight_* 样本权重
      L_distill: Teacher 蒸馏损失 (Confidence × Agreement)
    """

    def __init__(
        self,
        distill_weight: float = 0.3,
        gold_weight: float = 3.0,
        nlp_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        use_kl: bool = True,
        kl_weight: float = 0.15,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.distill_weight = float(distill_weight)
        self.gold_weight = float(gold_weight)
        self.nlp_weight = float(nlp_weight)

        from losses.focal_bce import FocalBCELoss
        self.focal = FocalBCELoss(gamma=focal_gamma, alpha=focal_alpha, reduction="mean")
        self.distill = TeacherDistillationLoss(
            use_kl=use_kl, kl_weight=kl_weight, temperature=temperature, reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,           # [B, 12]
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """组合损失.

        Args:
            logits: Student 模型原始 logits
            targets: dict, 需包含以下键::

                "nlp_prob":       [B, 12]  NLP 校准软标签
                "nlp_weight":     [B, 12]  NLP 逐样本权重
                "teacher_prob":   [B, 12]  Teacher 融合软标签
                "nlp_prob_for_agreement":  [B, 12]  (同 nlp_prob)
                "image_prob":     [B, 12]  Image Teacher 概率
                "gold_mask":      [B] bool  True=gold 样本
                "gold_labels":    [B, 12]  gold hard labels

        Returns:
            scalar loss
        """
        gold_mask = targets["gold_mask"]  # [B] bool

        # ── L_gold: gold hard label loss (如果 batch 中有 gold) ──
        loss_gold = torch.tensor(0.0, device=logits.device)
        if gold_mask.any():
            gold_logits = logits[gold_mask]
            gold_labels = targets["gold_labels"][gold_mask]
            gold_w = torch.full_like(gold_labels, self.gold_weight)
            loss_gold = self.focal(gold_logits, gold_labels, gold_w)

        # ── L_nlp: NLP 软标签 loss (全部样本) ──────────────────
        loss_nlp = self.focal(
            logits,
            targets["nlp_prob"],
            targets["nlp_weight"].float() * self.nlp_weight,
        )

        # ── L_distill: Teacher 蒸馏 loss ──────────────────────
        loss_distill = self.distill(
            logits,
            targets["teacher_prob"],
            targets["nlp_prob_for_agreement"],
            targets["image_prob"],
            sample_weights=targets.get("nlp_weight"),
        )

        return loss_gold + loss_nlp + self.distill_weight * loss_distill
