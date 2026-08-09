"""多模型集成模块 — Phase 4 最终提交方案.

组件:
  1. EnsembleModel — 多 backbone 联合训练 (learnable fusion)
  2. EnsembleInference — 加载多个 checkpoint, 加权平均预测

集成策略:
  - Weighted Mean: 每个模型一个可学习标量权重 → softmax → 加权平均概率
  - Concat Fusion: 拼接各模型特征 → 融合层 → 分类头
  - Simple Average (推理默认): 等权平均 sigmoid 概率 (最鲁棒)

使用:
  from models import EnsembleModel, EnsembleInference

  # 方式 A: 联合训练 (需同时加载 N 个 backbone)
  model = EnsembleModel(backbones=[effnet, convnext, swin])
  model(images)  # → [B, 12]

  # 方式 B: 后处理集成 (推荐 — 各模型独立训练, 推理时合并)
  inference = EnsembleInference(
      checkpoint_paths=["effnet.pt", "convnext.pt", "swin.pt"],
      weights=None,            # None=等权, 或传入 list[float]
  )
  submission = inference.run(test_loader)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import ClassificationHead

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# EnsembleModel — 多 backbone 联合训练
# ═══════════════════════════════════════════════════════════════════


class EnsembleModel(nn.Module):
    """多 backbone 集成模型.

    将 N 个预训练 backbone 的输出做可学习融合.

    Args:
        backbones: nn.Module 列表, 每个都有 extract_features(x) → [B, D_i]
        feature_dims: 各 backbone 的特征维度 (None → 自动探测)
        num_classes: 输出类别数
        fusion: "weighted_mean" | "concat"
        learnable_weights: True → 学习 per-model 权重 (softmax over logits)
        dropout: 分类头 dropout
    """

    def __init__(
        self,
        backbones: list[nn.Module],
        feature_dims: list[int] | None = None,
        num_classes: int = 12,
        fusion: str = "weighted_mean",
        learnable_weights: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_models = len(backbones)
        self.fusion = fusion

        # 注册 backbones 为 ModuleList
        self.backbones = nn.ModuleList(backbones)

        # ── 探测特征维度 ───────────────────────────────────────
        if feature_dims is None:
            feature_dims = []
            for b in backbones:
                if hasattr(b, "feature_dim"):
                    feature_dims.append(b.feature_dim)
                elif hasattr(b.backbone, "num_features"):
                    feature_dims.append(b.backbone.num_features)
                else:
                    feature_dims.append(512)  # fallback

        self.feature_dims = feature_dims
        logger.info(
            "Ensemble: %d models, dims=%s, fusion=%s",
            self.n_models, feature_dims, fusion,
        )

        # ── 可学习权重 (weighted_mean 模式) ─────────────────────
        self.learnable_weights = learnable_weights
        if fusion == "weighted_mean" and learnable_weights:
            # 每个模型一个标量 log-weight → softmax
            self.model_weights = nn.Parameter(torch.zeros(self.n_models))

        # ── Concat 融合层 ───────────────────────────────────────
        if fusion == "concat":
            total_dim = sum(feature_dims)
            hidden_dim = max(feature_dims)  # 用最大 feature_dim 做 hidden
            self.fusion_proj = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.head = ClassificationHead(
                in_features=hidden_dim,
                hidden_features=hidden_dim // 2,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif fusion == "weighted_mean":
            # 每个 backbone 用自己的 head
            # 如果 backbone 已有 head, 就用它; 否则创建新的
            self.heads = nn.ModuleList()
            for i, b in enumerate(backbones):
                if hasattr(b, "head") and not isinstance(b.head, nn.Identity):
                    self.heads.append(b.head)
                else:
                    self.heads.append(
                        ClassificationHead(
                            in_features=feature_dims[i],
                            hidden_features=feature_dims[i] // 2,
                            num_classes=num_classes,
                            dropout=dropout,
                        )
                    )
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """单个输入通过所有 backbone, 融合输出.

        Args:
            x: [B, 5, H, W] (所有 backbone 共享同一输入)

        Returns:
            logits: [B, num_classes]
        """
        if self.fusion == "weighted_mean":
            # 各 backbone 独立预测 → 加权平均
            logits_list = []
            for b, head in zip(self.backbones, self.heads):
                feat = b.extract_features(x)    # [B, D_i]
                logits_list.append(head(feat))  # [B, 12]

            stacked = torch.stack(logits_list, dim=-1)  # [B, 12, N]

            if self.learnable_weights:
                w = F.softmax(self.model_weights, dim=0)  # [N]
                w = w.view(1, 1, -1)                       # [1, 1, N]
                return (stacked * w).sum(dim=-1)            # [B, 12]
            else:
                return stacked.mean(dim=-1)

        elif self.fusion == "concat":
            # 拼接特征 → 融合层 → Head
            feats = [b.extract_features(x) for b in self.backbones]
            fused = self.fusion_proj(torch.cat(feats, dim=-1))  # [B, D_max]
            return self.head(fused)


# ═══════════════════════════════════════════════════════════════════
# EnsembleInference — 后处理集成 (独立训练 + 推理时合并)
# ═══════════════════════════════════════════════════════════════════


class EnsembleInference:
    """加载多个独立训练的 checkpoint, 在推理时合并预测.

    这是最实用的集成方式:
    - 各模型独立训练 (避免联合训练的复杂性和 VRAM 压力)
    - 推理时加载所有 checkpoint → 加权平均概率

    使用:
        ensemble = EnsembleInference(
            checkpoint_paths=["effnet.pt", "convnext.pt", "swin.pt"],
            model_classes=[EfficientNetV2S25D, ConvNeXt25D, Swin25D],
            model_kwargs=[{}, {}, {}],
            weights=[0.4, 0.3, 0.3],       # 基于 val AUC 的权重
        )
        probs = ensemble.predict(images)     # [B, 12] probabilities

    Args:
        checkpoint_paths: 各模型 .pt 文件路径
        model_classes: 各模型对应的 class
        model_kwargs: 各模型构造函数 kwargs (不含 pretrained)
        weights: 各模型权重 (None → 等权, or list[float])
        device: 推理设备
    """

    def __init__(
        self,
        checkpoint_paths: list[str | Path],
        model_classes: list[type],
        model_kwargs: list[dict] | None = None,
        weights: list[float] | None = None,
        device: str = "cuda",
    ):
        self.device = device
        self.n_models = len(checkpoint_paths)

        if model_kwargs is None:
            model_kwargs = [{}] * self.n_models

        if weights is None:
            self.weights = [1.0 / self.n_models] * self.n_models
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

        logger.info(
            "EnsembleInference: %d models, weights=%s",
            self.n_models,
            [f"{w:.3f}" for w in self.weights],
        )

        # ── 加载所有模型 ───────────────────────────────────────
        self.models = []
        for i in range(self.n_models):
            ckpt_path = Path(checkpoint_paths[i])
            cls = model_classes[i]
            kwargs = model_kwargs[i].copy()
            kwargs["pretrained"] = False  # 推理不需要 pretrained

            model = cls(**kwargs).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

            # 处理 checkpoint key (可能带 "model." 前缀)
            state_dict = ckpt.get("model", ckpt)
            model.load_state_dict(state_dict)
            model.eval()

            self.models.append(model)

            val_auc = ckpt.get("auc", ckpt.get("val_auc", 0))
            logger.info(
                "  模型 %d: %s  (val_auc=%.4f, weight=%.3f)",
                i, ckpt_path.name, val_auc, self.weights[i],
            )

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """单平面图像推理: 加权平均各模型的 sigmoid 概率.

        Args:
            images: [B, 5, H, W] tensor

        Returns:
            probs: [B, 12] 加权平均概率
        """
        images = images.to(self.device)
        probs_sum = None

        for model, w in zip(self.models, self.weights):
            logits = model(images)                     # [B, 12]
            probs = torch.sigmoid(logits)              # [B, 12]

            if probs_sum is None:
                probs_sum = probs * w
            else:
                probs_sum += probs * w

        return probs_sum

    @torch.no_grad()
    def predict_triplane(
        self,
        x_sag: torch.Tensor,
        x_cor: torch.Tensor,
        x_ax: torch.Tensor,
    ) -> torch.Tensor:
        """Tri-plane 推理: 加权平均各 tri-plane 模型的 sigmoid 概率.

        Args:
            x_sag, x_cor, x_ax: [B, 5, H, W] tensors

        Returns:
            probs: [B, 12]
        """
        x_sag = x_sag.to(self.device)
        x_cor = x_cor.to(self.device)
        x_ax = x_ax.to(self.device)
        probs_sum = None

        for model, w in zip(self.models, self.weights):
            try:
                logits = model(x_sag, x_cor, x_ax)     # TriPlaneModel
            except TypeError:
                # fallback: single-plane model, use Sagittal only
                logits = model(x_sag)
            probs = torch.sigmoid(logits)

            if probs_sum is None:
                probs_sum = probs * w
            else:
                probs_sum += probs * w

        return probs_sum


# ═══════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════


def ensemble_submissions(
    submission_paths: list[str | Path],
    weights: list[float] | None = None,
    output_path: str | Path | None = None,
) -> "pd.DataFrame":
    """合并多个 submission CSV (后处理集成).

    适用于: 各模型独立训练+推理, 已生成各自的 submission CSV.

    Args:
        submission_paths: submission CSV 文件列表
        weights: 权重 (None → 等权)
        output_path: 输出 CSV 路径 (None → 不保存)

    Returns:
        加权平均后的 submission DataFrame
    """
    import pandas as pd

    dfs = [pd.read_csv(p, index_col=0) for p in submission_paths]
    n = len(dfs)

    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    # 确保所有 submission 的 index 一致
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)

    avg = sum(w * df.loc[common_index] for w, df in zip(weights, dfs))

    if output_path:
        avg.to_csv(output_path)
        logger.info("Ensemble submission 已保存: %s (%d studies)", output_path, len(avg))

    return avg
