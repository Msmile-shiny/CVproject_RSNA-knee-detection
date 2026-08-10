"""Image Teacher & Per-Class Calibrated Fusion.

ImageTeacher:  5-fold CV 训练图像模型, 对全量训练数据生成 OOF logits.
PerClassFusion: 基于 58 gold 验证集, 逐类计算 NLP 和 Image Teacher
                的可信度权重, 生成融合后的 Teacher 软标签.

         ┌──────────────────────┐
         │  NLP prob (校准后)    │──┐
         └──────────────────────┘  │    ┌──────────────┐
                                   ├───→│ PerClass     │
         ┌──────────────────────┐  │    │ Fusion       │──→ Teacher prob
         │  Image OOF logits    │──┘    └──────────────┘
         └──────────────────────┘
"""

from __future__ import annotations

import gc
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

TARGET_COLUMNS = [
    "ACL", "MCL",
    "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


# ═══════════════════════════════════════════════════════════════════
# ImageTeacher
# ═══════════════════════════════════════════════════════════════════

class ImageTeacher:
    """5-Fold CV 图像教师模型.

    在全部训练数据上做 K-fold 交叉验证, 为每条数据生成无偏的
    OOF (out-of-fold) 预测 logits. 这些 logits 反映图像模型视角,
    与 NLP 文本视角互补.

    使用方式::

        teacher = ImageTeacher(
            model_factory=lambda: EfficientNetV2S25D(...),
            device=DEVICE,
        )
        oof_logits, fold_models = teacher.fit(
            train_dataset=train_ds,          # 含 pseudo + gold
            gold_mask=gold_study_mask,        # bool [N]
            gold_labels=gold_label_array,     # [N_gold, 12]
            n_folds=5, epochs=30, batch_size=8,
        )

    Args:
        model_factory: 无参 callable, 每次调用返回一个新模型实例
        device: 训练设备
        amp: 是否使用 AMP 混合精度
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        device: torch.device | str = "cuda",
        amp: bool = True,
    ):
        self.model_factory = model_factory
        self.device = torch.device(device)
        self.amp = amp and self.device.type == "cuda"

    # ── 公开 API ─────────────────────────────────────────────────

    def fit(
        self,
        train_dataset,
        gold_mask: np.ndarray,
        gold_labels: np.ndarray,
        n_folds: int = 5,
        epochs: int = 30,
        batch_size: int = 8,
        lr: float = 2e-4,
        patience: int = 5,
        criterion: nn.Module | None = None,
        save_dir: str | Path = "/kaggle/working/teacher_oof",
    ) -> tuple[np.ndarray, list[dict]]:
        """运行 K-fold CV, 返回 OOF logits 和各折 checkpoint.

        Returns:
            oof_logits:  [N_total, 12] float32 — 每条训练数据的无偏 OOF logits
            fold_info:   [{fold, model_state, val_auc, epoch}, ...]
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        N = len(train_dataset)
        # K-fold on indices
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2026)

        # 用 gold 样本的 any-positive 做分层 (gold 太少, pseudo 全归一类)
        stratify = np.zeros(N, dtype=np.int32)
        gold_idx = np.where(gold_mask)[0]
        if len(gold_idx) > 0:
            gold_any_pos = (gold_labels.sum(axis=1) > 0).astype(np.int32)
            stratify[gold_idx] = gold_any_pos + 1  # 0=unlabeled, 1=gold_neg, 2=gold_pos

        oof_logits = np.full((N, 12), np.nan, dtype=np.float32)
        fold_info: list[dict] = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(N), stratify)):
            logger.info("=== Fold %d/%d: train=%d, val=%d ===",
                        fold_idx + 1, n_folds, len(train_idx), len(val_idx))

            model = self.model_factory().to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6,
            )
            scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

            if criterion is None:
                from losses.focal_bce import FocalBCELoss
                criterion = FocalBCELoss(gamma=2.0, alpha=0.25)

            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=2, pin_memory=True,
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                num_workers=2, pin_memory=True,
            )

            best_auc, patience_ctr = 0.0, 0
            best_state = None
            t_start = time.time()

            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                for batch in train_loader:
                    x = batch["image"].to(self.device)
                    y = batch["labels"].to(self.device)
                    weights = batch.get("weights")
                    if weights is not None:
                        weights = weights.to(self.device)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                        loss = criterion(model(x), y, weights)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()

                scheduler.step()

                # Validation
                model.eval()
                val_logits_dict = defaultdict(list)
                val_targets_dict = {}
                with torch.inference_mode():
                    for batch in val_loader:
                        x = batch["image"].to(self.device)
                        logits = model(x).float().cpu().numpy()
                        for uid, z, lbl in zip(batch["study_uid"], logits,
                                                batch["labels"].numpy()):
                            val_logits_dict[uid].append(z)
                            val_targets_dict[uid] = lbl

                uids = sorted(val_logits_dict)
                z_val = np.stack([np.mean(val_logits_dict[u], 0) for u in uids])
                y_val = np.stack([val_targets_dict[u] for u in uids])
                val_auc = _compute_macro_auc(y_val, z_val)

                elapsed = time.time() - t_start
                avg_sec = elapsed / (epoch + 1)
                remaining = avg_sec * (epochs - epoch - 1)
                logger.info(
                    "  F%d E%02d: loss=%.4f  val_auc=%.4f  ⏱ %ds  ETA %dmin",
                    fold_idx + 1, epoch + 1, total_loss / max(len(train_loader), 1),
                    val_auc, int(elapsed), int(remaining / 60),
                )

                if val_auc > best_auc + 1e-4:
                    best_auc = val_auc
                    patience_ctr = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        logger.info("  Early stopping at epoch %d", epoch + 1)
                        break

            # ── 加载最佳状态, 对该 fold 的 val 集做 OOF 预测 ──
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()

            fold_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                num_workers=2, pin_memory=True,
            )
            fold_logits = []
            fold_indices = []
            with torch.inference_mode():
                for batch in fold_loader:
                    x = batch["image"].to(self.device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                        logits = model(x).float().cpu().numpy()
                    fold_logits.append(logits)
                    fold_indices.extend(
                        batch.get("_idx", []) if "_idx" in batch else []
                    )

            fold_logits = np.concatenate(fold_logits)
            if fold_indices:
                for j, idx in enumerate(fold_indices):
                    oof_logits[idx] = fold_logits[j]
            else:
                oof_logits[val_idx] = fold_logits

            # 保存 checkpoint
            ckpt_path = save_dir / f"teacher_fold{fold_idx}.pt"
            torch.save({"model": best_state or model.state_dict(),
                         "fold": fold_idx, "auc": best_auc}, ckpt_path)
            fold_info.append({"fold": fold_idx, "val_auc": best_auc,
                              "ckpt_path": str(ckpt_path)})

            del model, train_loader, val_loader, train_subset, val_subset
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # ── 全量推理: ensemble of K models (可选, 用于 final teacher) ──
        # 保存 OOF logits 供后续使用
        np.save(save_dir / "image_teacher_oof_logits.npy", oof_logits)
        logger.info("OOF logits saved: %s", save_dir / "image_teacher_oof_logits.npy")
        return oof_logits, fold_info


# ═══════════════════════════════════════════════════════════════════
# PerClassFusion
# ═══════════════════════════════════════════════════════════════════

class PerClassFusion:
    """逐类校准融合 NLP 和 Image Teacher 预测.

    基于 58 gold 验证集, 对每个类别分别评估 NLP 和 Image Teacher
    的 AUC, 用 temperature-scaled softmax 计算融合权重.
    强源权重高, 弱源权重低.

    使用方式::

        fusion = PerClassFusion(temperature=5.0)
        fusion.fit(
            nlp_probs=nlp_calibrated_probs,        # [N, 12]
            image_oof_logits=image_teacher_logits,  # [N, 12]
            gold_mask=gold_study_mask,              # bool [N]
            gold_labels=gold_label_array,           # [N, 12]
        )
        teacher_probs = fusion.fuse(nlp_probs, image_oof_logits)
        print(fusion.summary())

    Args:
        temperature: softmax temperature (越大越接近平均, 越小越极端)
    """

    def __init__(self, temperature: float = 5.0):
        self.temperature = float(temperature)
        self.weights: dict[str, dict[str, float]] = {}  # {cls: {nlp, image}}
        self.aucs: dict[str, dict[str, float]] = {}     # {cls: {nlp, image}}

    def fit(
        self,
        nlp_probs: np.ndarray,          # [N, 12]
        image_oof_logits: np.ndarray,   # [N, 12]
        gold_mask: np.ndarray,          # bool [N]
        gold_labels: np.ndarray,        # [N, 12]
    ) -> "PerClassFusion":
        """在 gold 验证集上计算逐类融合权重.

        Args:
            nlp_probs: NLP 校准后的概率 [0, 1]
            image_oof_logits: Image Teacher OOF 原始 logits
            gold_mask: 哪些行是 gold study
            gold_labels: gold 二值标签 (与 nlp_probs 行对齐)
        """
        gold_idx = np.where(gold_mask)[0]
        if len(gold_idx) < 10:
            logger.warning("gold 样本不足 (%d), 使用平均融合", len(gold_idx))
            for i, cls in enumerate(TARGET_COLUMNS):
                self.weights[cls] = {"nlp": 0.5, "image": 0.5}
                self.aucs[cls] = {"nlp": 0.5, "image": 0.5}
            return self

        image_probs = 1.0 / (1.0 + np.exp(-image_oof_logits))  # sigmoid

        for i, cls in enumerate(TARGET_COLUMNS):
            y_true = gold_labels[gold_idx, i]
            y_nlp = nlp_probs[gold_idx, i]
            y_img = image_probs[gold_idx, i]

            # 跳过该类别无正/负样本的情况
            unique = np.unique(y_true)
            if len(unique) < 2:
                self.weights[cls] = {"nlp": 0.5, "image": 0.5}
                self.aucs[cls] = {"nlp": 0.5, "image": 0.5}
                continue

            nlp_auc = _binary_auc(y_true, y_nlp)
            img_auc = _binary_auc(y_true, y_img)

            # Temperature-scaled softmax
            T = self.temperature
            e_nlp = np.exp(np.clip(nlp_auc * T, -50, 50))
            e_img = np.exp(np.clip(img_auc * T, -50, 50))
            w_nlp = float(e_nlp / (e_nlp + e_img))
            w_img = float(1.0 - w_nlp)

            self.weights[cls] = {"nlp": w_nlp, "image": w_img}
            self.aucs[cls] = {"nlp": nlp_auc, "image": img_auc}

        return self

    def fuse(
        self,
        nlp_probs: np.ndarray,          # [N, 12]
        image_oof_logits: np.ndarray,   # [N, 12]
    ) -> np.ndarray:
        """逐类融合 NLP 和 Image 预测.

        Returns:
            teacher_probs: [N, 12] float32 — 融合后的概率
        """
        image_probs = 1.0 / (1.0 + np.exp(-np.clip(image_oof_logits, -50, 50)))
        fused = np.zeros_like(nlp_probs, dtype=np.float32)

        for i, cls in enumerate(TARGET_COLUMNS):
            w = self.weights.get(cls, {"nlp": 0.5, "image": 0.5})
            fused[:, i] = (
                w["nlp"] * nlp_probs[:, i] + w["image"] * image_probs[:, i]
            ).astype(np.float32)

        return fused

    def summary(self) -> str:
        """返回可读的融合权重摘要."""
        lines = ["Per-Class Fusion Weights (NLP | Image):"]
        for cls in TARGET_COLUMNS:
            w = self.weights.get(cls, {"nlp": 0.5, "image": 0.5})
            a = self.aucs.get(cls, {"nlp": 0.5, "image": 0.5})
            dominant = "NLP" if w["nlp"] > w["image"] else "IMG"
            lines.append(
                f"  {cls:<20s} NLP={a['nlp']:.3f}(w={w['nlp']:.2f})  "
                f"IMG={a['image']:.3f}(w={w['image']:.2f})  → {dominant}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _compute_macro_auc(targets: np.ndarray, logits: np.ndarray) -> float:
    """12 类 macro AUC (不依赖 sklearn)."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    n_classes = targets.shape[1]
    aucs = []
    for c in range(n_classes):
        yt = targets[:, c]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(_binary_auc(yt, probs[:, c]))
    return float(np.mean(aucs)) if aucs else 0.5


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC (无 sklearn 依赖)."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(y_true)
    y_true, scores = y_true[mask], scores[mask]

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores)
    rank = np.zeros(len(scores))
    rank[order] = np.arange(1, len(scores) + 1)

    # Average rank for ties
    for val in np.unique(scores):
        mask_val = scores == val
        if mask_val.sum() > 1:
            rank[mask_val] = rank[mask_val].mean()

    return float((rank[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
