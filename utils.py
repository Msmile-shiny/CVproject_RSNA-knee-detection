"""工具函数 — 指标计算、聚合、辅助.

- macro-averaged AUC ROC (12 类)
- per-class AUC
- 切片级 → Study 级聚合 (Top-K 均值 / Slice Attention)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


# ── 12 类标签短名映射 ────────────────────────────────────────────
CLASS_ABBR = {
    "ACL": "ACL", "MCL": "MCL",
    "Medial Meniscus": "MMen", "Lateral Meniscus": "LMen",
    "Medial OA": "MOA", "Lateral OA": "LOA", "PF OA": "PFOA",
    "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak",
    "Contusion": "Con", "Fracture": "Frx",
}

TARGET_COLUMNS = list(CLASS_ABBR.keys())


def compute_macro_auc(targets: np.ndarray, logits: np.ndarray) -> float:
    """12 类 macro-averaged AUC ROC.

    Args:
        targets: [N, 12] 二值标签
        logits:  [N, 12] 模型原始 logits

    Returns:
        float: 12 类 AUC 算术平均. 某类无正/负样本则跳过.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))          # sigmoid
    n_classes = targets.shape[1]
    aucs = []
    for c in range(n_classes):
        unique_vals = np.unique(targets[:, c])
        if len(unique_vals) < 2:
            continue
        aucs.append(roc_auc_score(targets[:, c], probs[:, c]))
    return float(np.mean(aucs)) if aucs else 0.5


def compute_per_class_auc(targets: np.ndarray, logits: np.ndarray) -> dict[int, float]:
    """逐类 AUC.

    Returns: {class_idx: auc_value, ...}
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    n_classes = targets.shape[1]
    result = {}
    for c in range(n_classes):
        unique_vals = np.unique(targets[:, c])
        if len(unique_vals) < 2:
            result[c] = 0.5
        else:
            result[c] = float(roc_auc_score(targets[:, c], probs[:, c]))
    return result


def aggregate_to_study(
    slice_logits: np.ndarray,
    slice_study_ids: np.ndarray,
    topk_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """切片级 logits → Study 级 logits (Top-K 均值聚合).

    Args:
        slice_logits: [N_slices, 12]
        slice_study_ids: [N_slices] study UID
        topk_fraction: 每 study 保留的 top 切片比例

    Returns:
        study_logits: [N_studies, 12]
        study_ids: [N_studies]
    """
    unique_studies = np.unique(slice_study_ids)
    study_logits = np.zeros(
        (len(unique_studies), slice_logits.shape[1]), dtype=np.float32
    )

    for i, sid in enumerate(unique_studies):
        mask = slice_study_ids == sid
        sid_logits = slice_logits[mask]                   # [K, 12]
        k = max(1, int(len(sid_logits) * topk_fraction))
        top_vals = np.sort(sid_logits, axis=0)[-k:]
        study_logits[i] = top_vals.mean(axis=0)

    return study_logits, unique_studies


def format_per_class_auc(per_class: dict[int, float]) -> str:
    """将 per-class AUC dict 格式化为紧凑字符串."""
    parts = []
    for idx, col in enumerate(TARGET_COLUMNS):
        abbr = CLASS_ABBR.get(col, col[:4])
        auc_val = per_class.get(idx, 0.5)
        parts.append(f"{abbr}:{auc_val:.2f}")
    return "  ".join(parts)
