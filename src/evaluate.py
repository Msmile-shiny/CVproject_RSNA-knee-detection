"""2D/2.5D 切片级 OOF (Out-of-Fold) 评估.

- 5-fold Patient-level StratifiedGroupKFold
- 每 fold 保存 logits/triplet logits
- Study-level 聚合: top-K 均值
- Fold ensemble: mean logits
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import compute_macro_auc, compute_per_class_auc


def aggregate_to_study(
    slice_logits: np.ndarray,
    slice_study_ids: np.ndarray,
    topk_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """切片级 logits → study 级 logits (Top-K 均值).

    Args:
        slice_logits: [N_slices, 12]
        slice_study_ids: [N_slices] study UID 数组
        topk_fraction: 每个 study 保留的 top 切片比例

    Returns:
        study_logits: [N_studies, 12]
        study_ids: [N_studies] 对应的 study UID
    """
    unique_studies = np.unique(slice_study_ids)
    study_logits = np.zeros((len(unique_studies), slice_logits.shape[1]), dtype=np.float32)

    for i, sid in enumerate(unique_studies):
        mask = slice_study_ids == sid
        sid_logits = slice_logits[mask]                          # [K, 12]
        k = max(1, int(len(sid_logits) * topk_fraction))
        top_vals = np.sort(sid_logits, axis=0)[-k:]              # per-class top-K
        study_logits[i] = top_vals.mean(axis=0)

    return study_logits, unique_studies
