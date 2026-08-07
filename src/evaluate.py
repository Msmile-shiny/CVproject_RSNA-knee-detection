"""2.5D Triplane OOF (Out-of-Fold) 评估.

- 5-fold 交叉验证
- 每 fold 保存 logits
- 最终聚合: fold ensemble (mean logits) + top-K 均值
- 生成 per-class AUC 报告
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from src.data.dataset import TriplaneDataset
from src.models.triplane import TriplaneModel
from src.metrics import compute_macro_auc, compute_per_class_auc


def run_oof_evaluation(
    index_df: pd.DataFrame,
    config: dict,
    target_columns: list[str],
    npy_root: str = "data/mini/npy",
    n_folds: int = 5,
    device: str = "cuda",
) -> dict:
    """跑完整的 OOF 评估管线.

    Returns:
        dict: {"macro_auc": float, "per_class_auc": dict, "oof_logits": np.ndarray}
    """
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config.get("seed", 2026))

    # 生成伪标签用于 StratifiedGroupKFold (多标签 → 取最常见标签)
    labels = index_df[target_columns].values
    pseudo_y = labels.argmax(axis=1)

    oof_logits = np.zeros((len(index_df), len(target_columns)), dtype=np.float32)

    for fold_idx, (train_idx, valid_idx) in enumerate(
        skf.split(index_df, pseudo_y, groups=index_df["patient_id"])
    ):
        print(f"Fold {fold_idx + 1}/{n_folds}")
        # TODO: 训练一个 fold
        # 1. 构建 train/valid dataset
        # 2. 初始化 TriplaneModel
        # 3. train_one_epoch × N epochs
        # 4. validate → 保存 logits 到 oof_logits[valid_idx]

    macro_auc = compute_macro_auc(labels, oof_logits)
    per_class = compute_per_class_auc(labels, oof_logits)

    return {"macro_auc": macro_auc, "per_class_auc": per_class, "oof_logits": oof_logits}
