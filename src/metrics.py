"""评估指标 — Macro AUC ROC (12 类).

AUC 关注的是"排序质量"而非"绝对概率", 与竞赛指标一致.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_macro_auc(targets: np.ndarray, logits: np.ndarray) -> float:
    """计算 macro-averaged AUC ROC (12 类均值).

    Args:
        targets: [N, 12] 二值标签 (0/1)
        logits:  [N, 12] 模型原始输出 (未经过 sigmoid)

    Returns:
        float: 12 类 AUC 的算术平均值
    """
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    n_classes = targets.shape[1]
    aucs = []
    for c in range(n_classes):
        if targets[:, c].nunique() < 2:
            # 该类别在当前 fold 中只有一种标签, 跳过
            continue
        aucs.append(roc_auc_score(targets[:, c], probs[:, c]))
    return float(np.mean(aucs)) if aucs else 0.5


def compute_per_class_auc(targets: np.ndarray, logits: np.ndarray) -> dict[int, float]:
    """计算每个类别的 AUC.

    Returns:
        {class_idx: auc_value, ...}
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    n_classes = targets.shape[1]
    result = {}
    for c in range(n_classes):
        if targets[:, c].nunique() < 2:
            result[c] = 0.5
        else:
            result[c] = float(roc_auc_score(targets[:, c], probs[:, c]))
    return result
