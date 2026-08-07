"""评估指标 — Macro AUC ROC (12 类).

竞赛指标: "macro-averaged AUC ROC across the twelve targets"
含义: 对 12 个类别各自计算 AUC, 然后取算术平均
     AUC 衡量的是"排序质量" 而非 "绝对概率值"
     类别不均衡对 AUC 的影响天然小于 Accuracy/F1
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_macro_auc(targets: np.ndarray, logits: np.ndarray) -> float:
    """计算 macro-averaged AUC ROC (12 类算术平均).

    用法:
        val_metrics = validate_one_epoch(model, loader, criterion)
        print(f"macro AUC: {val_metrics['macro_auc']:.4f}")

    Args:
        targets: [N, 12] 二值标签 (0 或 1)
        logits:  [N, 12] 模型原始 logits (未经过 sigmoid)

    Returns:
        float: 12 类 AUC 的算术平均值. 如果某类只有一种标签 (无法算 AUC), 跳过
    """
    probs = 1.0 / (1.0 + np.exp(-logits))          # sigmoid
    n_classes = targets.shape[1]
    aucs = []
    for c in range(n_classes):
        # 检查该类别是否同时有正负样本 (AUC 需要两类都出现)
        unique_vals = np.unique(targets[:, c])
        if len(unique_vals) < 2:
            continue                                  # 跳过 (例如 mini 集某 fold 恰好没有正样本)
        aucs.append(roc_auc_score(targets[:, c], probs[:, c]))

    return float(np.mean(aucs)) if aucs else 0.5


def compute_per_class_auc(targets: np.ndarray, logits: np.ndarray) -> dict[int, float]:
    """计算每个类别的独立 AUC, 方便定位哪类效果最差.

    Returns:
        {class_idx: auc_value, ...}
        例如: {0: 0.72, 1: 0.58, ...}  → 第 1 类 (ACL) 0.72, 第 2 类 (MCL) 0.58
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    n_classes = targets.shape[1]
    result = {}
    for c in range(n_classes):
        unique_vals = np.unique(targets[:, c])
        if len(unique_vals) < 2:
            result[c] = 0.5                            # 无法计算, 返回 0.5 (随机基线)
        else:
            result[c] = float(roc_auc_score(targets[:, c], probs[:, c]))
    return result
