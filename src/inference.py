"""2.5D Triplane 推理管线.

- 加载所有 fold 的 checkpoint
- Top-K 切片均值聚合 (每个 study 取分数最高的 K% 切片)
- Fold ensemble: mean logits
- 生成 Kaggle submission.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TriplaneDataset
from src.models.triplane import TriplaneModel


def predict_study(
    model: TriplaneModel,
    study_npy_dir: Path,
    device: str = "cuda",
    topk_fraction: float = 0.25,
) -> np.ndarray:
    """对单个 study 的所有切片推理, 返回 Top-K 均值 logits.

    Args:
        model: 训练好的 TriplaneModel
        study_npy_dir: study 的 npy 目录
        device: 推理设备
        topk_fraction: 保留分数最高的切片比例

    Returns:
        np.ndarray: shape [12] 的 logits 均值
    """
    # TODO: 加载该 study 的所有切片
    # 1. 遍历所有 series/slices
    # 2. 每张切片走三平面 → logits
    # 3. 按每个类别的置信度排序, 取 top K% 取平均
    raise NotImplementedError


def generate_submission(
    config: dict,
    test_csv: str | Path,
    checkpoint_dir: str | Path,
    output_path: str | Path = "submission.csv",
) -> None:
    """生成 Kaggle 提交文件.

    Args:
        config: 模型/推理配置
        test_csv: 测试集索引 CSV
        checkpoint_dir: fold checkpoints 目录
        output_path: 输出 CSV 路径
    """
    # TODO: 完整的推理 → 提交管线
    # 1. 加载所有 fold 模型
    # 2. 对每个 test study 预测
    # 3. fold ensemble (mean logits)
    # 4. sigmoid → 写入 submission.csv
    raise NotImplementedError
