"""2.5D Triplane 训练循环.

特性:
- AMP fp16 混合精度 + channels_last 内存布局
- 梯度累积 (batch=4 * 4 steps = effective 16)
- Patient-level StratifiedGroupKFold (5 folds)
- Cosine warmup + early stopping
- 每 epoch 报告 per-class AUC + macro AUC
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from src.data.dataset import TriplaneDataset
from src.models.triplane import TriplaneModel
from src.losses import build_loss
from src.metrics import compute_macro_auc, compute_per_class_auc

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler | None,
    accumulation_steps: int = 1,
    grad_clip_norm: float = 1.0,
    device: str = "cuda",
) -> dict[str, float]:
    """训练一个 epoch."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        axial = batch["axial"].to(device, memory_format=torch.channels_last)
        coronal = batch["coronal"].to(device, memory_format=torch.channels_last)
        sagittal = batch["sagittal"].to(device, memory_format=torch.channels_last)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(axial, coronal, sagittal)
            loss = criterion(logits, labels) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return {"loss": total_loss / len(loader)}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> dict[str, Any]:
    """验证一个 epoch, 返回 loss + macro AUC + per-class AUC."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    for batch in loader:
        axial = batch["axial"].to(device, memory_format=torch.channels_last)
        coronal = batch["coronal"].to(device, memory_format=torch.channels_last)
        sagittal = batch["sagittal"].to(device, memory_format=torch.channels_last)
        labels = batch["labels"].to(device)

        logits = model(axial, coronal, sagittal)
        total_loss += criterion(logits, labels).item()

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_labels)

    macro_auc = compute_macro_auc(targets, logits)
    per_class = compute_per_class_auc(targets, logits)

    return {
        "loss": total_loss / len(loader),
        "macro_auc": macro_auc,
        "per_class_auc": per_class,
    }


def main(config: dict) -> None:
    """完整训练入口."""
    # TODO: 从 config 加载数据、创建 folds、循环训练
    # 1. 加载 image_index_csv → DataFrame
    # 2. StratifiedGroupKFold split (patient-level)
    # 3. for each fold: train_one_epoch → validate → checkpoint
    # 4. OOF 聚合 + 最终 macro AUC 报告
    raise NotImplementedError("训练入口待实现 — 见 TODO 注释")
