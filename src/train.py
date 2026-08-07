"""2D/2.5D 切片级训练循环.

特性:
- AMP fp16 + channels_last
- Patient-level StratifiedGroupKFold (5 folds)
- Cosine warmup + early stopping (monitor: val_macro_auc)
- 2D (in_channels=1) 和 2.5D triplet (in_channels=3) 通用
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from src.data.dataset import KneeSliceDataset
from src.data.transforms import build_transforms
from src.models.classifier import KneeClassifier2D
from src.losses import build_loss
from src.metrics import compute_macro_auc, compute_per_class_auc

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_clip_norm: float = 1.0,
    device: str = "cuda",
) -> float:
    """训练一个 epoch, 返回平均 loss."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch in loader:
        images = batch["image"].to(device, memory_format=torch.channels_last)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> dict:
    """验证, 返回 loss + macro AUC + per-class AUC (均在切片级)."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, memory_format=torch.channels_last)
        labels = batch["labels"].to(device)

        logits = model(images)
        total_loss += criterion(logits, labels).item()

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_labels)

    return {
        "loss": total_loss / len(loader),
        "macro_auc": compute_macro_auc(targets, logits),
        "per_class_auc": compute_per_class_auc(targets, logits),
    }


def run_fold(
    config: dict,
    fold_idx: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    device: str = "cuda",
) -> dict:
    """训练单个 fold.

    Args:
        config: 完整配置字典
        fold_idx: fold 编号 (0-based)
        train_df: 训练集切片元数据
        valid_df: 验证集切片元数据
        device: 训练设备

    Returns:
        包含 fold_idx, best_auc, oof_logits 等的字典
    """
    model_cfg = config["model"]
    train_cfg = config["train"]
    data_cfg = config["data"]

    # 加载标签
    labels_df = pd.read_csv(Path(config["paths"]["train_csv"]))

    # 构建数据增强管道
    aug_cfg = config.get("augmentation", {})
    image_size = data_cfg["image_size"]
    train_transform = build_transforms(aug_cfg, image_size, is_train=True)
    valid_transform = build_transforms(aug_cfg, image_size, is_train=False)

    # 构建 dataset / loader
    train_ds = KneeSliceDataset(
        train_df, labels_df,
        npy_root=config["paths"]["npy_root"],
        image_size=image_size,
        in_channels=data_cfg.get("in_channels", 3),
        slice_offset=data_cfg.get("slice_offset", 1),
        is_train=True,
        transform=train_transform,
    )
    valid_ds = KneeSliceDataset(
        valid_df, labels_df,
        npy_root=config["paths"]["npy_root"],
        image_size=image_size,
        in_channels=data_cfg.get("in_channels", 3),
        slice_offset=data_cfg.get("slice_offset", 1),
        is_train=False,
        transform=valid_transform,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=data_cfg["loader"]["train_workers"],
        pin_memory=data_cfg["loader"]["pin_memory"],
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["loader"]["valid_workers"],
        pin_memory=data_cfg["loader"]["pin_memory"],
    )

    # 模型
    model = KneeClassifier2D(
        arch=model_cfg["arch"],
        pretrained=model_cfg["pretrained"],
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        drop_path_rate=model_cfg["drop_path_rate"],
    ).to(device)

    # 损失 & 优化器
    criterion = build_loss(config["loss"]["name"], **{k: v for k, v in config["loss"].items() if k != "name"})
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["optimizer"]["backbone_lr"]},
        {"params": model.head.parameters(), "lr": config["optimizer"]["head_lr"]},
    ], weight_decay=config["optimizer"]["weight_decay"])

    scaler = torch.cuda.amp.GradScaler() if train_cfg["mixed_precision"] else None

    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg["early_stopping"]["patience"]

    for epoch in range(train_cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device=device)
        val_metrics = validate_one_epoch(model, valid_loader, criterion, device=device)

        logger.info(f"Fold {fold_idx} Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  val_auc={val_metrics['macro_auc']:.4f}")

        if val_metrics["macro_auc"] > best_auc + train_cfg["early_stopping"]["min_delta"]:
            best_auc = val_metrics["macro_auc"]
            patience_counter = 0
            ckpt_path = Path(config["paths"]["checkpoint_dir"]) / f"fold{fold_idx}_best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "auc": best_auc}, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    return {"fold": fold_idx, "best_auc": best_auc}
