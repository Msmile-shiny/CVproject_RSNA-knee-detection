"""2D/2.5D 切片级训练循环.

特性:
- AMP fp16 + channels_last
- Patient-level StratifiedGroupKFold (5 folds)
- Cosine warmup + early stopping (monitor: val_macro_auc)
- 2D (in_channels=1) 和 2.5D triplet (in_channels=3) 通用
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# 静默 albumentations 版本更新检查
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# 允许 python src/train.py 直接运行（把项目根目录加入 Python 路径）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from src.data.dataset import KneeSliceDataset
from src.data.transforms import build_transforms
from src.models.classifier import KneeClassifier2D
from src.losses import build_loss
from src.metrics import compute_macro_auc, compute_per_class_auc
from src.evaluate import aggregate_to_study

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    labels_df = pd.read_csv(Path(config["paths"]["train_csv"]), encoding="utf-8")
    target_cols = data_cfg.get("target_columns") or [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    ]
    for col in target_cols:
        labels_df[col] = pd.to_numeric(labels_df[col], errors="coerce").fillna(0).astype(int)

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

    # 损失 & 优化器（pos_weight="auto" → 先关掉，后续 ASL 自然会处理不平衡）
    loss_kwargs = {k: v for k, v in config["loss"].items() if k != "name"}
    if loss_kwargs.get("pos_weight") == "auto":
        loss_kwargs["pos_weight"] = None
    criterion = build_loss(config["loss"]["name"], **loss_kwargs)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["optimizer"]["backbone_lr"]},
        {"params": model.head.parameters(), "lr": config["optimizer"]["head_lr"]},
    ], weight_decay=config["optimizer"]["weight_decay"])

    scaler = torch.cuda.amp.GradScaler() if train_cfg["mixed_precision"] else None

    # 学习率调度器: warmup → cosine decay
    sched_cfg = config.get("scheduler", {})
    warmup_epochs = sched_cfg.get("warmup_epochs", 3)
    total_epochs = train_cfg["epochs"]

    def _lr_lambda(epoch: int) -> float:
        """线性 warmup → 余弦退火, 对所有 param group 统一缩放."""
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg["early_stopping"]["patience"]
    fold_start = time.time()

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device=device)
        val_metrics = validate_one_epoch(model, valid_loader, criterion, device=device)
        elapsed = time.time() - t0

        # ETA
        epochs_done = epoch + 1
        avg_sec = (time.time() - fold_start) / epochs_done
        remaining = avg_sec * (train_cfg["epochs"] - epochs_done)
        eta_str = f"{remaining/60:.0f}min" if remaining < 3600 else f"{remaining/3600:.1f}h"

        # GPU 显存
        gpu_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
        gpu_reserved = torch.cuda.memory_reserved(device) / 1024**3
        torch.cuda.reset_peak_memory_stats(device)

        lr_now = scheduler.get_last_lr()[0]
        logger.info(
            f"Fold {fold_idx} Epoch {epoch:3d}: "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_auc={val_metrics['macro_auc']:.4f}  lr={lr_now:.2e}  "
            f"⏱ {elapsed:.0f}s/epoch  ETA {eta_str}  "
            f"VRAM peak={gpu_alloc:.1f}/{gpu_reserved:.1f}GB"
        )

        scheduler.step()

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


# ── 主入口 ────────────────────────────────────────────────────────


def main(config_path: str | Path, epochs_override: int | None = None) -> dict:
    """完整训练入口: 加载配置 → KFold split → 逐 fold 训练 → OOF 评估.

    Args:
        config_path: YAML 配置文件路径
        epochs_override: 覆盖配置文件中的 epochs (用于快速干跑)

    Returns:
        包含 oof_macro_auc, per_class_auc, fold_results 的字典
    """
    # 1. 加载配置 ──────────────────────────────────────────────
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if epochs_override is not None:
        config["train"]["epochs"] = epochs_override
        # 早停 patience 也要缩短, 否则 3 epoch 内永远不会触发
        config["train"]["early_stopping"]["patience"] = min(
            config["train"]["early_stopping"]["patience"], epochs_override
        )

    exp_name = config["experiment"]["name"]
    logger.info(f"实验: {exp_name}")
    logger.info(f"配置: {config_path}")

    # 2. 加载数据 ──────────────────────────────────────────────
    paths = config["paths"]
    meta_df = pd.read_csv(Path(paths["metadata_csv"]), encoding="utf-8")
    labels_df = pd.read_csv(Path(paths["train_csv"]), encoding="utf-8")
    data_cfg = config["data"]
    target_cols = data_cfg.get("target_columns") or [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    ]

    # 清洗标签
    for col in target_cols:
        labels_df[col] = pd.to_numeric(labels_df[col], errors="coerce").fillna(0).astype(int)

    # 过滤: 只保留在 labels_df 中有标签的 study
    valid_studies = set(meta_df["StudyInstanceUID"]) & set(labels_df["StudyInstanceUID"])
    meta_df = meta_df[meta_df["StudyInstanceUID"].isin(valid_studies)].copy()
    logger.info(f"切片元数据: {len(meta_df):,} 条")
    logger.info(f"有效 study:  {len(valid_studies):,}")

    # 3. Study 级 KFold split ──────────────────────────────────
    val_cfg = config["validation"]
    n_folds = val_cfg["folds"]

    # 构建 study 级表: 每个 study 一行, 含伪标签
    study_labels = labels_df.set_index("StudyInstanceUID").loc[list(valid_studies)]
    study_label_arr = study_labels[target_cols].values
    pseudo_y = (study_label_arr.sum(axis=1) > 0).astype(int)  # 0=完全正常, 1=有异常
    study_ids = study_labels.index.values

    skf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=config["experiment"]["seed"]
    )

    fold_results = []
    all_meta_rows = []       # 收集每 fold 的 (slice_idx, study_uid, fold_logits)
    study_meta = {}           # study_uid → [slices 的全局索引]

    for fold_idx, (train_sids, valid_sids) in enumerate(
        skf.split(study_ids, pseudo_y, groups=study_ids)
    ):
        train_studies = set(study_ids[train_sids])
        valid_studies = set(study_ids[valid_sids])

        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold_idx + 1}/{n_folds}: train studies={len(train_studies)}, valid studies={len(valid_studies)}")
        logger.info(f"{'='*50}")

        # 切片级过滤
        train_meta = meta_df[meta_df["StudyInstanceUID"].isin(train_studies)]
        valid_meta = meta_df[meta_df["StudyInstanceUID"].isin(valid_studies)]

        result = run_fold(config, fold_idx, train_meta, valid_meta)
        fold_results.append(result)
        logger.info(f"Fold {fold_idx} best AUC: {result['best_auc']:.4f}")

    # 4. 汇总结果 ──────────────────────────────────────────────
    fold_aucs = [r["best_auc"] for r in fold_results]
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    logger.info(f"\n{'='*50}")
    logger.info(f"训练完成 — KFold 结果")
    logger.info(f"{'='*50}")
    logger.info(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    logger.info(f"Mean AUC:  {mean_auc:.4f} ± {std_auc:.4f}")

    # 5. 保存摘要 ──────────────────────────────────────────────
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": exp_name,
        "fold_aucs": [float(a) for a in fold_aucs],
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }
    import json
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"摘要已保存: {out_dir / 'summary.json'}")

    return {"mean_auc": mean_auc, "std_auc": std_auc, "fold_results": fold_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSNA Knee — 训练入口")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 epochs 数 (快速干跑)")
    args = parser.parse_args()
    main(args.config, epochs_override=args.epochs)
