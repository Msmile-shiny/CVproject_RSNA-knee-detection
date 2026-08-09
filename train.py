"""Phase 1 训练入口 — EfficientNetV2-S 2.5D Baseline.

特性:
- EfficientNetV2-S, 5 通道输入, 384×384
- Focal BCE Loss (γ=2, α=0.25)
- AdamW + CosineAnnealingWarmRestarts
- Patient-level StratifiedGroupKFold (5 folds)
- AMP fp16, gradient clipping, early stopping
- Per-class AUC 日志, VRAM 追踪, ETA
"""

from __future__ import annotations

import argparse
import json
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

from datasets import Knee25DDataset
from models import EfficientNetV2S25D
from losses import FocalBCELoss
from utils import (
    compute_macro_auc,
    compute_per_class_auc,
    format_per_class_auc,
    TARGET_COLUMNS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 训练 / 验证 循环
# ═══════════════════════════════════════════════════════════════════


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None,
    grad_clip_norm: float = 1.0,
    device: str = "cuda",
) -> float:
    """训练一个 epoch, 返回平均 loss."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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
    """验证, 返回 loss + macro AUC + per-class AUC (切片级)."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
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


# ═══════════════════════════════════════════════════════════════════
# 单 Fold 训练
# ═══════════════════════════════════════════════════════════════════


def run_fold(
    config: dict,
    fold_idx: int,
    train_meta: pd.DataFrame,
    valid_meta: pd.DataFrame,
    labels_df: pd.DataFrame,
    device: str = "cuda",
) -> dict:
    """训练单个 fold.

    Returns:
        {"fold": int, "best_auc": float}
    """
    model_cfg = config["model"]
    train_cfg = config["train"]
    data_cfg = config["data"]
    paths = config["paths"]

    # ── Dataset / Loader ──────────────────────────────────────
    ds_kwargs = dict(
        dicom_root=paths["dicom_root"],
        planes=data_cfg["planes"],
        image_size=data_cfg["image_size"],
        slice_count=data_cfg["slice_count"],
        fluid_sensitive_only=data_cfg.get("fluid_sensitive_only", True),
        fat_suppression_only=data_cfg.get("fat_suppression_only", True),
    )

    train_ds = Knee25DDataset(train_meta, labels_df, is_train=True, **ds_kwargs)
    valid_ds = Knee25DDataset(valid_meta, labels_df, is_train=False, **ds_kwargs)

    loader_kwargs = dict(
        num_workers=data_cfg["loader"]["train_workers"],
        pin_memory=data_cfg["loader"]["pin_memory"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True, **loader_kwargs
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["loader"]["valid_workers"],
        pin_memory=data_cfg["loader"]["pin_memory"],
    )

    # ── 模型 ──────────────────────────────────────────────────
    model = EfficientNetV2S25D(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    ).to(device)

    # ── 损失 & 优化器 ─────────────────────────────────────────
    loss_cfg = config["loss"]
    criterion = FocalBCELoss(
        gamma=loss_cfg["gamma"],
        alpha=loss_cfg["alpha"],
    )

    opt_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=opt_cfg["betas"],
    )

    scaler = torch.amp.GradScaler("cuda") if train_cfg["mixed_precision"] else None

    # ── 调度器: CosineAnnealingWarmRestarts ───────────────────
    sched_cfg = config.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg.get("T_0", 10),
        T_mult=sched_cfg.get("T_mult", 2),
        eta_min=sched_cfg.get("eta_min", 1e-6),
    )

    # ── 训练循环 ──────────────────────────────────────────────
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg["early_stopping"]["patience"]
    fold_start = time.time()

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            grad_clip_norm=train_cfg["gradient_clip_norm"], device=device,
        )
        val_metrics = validate_one_epoch(model, valid_loader, criterion, device=device)

        elapsed = time.time() - t0

        # ETA
        epochs_done = epoch + 1
        avg_sec = (time.time() - fold_start) / epochs_done
        remaining = avg_sec * (train_cfg["epochs"] - epochs_done)
        eta_str = f"{remaining/60:.0f}min" if remaining < 3600 else f"{remaining/3600:.1f}h"

        # VRAM
        gpu_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
        torch.cuda.reset_peak_memory_stats(device)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Fold {fold_idx} Epoch {epoch:3d}: "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_auc={val_metrics['macro_auc']:.4f}  lr={lr_now:.2e}  "
            f"⏱ {elapsed:.0f}s/epoch  ETA {eta_str}  VRAM peak={gpu_alloc:.1f}GB"
        )

        # Per-class AUC
        per_class = val_metrics.get("per_class_auc", {})
        if per_class:
            logger.info(f"Fold {fold_idx}          per-class → {format_per_class_auc(per_class)}")

        scheduler.step()

        # Early stopping
        if val_metrics["macro_auc"] > best_auc + train_cfg["early_stopping"]["min_delta"]:
            best_auc = val_metrics["macro_auc"]
            patience_counter = 0
            ckpt_path = Path(paths["checkpoint_dir"]) / f"fold{fold_idx}_best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "auc": best_auc},
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Fold {fold_idx}: early stopping at epoch {epoch}")
                break

    return {"fold": fold_idx, "best_auc": best_auc}


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════


def main(config_path: str | Path, epochs_override: int | None = None) -> dict:
    """完整训练管线.

    Args:
        config_path: YAML 配置文件路径
        epochs_override: 覆盖 epochs (用于快速验证)
    """
    # 1. 加载配置
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if epochs_override is not None:
        config["train"]["epochs"] = epochs_override
        config["train"]["early_stopping"]["patience"] = min(
            config["train"]["early_stopping"]["patience"], epochs_override
        )

    exp_name = config["experiment"]["name"]
    logger.info(f"实验: {exp_name} | 配置: {config_path}")

    paths = config["paths"]
    data_cfg = config["data"]
    val_cfg = config["validation"]

    # 2. 加载元数据
    series_df = pd.read_csv(paths["series_csv"])
    labels_df = pd.read_csv(paths["train_csv"])

    # 清洗标签
    for col in TARGET_COLUMNS:
        if col in labels_df.columns:
            labels_df[col] = pd.to_numeric(
                labels_df[col], errors="coerce"
            ).fillna(0).astype(int)

    # 只保留有标签的 study (58 个标注样本)
    valid_studies = set(series_df["StudyInstanceUID"]) & set(labels_df["StudyInstanceUID"])
    series_df = series_df[series_df["StudyInstanceUID"].isin(valid_studies)].copy()

    logger.info(f"Series 元数据: {len(series_df):,} 行")
    logger.info(f"有效 study:    {len(valid_studies):,}")

    # 3. Study 级 KFold Split
    n_folds = val_cfg["folds"]
    study_labels = labels_df.set_index("StudyInstanceUID").loc[list(valid_studies)]
    study_label_arr = study_labels[TARGET_COLUMNS].values
    pseudo_y = (study_label_arr.sum(axis=1) > 0).astype(int)
    study_ids = study_labels.index.values

    skf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True,
        random_state=config["experiment"]["seed"],
    )

    fold_results = []

    for fold_idx, (train_sids, valid_sids) in enumerate(
        skf.split(study_ids, pseudo_y, groups=study_ids)
    ):
        train_studies = set(study_ids[train_sids])
        valid_studies_set = set(study_ids[valid_sids])

        logger.info(f"\n{'='*50}")
        logger.info(
            f"Fold {fold_idx + 1}/{n_folds}: "
            f"train={len(train_studies)}, valid={len(valid_studies_set)}"
        )
        logger.info(f"{'='*50}")

        train_meta = series_df[series_df["StudyInstanceUID"].isin(train_studies)]
        valid_meta = series_df[series_df["StudyInstanceUID"].isin(valid_studies_set)]

        result = run_fold(config, fold_idx, train_meta, valid_meta, labels_df)
        fold_results.append(result)
        logger.info(f"Fold {fold_idx} best AUC: {result['best_auc']:.4f}")

    # 4. 汇总
    fold_aucs = [r["best_auc"] for r in fold_results]
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    logger.info(f"\n{'='*50}")
    logger.info(f"训练完成 — KFold 结果")
    logger.info(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    logger.info(f"Mean AUC:  {mean_auc:.4f} ± {std_auc:.4f}")

    # 5. 保存摘要
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": exp_name,
        "fold_aucs": [float(a) for a in fold_aucs],
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"摘要已保存: {out_dir / 'summary.json'}")

    return {"mean_auc": mean_auc, "std_auc": std_auc, "fold_results": fold_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSNA Knee — Phase 1 Training")
    parser.add_argument("--config", type=str, default="configs/efficientnet.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.config, epochs_override=args.epochs)
