"""Phase 1/2 训练入口 — EfficientNetV2-S 2.5D Baseline.

特性:
- EfficientNetV2-S, 5 通道输入, 384×384
- Focal BCE Loss (γ=2, α=0.25)
- AdamW + CosineAnnealingWarmRestarts
- Patient-level StratifiedGroupKFold (5 folds) — gold-only 模式
- 伪标签训练模式 — pseudo=train, gold=val
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

from datasets import Knee25DDataset, PseudoLabelLoader
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


def main(
    config_path: str | Path,
    epochs_override: int | None = None,
    pseudo_csv: str | None = None,
    confidence: str = "HIGH",
) -> dict:
    """完整训练管线.

    Args:
        config_path: YAML 配置文件路径
        epochs_override: 覆盖 epochs (用于快速验证)
        pseudo_csv: 伪标签 CSV 路径. 若提供, 进入伪标签训练模式:
                    训练集=pseudo labels, 验证集=gold labels (58 个标注样本)
                    若为 None, 使用原有的 KFold 模式 (仅 gold labels)
        confidence: 伪标签置信度过滤 ("HIGH" | "HIGH_PLUS_MEDIUM" | "ALL")
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

    # 2. 加载 series 元数据
    series_df = pd.read_csv(paths["series_csv"])

    # ── 伪标签模式 ───────────────────────────────────────────────
    if pseudo_csv is not None:
        return _main_pseudo(
            config=config,
            series_df=series_df,
            pseudo_csv=pseudo_csv,
            gold_csv=paths["train_csv"],
            confidence=confidence,
        )

    # ── Gold-only KFold 模式 ─────────────────────────────────────
    val_cfg = config["validation"]
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
        "mode": "kfold_gold_only",
        "fold_aucs": [float(a) for a in fold_aucs],
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"摘要已保存: {out_dir / 'summary.json'}")

    return {"mean_auc": mean_auc, "std_auc": std_auc, "fold_results": fold_results}


# ═══════════════════════════════════════════════════════════════════
# 伪标签训练模式
# ═══════════════════════════════════════════════════════════════════


def _main_pseudo(
    config: dict,
    series_df: pd.DataFrame,
    pseudo_csv: str,
    gold_csv: str,
    confidence: str = "HIGH",
) -> dict:
    """伪标签训练: train on pseudo labels, validate on gold labels.

    不执行 KFold — 使用 NLP 管线定义的固定 split:
    - 训练集: pseudo_labels.csv (置信度过滤后) ~3500-4000 studies
    - 验证集: gold labels (58 个标注样本)

    Args:
        config: 完整配置 dict
        series_df: train_series.csv DataFrame
        pseudo_csv: 伪标签 CSV 路径
        gold_csv: gold label CSV 路径
        confidence: 置信度过滤级别

    Returns:
        {"best_auc": float, "summary": dict}
    """
    paths = config["paths"]
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    opt_cfg = config["optimizer"]
    sched_cfg = config.get("scheduler", {})

    exp_name = config["experiment"]["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. 加载标签 ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("伪标签训练模式 — Pseudo-Label Training")
    logger.info(f"  伪标签文件:   {pseudo_csv}")
    logger.info(f"  置信度过滤:   {confidence}")
    logger.info("=" * 60)

    loader = PseudoLabelLoader(pseudo_csv=pseudo_csv)

    # 打印伪标签统计
    stats = loader.stats()
    logger.info(
        "伪标签总体: %d studies, HIGH=%d, HIGH+MEDIUM=%d",
        stats["total"],
        stats["by_confidence_overall"].get("HIGH", 0),
        stats["by_confidence_overall"].get("HIGH_PLUS_MEDIUM", 0),
    )

    # 加载并合并
    train_labels_df, val_labels_df = loader.merge_with_gold(
        gold_csv=gold_csv,
        confidence=confidence,
        val_from_gold=True,
    )

    logger.info(f"训练集: {len(train_labels_df)} studies (伪标签)")
    logger.info(f"验证集: {len(val_labels_df)} studies (gold)")

    # 打印类别分布
    for col in TARGET_COLUMNS:
        train_pos = train_labels_df[col].sum() if col in train_labels_df.columns else 0
        val_pos = val_labels_df[col].sum() if col in val_labels_df.columns else 0
        logger.info(f"  {col:<20s}  train_pos={int(train_pos):5d}  val_pos={int(val_pos):3d}")

    # ── 2. 构建 Dataset ────────────────────────────────────────────
    ds_kwargs = dict(
        dicom_root=paths["dicom_root"],
        planes=data_cfg["planes"],
        image_size=data_cfg["image_size"],
        slice_count=data_cfg["slice_count"],
        fluid_sensitive_only=data_cfg.get("fluid_sensitive_only", True),
        fat_suppression_only=data_cfg.get("fat_suppression_only", True),
    )

    train_ds = Knee25DDataset(series_df, train_labels_df, is_train=True, **ds_kwargs)
    val_ds = Knee25DDataset(series_df, val_labels_df, is_train=False, **ds_kwargs)

    logger.info(f"训练样本数 (切片级): {len(train_ds):,}")
    logger.info(f"验证样本数 (切片级): {len(val_ds):,}")

    # ── 3. DataLoader ──────────────────────────────────────────────
    loader_cfg = data_cfg["loader"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=loader_cfg["train_workers"],
        pin_memory=loader_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["valid_workers"],
        pin_memory=loader_cfg["pin_memory"],
    )

    # ── 4. 模型 ────────────────────────────────────────────────────
    model = EfficientNetV2S25D(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    ).to(device)

    # ── 5. 损失 & 优化器 & 调度器 ─────────────────────────────────
    criterion = FocalBCELoss(
        gamma=loss_cfg["gamma"],
        alpha=loss_cfg["alpha"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=opt_cfg["betas"],
    )

    scaler = torch.amp.GradScaler("cuda") if train_cfg["mixed_precision"] else None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg.get("T_0", 10),
        T_mult=sched_cfg.get("T_mult", 2),
        eta_min=sched_cfg.get("eta_min", 1e-6),
    )

    # ── 6. 训练循环 ────────────────────────────────────────────────
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg["early_stopping"]["patience"]
    train_start = time.time()

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            grad_clip_norm=train_cfg["gradient_clip_norm"], device=device,
        )
        val_metrics = validate_one_epoch(model, val_loader, criterion, device=device)

        elapsed = time.time() - t0

        # ETA
        epochs_done = epoch + 1
        avg_sec = (time.time() - train_start) / epochs_done
        remaining = avg_sec * (train_cfg["epochs"] - epochs_done)
        eta_str = f"{remaining/60:.0f}min" if remaining < 3600 else f"{remaining/3600:.1f}h"

        # VRAM
        gpu_alloc = torch.cuda.max_memory_allocated(device) / 1024**3 if device == "cuda" else 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:3d}: "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_auc={val_metrics['macro_auc']:.4f}  lr={lr_now:.2e}  "
            f"⏱ {elapsed:.0f}s/epoch  ETA {eta_str}  VRAM peak={gpu_alloc:.1f}GB"
        )

        # Per-class AUC
        per_class = val_metrics.get("per_class_auc", {})
        if per_class:
            logger.info(f"        per-class → {format_per_class_auc(per_class)}")

        scheduler.step()

        # Early stopping + checkpoint
        if val_metrics["macro_auc"] > best_auc + train_cfg["early_stopping"]["min_delta"]:
            best_auc = val_metrics["macro_auc"]
            patience_counter = 0
            ckpt_path = Path(paths["checkpoint_dir"]) / "best_model.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "auc": best_auc,
                    "config_summary": {
                        "confidence": confidence,
                        "n_train": len(train_labels_df),
                        "n_val": len(val_labels_df),
                    },
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ── 7. 保存摘要 ────────────────────────────────────────────────
    total_time = time.time() - train_start
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": exp_name,
        "mode": "pseudo_label",
        "confidence": confidence,
        "n_train_studies": len(train_labels_df),
        "n_val_studies": len(val_labels_df),
        "best_auc": best_auc,
        "total_time_hours": round(total_time / 3600, 2),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"摘要已保存: {out_dir / 'summary.json'}")

    logger.info(f"\n{'='*60}")
    logger.info(f"伪标签训练完成! Best AUC: {best_auc:.4f} | 耗时: {total_time/3600:.1f}h")
    logger.info(f"{'='*60}")

    return {"best_auc": best_auc, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSNA Knee — Training Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/efficientnet.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="覆盖配置文件中的 epochs 数",
    )
    parser.add_argument(
        "--pseudo_csv", type=str, default=None,
        help="伪标签 CSV 路径 (启用伪标签训练模式)",
    )
    parser.add_argument(
        "--confidence", type=str, default="HIGH",
        choices=["HIGH", "HIGH_PLUS_MEDIUM", "ALL"],
        help="伪标签置信度过滤级别 (默认 HIGH)",
    )
    args = parser.parse_args()
    main(
        args.config,
        epochs_override=args.epochs,
        pseudo_csv=args.pseudo_csv,
        confidence=args.confidence,
    )
