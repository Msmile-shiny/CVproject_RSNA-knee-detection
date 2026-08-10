"""Phase 1/2 训练入口 — EfficientNetV2-S 2.5D Baseline.

特性:
- EfficientNetV2-S, 5 通道输入, 384×384 (单平面 / 三平面)
- Focal BCE Loss (γ=2, α=0.25)
- AdamW + CosineAnnealingWarmRestarts
- Patient-level StratifiedGroupKFold (5 folds) — gold-only 模式
- 伪标签训练模式 — pseudo=train, gold=val
- 三平面训练模式 — shared backbone + MultiPlaneFusion
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

from datasets import Knee25DDataset, TriPlaneDataset, VolumeDataset, PseudoLabelLoader
from models import EfficientNetV2S25D, TriPlaneModel, ResNet3DModel, ConvNeXt25D, Swin25D, DenseNet25D
from losses import FocalBCELoss
from utils import (
    aggregate_to_study,
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
# Tri-Plane 训练 / 验证 循环
# ═══════════════════════════════════════════════════════════════════


def train_one_epoch_triplane(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None,
    grad_clip_norm: float = 1.0,
    grad_accum_steps: int = 1,
    device: str = "cuda",
) -> float:
    """Tri-plane 训练一个 epoch (支持梯度累积).

    Args:
        grad_accum_steps: 梯度累积步数. effective_batch = batch_size × grad_accum_steps.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        sag = batch["sag"].to(device)
        cor = batch["cor"].to(device)
        ax = batch["ax"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(sag, cor, ax)
            loss = criterion(logits, labels) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 累积梯度, 每 grad_accum_steps 步或最后一个 batch 时更新
        is_accum_step = (batch_idx + 1) % grad_accum_steps == 0
        is_last = (batch_idx + 1) == n_batches

        if is_accum_step or is_last:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps  # 恢复原始 loss 用于日志

    return total_loss / n_batches


@torch.no_grad()
def validate_one_epoch_triplane(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> dict:
    """Tri-plane 验证."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    for batch in loader:
        sag = batch["sag"].to(device)
        cor = batch["cor"].to(device)
        ax = batch["ax"].to(device)
        labels = batch["labels"].to(device)

        logits = model(sag, cor, ax)
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
# 3D 训练 / 验证 循环 (Phase 3)
# ═══════════════════════════════════════════════════════════════════


def train_one_epoch_3d(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None,
    grad_clip_norm: float = 1.0,
    grad_accum_steps: int = 1,
    device: str = "cuda",
) -> float:
    """3D volume 训练一个 epoch (支持梯度累积)."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        volume = batch["volume"].to(device)       # [B, 1, D, H, W]
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(volume)
            loss = criterion(logits, labels) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_accum_step = (batch_idx + 1) % grad_accum_steps == 0
        is_last = (batch_idx + 1) == n_batches

        if is_accum_step or is_last:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

    return total_loss / n_batches


@torch.no_grad()
def validate_one_epoch_3d(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> dict:
    """3D volume 验证."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    for batch in loader:
        volume = batch["volume"].to(device)
        labels = batch["labels"].to(device)

        logits = model(volume)
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


# ═══════════════════════════════════════════════════════════════════
# 三平面训练模式 (Phase 2)
# ═══════════════════════════════════════════════════════════════════


def _main_triplane(
    config: dict,
    series_df: pd.DataFrame,
    pseudo_csv: str | None = None,
    gold_csv: str | None = None,
    confidence: str = "HIGH",
) -> dict:
    """三平面训练: Sagittal + Coronal + Axial → fusion → head.

    使用 TriPlaneDataset + TriPlaneModel.
    支持伪标签训练 (pseudo=train, gold=val) 或 gold-only KFold.
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
    pseudo_enabled = (
        config.get("pseudo_label", {}).get("enabled", False)
        and pseudo_csv
        and Path(pseudo_csv).exists()
        and gold_csv
    )

    if pseudo_enabled:
        logger.info("=" * 60)
        logger.info("Tri-Plane 训练模式 — Pseudo-Label + Tri-Plane")
        logger.info(f"  伪标签: {pseudo_csv}")
        logger.info(f"  置信度:  {confidence}")
        logger.info("=" * 60)

        loader = PseudoLabelLoader(pseudo_csv=pseudo_csv)
        stats = loader.stats()
        logger.info(
            "伪标签: %d total, HIGH=%d, HIGH+MEDIUM=%d",
            stats["total"],
            stats["by_confidence_overall"].get("HIGH", 0),
            stats["by_confidence_overall"].get("HIGH_PLUS_MEDIUM", 0),
        )

        train_labels_df, val_labels_df = loader.merge_with_gold(
            gold_csv=gold_csv, confidence=confidence, val_from_gold=True,
        )
    else:
        logger.info("=" * 60)
        logger.info("Tri-Plane 训练模式 — Gold-Only (不含伪标签)")
        logger.info("=" * 60)

        labels_df = pd.read_csv(paths["train_csv"])
        for col in TARGET_COLUMNS:
            if col in labels_df.columns:
                labels_df[col] = pd.to_numeric(
                    labels_df[col], errors="coerce"
                ).fillna(0).astype(int)

        # 只保留有标签的行 (排除纯提交样本)
        has_label = labels_df[TARGET_COLUMNS].notna().any(axis=1)
        labels_df = labels_df[has_label].copy()

        # 找有 Sagittal 的 study (tri-plane 以 Sagittal 为锚定平面)
        sag_studies = set(
            series_df[series_df["Anatomical_Plane"] == "Sagittal"]["StudyInstanceUID"]
        )
        labeled_studies = set(labels_df["StudyInstanceUID"])
        study_ids = sorted(sag_studies & labeled_studies)
        logger.info(
            f"有标签且有 Sagittal 的 studies: {len(study_ids)} "
            f"(全部标签: {len(labeled_studies)}, 有Sag: {len(sag_studies)})"
        )

        np.random.RandomState(config["experiment"]["seed"]).shuffle(study_ids)
        split = int(len(study_ids) * 0.8)
        train_ids = set(study_ids[:split])
        valid_ids = set(study_ids[split:])

        train_labels_df = labels_df[
            labels_df["StudyInstanceUID"].isin(train_ids)
        ].set_index("StudyInstanceUID")
        val_labels_df = labels_df[
            labels_df["StudyInstanceUID"].isin(valid_ids)
        ].set_index("StudyInstanceUID")

    logger.info(f"训练 studies: {len(train_labels_df)}")
    logger.info(f"验证 studies: {len(val_labels_df)}")

    for col in TARGET_COLUMNS:
        t_pos = train_labels_df[col].sum() if col in train_labels_df.columns else 0
        v_pos = val_labels_df[col].sum() if col in val_labels_df.columns else 0
        logger.info(f"  {col:<20s}  train_pos={int(t_pos):5d}  val_pos={int(v_pos):3d}")

    # ── 2. TriPlaneDataset ─────────────────────────────────────────
    ds_kwargs = dict(
        dicom_root=paths["dicom_root"],
        image_size=data_cfg["image_size"],
        slice_count=data_cfg["slice_count"],
        planes=data_cfg["planes"],
    )

    train_ds = TriPlaneDataset(series_df, train_labels_df, is_train=True, **ds_kwargs)
    val_ds = TriPlaneDataset(series_df, val_labels_df, is_train=False, **ds_kwargs)

    # ── 3. DataLoader ──────────────────────────────────────────────
    loader_cfg = data_cfg["loader"]
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=loader_cfg["train_workers"], pin_memory=loader_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=loader_cfg["valid_workers"], pin_memory=loader_cfg["pin_memory"],
    )

    # ── 4. TriPlaneModel ───────────────────────────────────────────
    model = TriPlaneModel(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        feature_dim=model_cfg["feature_dim"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
        shared_backbone=model_cfg.get("shared_backbone", True),
        fusion=model_cfg.get("fusion", "concat"),
        fusion_heads=model_cfg.get("fusion_heads", 8),
        fusion_layers=model_cfg.get("fusion_layers", 2),
        use_slice_attention=model_cfg.get("use_slice_attention", False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"模型参数量: {n_params:.1f}M total, {n_trainable:.1f}M trainable")

    # ── 5. 损失 & 优化器 ───────────────────────────────────────────
    criterion = FocalBCELoss(gamma=loss_cfg["gamma"], alpha=loss_cfg["alpha"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"], betas=opt_cfg["betas"],
    )
    scaler = torch.amp.GradScaler("cuda") if train_cfg["mixed_precision"] else None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=sched_cfg.get("T_0", 10),
        T_mult=sched_cfg.get("T_mult", 2), eta_min=sched_cfg.get("eta_min", 1e-6),
    )

    # ── 6. 训练循环 ────────────────────────────────────────────────
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg["early_stopping"]["patience"]
    train_start = time.time()

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch_triplane(
            model, train_loader, optimizer, criterion, scaler,
            grad_clip_norm=train_cfg["gradient_clip_norm"],
            grad_accum_steps=train_cfg.get("gradient_accumulation_steps", 1),
            device=device,
        )
        val_metrics = validate_one_epoch_triplane(
            model, val_loader, criterion, device=device,
        )

        elapsed = time.time() - t0
        epochs_done = epoch + 1
        avg_sec = (time.time() - train_start) / epochs_done
        remaining = avg_sec * (train_cfg["epochs"] - epochs_done)
        eta_str = f"{remaining/60:.0f}min" if remaining < 3600 else f"{remaining/3600:.1f}h"

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

        per_class = val_metrics.get("per_class_auc", {})
        if per_class:
            logger.info(f"        per-class → {format_per_class_auc(per_class)}")

        scheduler.step()

        if val_metrics["macro_auc"] > best_auc + train_cfg["early_stopping"]["min_delta"]:
            best_auc = val_metrics["macro_auc"]
            patience_counter = 0
            ckpt_path = Path(paths["checkpoint_dir"]) / "best_model.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "auc": best_auc},
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
        "mode": "triplane",
        "fusion": model_cfg.get("fusion", "concat"),
        "n_train_studies": len(train_labels_df),
        "n_val_studies": len(val_labels_df),
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "best_auc": best_auc,
        "total_time_hours": round(total_time / 3600, 2),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"摘要已保存: {out_dir / 'summary.json'}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Tri-Plane 训练完成! Best AUC: {best_auc:.4f} | 耗时: {total_time/3600:.1f}h")
    logger.info(f"{'='*60}")

    return {"best_auc": best_auc, "summary": summary}


# ═══════════════════════════════════════════════════════════════════
# 3D 训练模式 (Phase 3)
# ═══════════════════════════════════════════════════════════════════


def _main_3d(
    config: dict,
    series_df: pd.DataFrame,
    pseudo_csv: str | None = None,
    gold_csv: str | None = None,
    confidence: str = "HIGH",
) -> dict:
    """3D volumetric 训练: ResNet3D-18 在 128×128×32 volume 上.

    使用 pseudo-label 训练, gold-label 验证.
    Gradient checkpointing + AMP 适配 8GB VRAM.
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
    pseudo_enabled = (
        config.get("pseudo_label", {}).get("enabled", False)
        and pseudo_csv
        and Path(pseudo_csv).exists()
        and gold_csv
    )

    if pseudo_enabled:
        logger.info("=" * 60)
        logger.info("3D 训练模式 — Pseudo-Label + ResNet3D-18")
        logger.info(f"  伪标签: {pseudo_csv}")
        logger.info(f"  置信度:  {confidence}")
        logger.info("=" * 60)

        loader = PseudoLabelLoader(pseudo_csv=pseudo_csv)
        stats = loader.stats()
        logger.info(
            "伪标签: %d total, HIGH=%d",
            stats["total"],
            stats["by_confidence_overall"].get("HIGH", 0),
        )
        train_labels_df, val_labels_df = loader.merge_with_gold(
            gold_csv=gold_csv, confidence=confidence, val_from_gold=True,
        )
    else:
        logger.info("=" * 60)
        logger.info("3D 训练模式 — Gold-Only")
        logger.info("=" * 60)
        labels_df = pd.read_csv(paths["train_csv"])
        for col in TARGET_COLUMNS:
            if col in labels_df.columns:
                labels_df[col] = pd.to_numeric(labels_df[col], errors="coerce").fillna(0).astype(int)
        has_label = labels_df[TARGET_COLUMNS].notna().any(axis=1)
        labels_df = labels_df[has_label].copy()
        sag_studies = set(series_df[series_df["Anatomical_Plane"] == "Sagittal"]["StudyInstanceUID"])
        study_ids = sorted(sag_studies & set(labels_df["StudyInstanceUID"]))
        np.random.RandomState(config["experiment"]["seed"]).shuffle(study_ids)
        split = int(len(study_ids) * 0.8)
        train_labels_df = labels_df[labels_df["StudyInstanceUID"].isin(set(study_ids[:split]))].set_index("StudyInstanceUID")
        val_labels_df = labels_df[labels_df["StudyInstanceUID"].isin(set(study_ids[split:]))].set_index("StudyInstanceUID")

    logger.info(f"训练 studies: {len(train_labels_df)}")
    logger.info(f"验证 studies: {len(val_labels_df)}")

    # ── 2. VolumeDataset ───────────────────────────────────────────
    train_ds = VolumeDataset(
        series_df=series_df,
        labels_df=train_labels_df,
        dicom_root=paths["dicom_root"],
        volume_depth=data_cfg["volume_depth"],
        volume_size=data_cfg["volume_size"],
        plane=data_cfg.get("plane", "Sagittal"),
        is_train=True,
    )
    val_ds = VolumeDataset(
        series_df=series_df,
        labels_df=val_labels_df,
        dicom_root=paths["dicom_root"],
        volume_depth=data_cfg["volume_depth"],
        volume_size=data_cfg["volume_size"],
        plane=data_cfg.get("plane", "Sagittal"),
        is_train=False,
    )

    # ── 3. DataLoader ──────────────────────────────────────────────
    loader_cfg = data_cfg["loader"]
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=loader_cfg["train_workers"], pin_memory=loader_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=loader_cfg["valid_workers"], pin_memory=loader_cfg["pin_memory"],
    )

    # ── 4. ResNet3DModel ───────────────────────────────────────────
    model = ResNet3DModel(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
        feature_dim=model_cfg["feature_dim"],
        use_grad_checkpoint=model_cfg.get("use_grad_checkpoint", True),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"模型参数量: {n_params:.1f}M total, {n_trainable:.1f}M trainable")

    # ── 5. 损失 & 优化器 ───────────────────────────────────────────
    criterion = FocalBCELoss(gamma=loss_cfg["gamma"], alpha=loss_cfg["alpha"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"], betas=opt_cfg["betas"],
    )
    scaler = torch.amp.GradScaler("cuda") if train_cfg["mixed_precision"] else None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=sched_cfg.get("T_0", 10),
        T_mult=sched_cfg.get("T_mult", 2), eta_min=sched_cfg.get("eta_min", 1e-6),
    )

    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

    # ── 6. 训练循环 ────────────────────────────────────────────────
    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg["early_stopping"]["patience"]
    train_start = time.time()

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch_3d(
            model, train_loader, optimizer, criterion, scaler,
            grad_clip_norm=train_cfg["gradient_clip_norm"],
            grad_accum_steps=accum_steps, device=device,
        )
        val_metrics = validate_one_epoch_3d(model, val_loader, criterion, device=device)

        elapsed = time.time() - t0
        epochs_done = epoch + 1
        avg_sec = (time.time() - train_start) / epochs_done
        remaining = avg_sec * (train_cfg["epochs"] - epochs_done)
        eta_str = f"{remaining/60:.0f}min" if remaining < 3600 else f"{remaining/3600:.1f}h"

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

        per_class = val_metrics.get("per_class_auc", {})
        if per_class:
            logger.info(f"        per-class → {format_per_class_auc(per_class)}")

        scheduler.step()

        if val_metrics["macro_auc"] > best_auc + train_cfg["early_stopping"]["min_delta"]:
            best_auc = val_metrics["macro_auc"]
            patience_counter = 0
            ckpt_path = Path(paths["checkpoint_dir"]) / "best_model.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "auc": best_auc},
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
        "mode": "3d",
        "volume_depth": data_cfg["volume_depth"],
        "volume_size": data_cfg["volume_size"],
        "n_train_studies": len(train_labels_df),
        "n_val_studies": len(val_labels_df),
        "best_auc": best_auc,
        "total_time_hours": round(total_time / 3600, 2),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"摘要已保存: {out_dir / 'summary.json'}")

    logger.info(f"\n{'='*60}")
    logger.info(f"3D 训练完成! Best AUC: {best_auc:.4f} | 耗时: {total_time/3600:.1f}h")
    logger.info(f"{'='*60}")

    return {"best_auc": best_auc, "summary": summary}


# ═══════════════════════════════════════════════════════════════════
# Ensemble Backbone 训练 (Phase 4)
# ═══════════════════════════════════════════════════════════════════


def _main_ensemble_backbone(
    config: dict,
    series_df: pd.DataFrame,
    pseudo_csv: str | None = None,
    gold_csv: str | None = None,
    confidence: str = "HIGH",
) -> dict:
    """Phase 4: 训练单个 backbone (用于后续集成).

    根据 config.model.arch 选择 backbone:
      - "efficientnetv2_s" → EfficientNetV2S25D
      - "convnext_small"   → ConvNeXt25D
      - "swin_tiny"        → Swin25D

    所有 backbone 使用相同的训练管线 (单平面 Sagittal).
    """
    paths = config["paths"]
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    opt_cfg = config["optimizer"]
    sched_cfg = config.get("scheduler", {})

    exp_name = config["experiment"]["name"]
    arch = model_cfg["arch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. 加载标签 ───────────────────────────────────────────────
    pseudo_enabled = (
        config.get("pseudo_label", {}).get("enabled", False)
        and pseudo_csv and Path(pseudo_csv).exists() and gold_csv
    )

    if pseudo_enabled:
        logger.info("=" * 60)
        logger.info(f"Ensemble Backbone 训练 — {arch} + Pseudo-Label")
        logger.info(f"  伪标签: {pseudo_csv}, 置信度: {confidence}")
        logger.info("=" * 60)
        loader = PseudoLabelLoader(pseudo_csv=pseudo_csv)
        stats = loader.stats()
        logger.info("伪标签: %d total, HIGH=%d", stats["total"], stats["by_confidence_overall"].get("HIGH", 0))
        train_labels_df, val_labels_df = loader.merge_with_gold(
            gold_csv=gold_csv, confidence=confidence, val_from_gold=True,
        )
    else:
        logger.info(f"Ensemble Backbone 训练 — {arch} (Gold-Only)")
        labels_df = pd.read_csv(paths["train_csv"])
        for col in TARGET_COLUMNS:
            if col in labels_df.columns:
                labels_df[col] = pd.to_numeric(labels_df[col], errors="coerce").fillna(0).astype(int)
        has_label = labels_df[TARGET_COLUMNS].notna().any(axis=1)
        labels_df = labels_df.loc[has_label]
        sag_studies = set(series_df[series_df["Anatomical_Plane"] == "Sagittal"]["StudyInstanceUID"])
        study_ids = sorted(sag_studies & set(labels_df["StudyInstanceUID"]))
        np.random.RandomState(config["experiment"]["seed"]).shuffle(study_ids)
        split = int(len(study_ids) * 0.8)
        train_labels_df = labels_df[labels_df["StudyInstanceUID"].isin(set(study_ids[:split]))].set_index("StudyInstanceUID")
        val_labels_df = labels_df[labels_df["StudyInstanceUID"].isin(set(study_ids[split:]))].set_index("StudyInstanceUID")

    logger.info(f"训练: {len(train_labels_df)}, 验证: {len(val_labels_df)}")

    # ── 2. Dataset (单平面) ────────────────────────────────────────
    ds_kwargs = dict(
        dicom_root=paths["dicom_root"],
        planes=data_cfg["planes"],
        image_size=data_cfg["image_size"],
        slice_count=data_cfg["slice_count"],
        fluid_sensitive_only=data_cfg.get("fluid_sensitive_only", False),
        fat_suppression_only=data_cfg.get("fat_suppression_only", False),
    )
    train_ds = Knee25DDataset(series_df, train_labels_df, is_train=True, **ds_kwargs)
    val_ds = Knee25DDataset(series_df, val_labels_df, is_train=False, **ds_kwargs)

    loader_cfg = data_cfg["loader"]
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=loader_cfg["train_workers"], pin_memory=loader_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=loader_cfg["valid_workers"], pin_memory=loader_cfg["pin_memory"],
    )

    # ── 3. 模型 ────────────────────────────────────────────────────
    if arch == "efficientnetv2_s":
        model = EfficientNetV2S25D(
            in_channels=model_cfg["in_channels"], num_classes=model_cfg["num_classes"],
            pretrained=model_cfg["pretrained"], dropout=model_cfg["dropout"],
        ).to(device)
    elif arch == "convnext_small":
        model = ConvNeXt25D(
            in_channels=model_cfg["in_channels"], num_classes=model_cfg["num_classes"],
            pretrained=model_cfg["pretrained"], dropout=model_cfg["dropout"],
        ).to(device)
    elif arch == "swin_tiny":
        model = Swin25D(
            in_channels=model_cfg["in_channels"], num_classes=model_cfg["num_classes"],
            pretrained=model_cfg["pretrained"], dropout=model_cfg["dropout"],
        ).to(device)
    elif arch == "densenet121":
        model = DenseNet25D(
            in_channels=model_cfg["in_channels"], num_classes=model_cfg["num_classes"],
            pretrained=model_cfg["pretrained"], dropout=model_cfg["dropout"],
        ).to(device)
    else:
        raise ValueError(f"Unknown ensemble backbone arch: {arch}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"{arch}: {n_params:.1f}M total, {n_trainable:.1f}M trainable")

    # ── 4. 损失 & 优化器 ───────────────────────────────────────────
    criterion = FocalBCELoss(gamma=loss_cfg["gamma"], alpha=loss_cfg["alpha"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"], betas=opt_cfg["betas"],
    )
    scaler = torch.amp.GradScaler("cuda") if train_cfg["mixed_precision"] else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=sched_cfg.get("T_0", 10),
        T_mult=sched_cfg.get("T_mult", 2), eta_min=sched_cfg.get("eta_min", 1e-6),
    )

    # ── 5. 训练循环 ────────────────────────────────────────────────
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

        epochs_done = epoch + 1
        avg_sec = (time.time() - train_start) / epochs_done
        remaining = avg_sec * (train_cfg["epochs"] - epochs_done)
        eta_str = f"{remaining/60:.0f}min" if remaining < 3600 else f"{remaining/3600:.1f}h"

        gpu_alloc = torch.cuda.max_memory_allocated(device) / 1024**3 if device == "cuda" else 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[{arch}] Epoch {epoch:3d}: "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_auc={val_metrics['macro_auc']:.4f}  lr={lr_now:.2e}  "
            f"⏱ {elapsed:.0f}s  ETA {eta_str}  VRAM={gpu_alloc:.1f}GB"
        )
        per_class = val_metrics.get("per_class_auc", {})
        if per_class:
            logger.info(f"        per-class → {format_per_class_auc(per_class)}")

        scheduler.step()

        if val_metrics["macro_auc"] > best_auc + train_cfg["early_stopping"]["min_delta"]:
            best_auc = val_metrics["macro_auc"]
            patience_counter = 0
            ckpt_path = Path(paths["checkpoint_dir"]) / f"{arch}_best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "auc": best_auc, "arch": arch},
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[{arch}] Early stopping at epoch {epoch}")
                break

    total_time = time.time() - train_start
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": exp_name, "mode": "ensemble_backbone", "arch": arch,
        "n_train_studies": len(train_labels_df), "n_val_studies": len(val_labels_df),
        "best_auc": best_auc, "total_time_hours": round(total_time / 3600, 2),
    }
    with open(out_dir / f"summary_{arch}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"[{arch}] 训练完成! Best AUC: {best_auc:.4f} | 耗时: {total_time/3600:.1f}h")
    logger.info(f"{'='*60}")
    return {"best_auc": best_auc, "summary": summary}


# ═══════════════════════════════════════════════════════════════════
# Ensemble 推理 (Phase 4) — 加载多个 checkpoint 加权平均
# ═══════════════════════════════════════════════════════════════════


def run_ensemble_inference(
    config: dict,
    ensemble_cfg: dict,
    device: str = "cuda",
) -> "pd.DataFrame":
    """Phase 4 集成推理: 加载多个模型 checkpoint, 加权平均生成 submission.

    Args:
        config: 完整配置 dict
        ensemble_cfg: config["ensemble"] section — 列出模型/checkpoint/weight
        device: 推理设备

    Returns:
        submission_df: 加权平均后的 submission DataFrame
    """
    from models import EnsembleInference, ensemble_submissions

    paths = config["paths"]
    data_cfg = config["data"]
    inference_cfg = config.get("inference", {})

    logger.info("=" * 60)
    logger.info("Phase 4 — Ensemble 集成推理")
    logger.info(f"  模型数: {len(ensemble_cfg['models'])}")
    logger.info("=" * 60)

    checkpoint_paths = []
    weights = []

    for m in ensemble_cfg["models"]:
        checkpoint_paths.append(m["checkpoint"])
        weights.append(m["weight"])
        logger.info(f"  {m['name']:<25s} weight={m['weight']:.2f}  ckpt={m['checkpoint']}")

    # ── 如果启用了 submission 合并模式 ──────────────────────────
    if ensemble_cfg.get("submission_merge", {}).get("enabled", False):
        # 查找已生成的 submission CSV
        submission_paths = []
        for m in ensemble_cfg["models"]:
            # 推断 submission 路径
            sub_path = Path(paths["output_dir"]).parent / m["name"] / f"submission_best_model.csv"
            if sub_path.exists():
                submission_paths.append(sub_path)
            else:
                logger.warning(f"Submission 不存在: {sub_path}")

        if len(submission_paths) >= 2:
            logger.info(f"合并 {len(submission_paths)} 个 submission CSV...")
            output_path = Path(paths["output_dir"]) / "submission_ensemble.csv"
            return ensemble_submissions(
                submission_paths, weights=weights[:len(submission_paths)],
                output_path=output_path,
            )

    logger.warning(
        "Ensemble 推理需要各模型已生成 submission CSV.\n"
        "  先运行各 backbone 的独立推理:\n"
        "  python train.py --config configs/efficientnet.yaml --inference <ckpt>\n"
        "  python train.py --config configs/phase4_ensemble.yaml --inference <ckpt>  (×2, 改 arch)\n"
        "  ...\n"
        "  再用本函数合并: ensemble_submissions([sub1.csv, sub2.csv, ...], weights=[...])"
    )
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# 三平面推理 (Phase 2)
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def run_inference_triplane(
    config: dict,
    checkpoint_path: str | Path,
    test_series_csv: str | Path = "data/metadata/test_series.csv",
    submission_csv: str | Path | None = None,
    batch_size: int = 8,
    device: str = "cuda",
) -> pd.DataFrame:
    """Tri-plane 推理: 加载模型, 在测试集上逐 slice 推理, 聚合至 study 级.

    Args:
        config: 配置 dict
        checkpoint_path: 训练好的模型权重 .pt 文件
        test_series_csv: 测试集 series 元数据
        submission_csv: 输出 submission CSV 路径 (None → 自动生成)
        batch_size: 推理 batch size (比训练时大, 因为不用存梯度)
        device: 推理设备

    Returns:
        submission_df: StudyInstanceUID × 12 概率的 DataFrame
    """
    paths = config["paths"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    checkpoint_path = Path(checkpoint_path)
    test_series_csv = Path(test_series_csv)

    logger.info("=" * 60)
    logger.info("Tri-Plane 推理模式")
    logger.info(f"  模型: {checkpoint_path}")
    logger.info(f"  数据: {test_series_csv}")
    logger.info("=" * 60)

    # ── 1. 加载测试集元数据 ────────────────────────────────────
    test_series_df = pd.read_csv(test_series_csv)
    test_study_uids = sorted(test_series_df["StudyInstanceUID"].unique())
    logger.info(f"测试 studies: {len(test_study_uids):,}")

    # 构建 dummy labels (TriPlaneDataset 需要 label_map, 推理时全填 0)
    dummy_labels = pd.DataFrame({
        "StudyInstanceUID": test_study_uids,
        **{col: 0 for col in TARGET_COLUMNS},
    }).set_index("StudyInstanceUID")

    # ── 2. 构建 TriPlaneDataset ─────────────────────────────────
    test_ds = TriPlaneDataset(
        series_df=test_series_df,
        labels_df=dummy_labels,
        dicom_root=paths.get("test_dicom_root", paths.get("dicom_root", "dataset/test_series")),
        image_size=data_cfg["image_size"],
        slice_count=data_cfg["slice_count"],
        planes=data_cfg["planes"],
        is_train=False,
    )
    logger.info(f"测试样本数 (切片级): {len(test_ds):,}")

    # ── 3. DataLoader ───────────────────────────────────────────
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg["loader"]["valid_workers"],
        pin_memory=data_cfg["loader"]["pin_memory"],
    )

    # ── 4. 构建 & 加载模型 ──────────────────────────────────────
    model = TriPlaneModel(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        feature_dim=model_cfg["feature_dim"],
        pretrained=False,  # 推理时不需要 pretrained
        dropout=model_cfg["dropout"],
        shared_backbone=model_cfg.get("shared_backbone", True),
        fusion=model_cfg.get("fusion", "concat"),
        fusion_heads=model_cfg.get("fusion_heads", 8),
        fusion_layers=model_cfg.get("fusion_layers", 2),
        use_slice_attention=model_cfg.get("use_slice_attention", False),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        f"模型加载完成: {n_params:.1f}M params, "
        f"epoch={ckpt.get('epoch', '?')}, auc={ckpt.get('auc', '?'):.4f}"
    )

    # ── 5. 逐 slice 推理 ────────────────────────────────────────
    all_logits = []
    all_study_uids = []

    inference_cfg = config.get("inference", {})
    tta_enabled = inference_cfg.get("tta", False)

    logger.info(f"推理中... (batch_size={batch_size}, TTA={tta_enabled})")
    t0 = time.time()

    for batch in test_loader:
        sag = batch["sag"].to(device)
        cor = batch["cor"].to(device)
        ax = batch["ax"].to(device)

        logits = model(sag, cor, ax)  # [B, 12]

        # Optional: TTA (horizontal flip)
        if tta_enabled:
            sag_flip = torch.flip(sag, dims=[-1])
            cor_flip = torch.flip(cor, dims=[-1])
            ax_flip = torch.flip(ax, dims=[-1])
            logits_flip = model(sag_flip, cor_flip, ax_flip)
            logits = (logits + logits_flip) / 2.0

        all_logits.append(logits.cpu().numpy())
        all_study_uids.extend(batch["study_uid"])

    elapsed = time.time() - t0
    logger.info(f"推理完成: {len(all_study_uids):,} slices, {elapsed:.0f}s "
                 f"({len(all_study_uids)/elapsed:.0f} slices/s)")

    # ── 6. Slice → Study 聚合 ───────────────────────────────────
    slice_logits = np.concatenate(all_logits)                 # [N_slices, 12]
    slice_sids = np.array(all_study_uids)

    topk_frac = inference_cfg.get("topk_fraction", 0.25)
    study_logits, study_ids = aggregate_to_study(
        slice_logits, slice_sids, topk_fraction=topk_frac,
    )
    study_probs = 1.0 / (1.0 + np.exp(-study_logits))       # sigmoid

    logger.info(
        f"聚合: {len(slice_sids):,} slices → {len(study_ids):,} studies "
        f"(top-{topk_frac:.0%} mean)"
    )

    # ── 7. 构建 Submission DataFrame ────────────────────────────
    submission_df = pd.DataFrame(
        study_probs,
        index=study_ids,
        columns=TARGET_COLUMNS,
    )
    submission_df.index.name = "StudyInstanceUID"

    # 补齐缺失的 test study (给 0.5)
    missing = set(test_study_uids) - set(study_ids)
    if missing:
        logger.warning(
            f"{len(missing)} studies 无推理结果 (DICOM缺失), 填充 0.5"
        )
        for sid in missing:
            submission_df.loc[sid] = 0.5

    submission_df = submission_df.loc[
        [s for s in test_study_uids if s in submission_df.index]
    ]

    # ── 8. 保存 ─────────────────────────────────────────────────
    if submission_csv is None:
        ckpt_stem = checkpoint_path.stem
        submission_csv = (
            Path(paths["output_dir"]) / f"submission_{ckpt_stem}.csv"
        )
    submission_csv = Path(submission_csv)
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_csv)
    logger.info(f"Submission 已保存: {submission_csv} ({len(submission_df)} studies)")

    return submission_df


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
    parser.add_argument(
        "--triplane", action="store_true",
        help="启用三平面训练模式 (否则自动从 config model.arch 检测)",
    )
    parser.add_argument(
        "--inference", type=str, default=None,
        help="推理模式: 指定模型 checkpoint 路径",
    )
    parser.add_argument(
        "--test_series", type=str, default="data/metadata/test_series.csv",
        help="测试集 series 元数据 CSV",
    )
    parser.add_argument(
        "--submission", type=str, default=None,
        help="输出 submission CSV 路径 (默认 outputs/<exp>/submission_<ckpt>.csv)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="推理 batch size (覆盖配置文件)",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Phase 4 ensemble 集成推理模式 (合并多个模型 prediction)",
    )
    args = parser.parse_args()

    # ── 加载配置 ───────────────────────────────────────────────
    config_path = Path(args.config)
    with open(config_path, encoding="utf-8") as f:
        config_check = yaml.safe_load(f)

    stage = config_check.get("experiment", {}).get("stage", "phase1")
    model_arch = config_check.get("model", {}).get("arch", "")
    paths = config_check["paths"]

    if args.epochs is not None:
        config_check["train"]["epochs"] = args.epochs

    # ── Phase 4 Ensemble 推理 ──────────────────────────────────
    if args.ensemble:
        ensemble_cfg = config_check.get("ensemble", {})
        run_ensemble_inference(config_check, ensemble_cfg)

    # ── Phase 3: 3D 训练 ───────────────────────────────────────
    elif stage == "phase3" or args.inference and model_arch == "resnet3d":
        series_df = pd.read_csv(paths.get("series_csv", "data/metadata/train_series.csv"))
        pseudo_csv = args.pseudo_csv or config_check.get("pseudo_label", {}).get("pseudo_csv")
        gold_csv = paths.get("train_csv", "data/metadata/train.csv")
        confidence = args.confidence or config_check.get("pseudo_label", {}).get("confidence", "HIGH")

        _main_3d(
            config_check, series_df,
            pseudo_csv=pseudo_csv,
            gold_csv=gold_csv,
            confidence=confidence,
        )

    # ── Inference: Tri-plane 推理 ──────────────────────────────
    elif args.inference is not None and model_arch != "resnet3d":
        run_inference_triplane(
            config_check,
            checkpoint_path=args.inference,
            test_series_csv=args.test_series,
            submission_csv=args.submission,
            batch_size=args.batch_size or config_check["train"]["batch_size"] * 2,
        )

    # ── Phase 4: Ensemble backbone 训练 ────────────────────────
    elif stage == "phase4":
        series_df = pd.read_csv(paths.get("series_csv", "data/metadata/train_series.csv"))
        pseudo_csv = args.pseudo_csv or config_check.get("pseudo_label", {}).get("pseudo_csv")
        gold_csv = paths.get("train_csv", "data/metadata/train.csv")
        confidence = args.confidence or config_check.get("pseudo_label", {}).get("confidence", "HIGH")

        _main_ensemble_backbone(
            config_check, series_df,
            pseudo_csv=pseudo_csv,
            gold_csv=gold_csv,
            confidence=confidence,
        )

    # ── Phase 2: Tri-plane 训练 ────────────────────────────────
    elif (
        args.triplane
        or model_arch.startswith("triplane")
        or stage == "phase2"
    ):
        series_df = pd.read_csv(paths["series_csv"])
        pseudo_csv = args.pseudo_csv or config_check.get("pseudo_label", {}).get("pseudo_csv")
        gold_csv = paths["train_csv"]
        confidence = args.confidence or config_check.get("pseudo_label", {}).get("confidence", "HIGH")

        _main_triplane(
            config_check, series_df,
            pseudo_csv=pseudo_csv,
            gold_csv=gold_csv,
            confidence=confidence,
        )

    # ── Phase 1: 默认 single-plane 训练 ────────────────────────
    else:
        main(
            args.config,
            epochs_override=args.epochs,
            pseudo_csv=args.pseudo_csv,
            confidence=args.confidence,
        )
