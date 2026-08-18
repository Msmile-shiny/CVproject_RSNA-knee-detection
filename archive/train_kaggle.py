"""Kaggle Training Script — DINOv2 + CNN Refiner for RSNA 2026 Knee MRI.

Usage on Kaggle:
    # First upload pseudo_labels.csv as a Kaggle Dataset named "rsna-knee-pseudo-labels"
    # Then run this notebook/script.

    !pip install timm pydicom opencv-python
    !python train_kaggle.py

Or import and run:
    from train_kaggle import main
    main()
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

# --- Paths (Kaggle defaults) ---
KAGGLE_INPUT = Path("/kaggle/input/rsna-2026-knee-abnormality-detection")
PSEUDO_INPUT = Path("/kaggle/input/rsna-knee-pseudo-labels")  # Upload pseudo_labels.csv here
KAGGLE_OUTPUT = Path("/kaggle/working")

TRAIN_CSV = KAGGLE_INPUT / "train.csv"
SERIES_CSV = KAGGLE_INPUT / "train_series.csv"
DICOM_ROOT = KAGGLE_INPUT / "train_series"
PSEUDO_CSV = PSEUDO_INPUT / "pseudo_labels.csv"

# --- Training Config ---
TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

CONFIG = {
    # Data
    "image_size": 384,
    "slice_count": 5,
    "confidence_filter": "HIGH",  # "HIGH" | "HIGH_PLUS_MEDIUM" | "ALL"
    # Model
    "dinov2_variant": "vit_small_patch14_dinov2.lvd142m",
    "cls_dim": 384,
    "num_classes": 12,
    # Training
    "batch_size": 16,          # per GPU
    "gradient_accumulation": 1,
    "epochs": 50,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "lr_warmup_epochs": 3,
    "lr_t0": 10,               # CosineAnnealingWarmRestarts T_0
    "lr_t_mult": 2,
    "lr_eta_min": 1e-6,
    # Loss
    "focal_gamma": 2.0,
    "focal_alpha": 0.25,
    # Regularization
    "dropout": 0.1,            # fusion/slice-transformer
    "head_dropout": 0.3,       # classification head
    "gradient_clip_norm": 1.0,
    # Early stopping
    "early_stop_patience": 10,
    "early_stop_min_delta": 0.001,
    # System
    "num_workers": 4,
    "mixed_precision": True,
    "log_interval": 20,        # batches between loss logs
    "checkpoint_interval": 5,  # epochs between checkpoints
}

# ═══════════════════════════════════════════════════════════════
# DDP Setup
# ═══════════════════════════════════════════════════════════════


def setup_ddp() -> tuple[int, int, bool]:
    """Initialize distributed training. Returns (rank, world_size, is_main)."""
    if "LOCAL_RANK" in os.environ:
        # Launched via torchrun or similar
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return local_rank, world_size, (local_rank == 0)
    else:
        # Single GPU or CPU
        if torch.cuda.is_available():
            return 0, 1, True
        else:
            return -1, 1, True


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════


def load_and_merge_labels(rank: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv, pseudo_labels.csv, split into train/val.

    Returns:
        train_labels: StudyInstanceUID-indexed DataFrame with 12 label columns
        val_labels:   StudyInstanceUID-indexed DataFrame with 12 label columns (58 gold)
    """
    if rank == 0:
        print("Loading labels...")

    # Load competition metadata
    train_meta = pd.read_csv(TRAIN_CSV)
    series_meta = pd.read_csv(SERIES_CSV)

    # Identify gold-labeled vs unlabeled studies
    label_cols_present = [c for c in TARGET_COLUMNS if c in train_meta.columns]
    has_gold = train_meta[label_cols_present].notna().all(axis=1)
    gold_studies = train_meta[has_gold].copy()
    unlabeled_studies = train_meta[~has_gold].copy()

    if rank == 0:
        print(f"  Gold-labeled studies:    {len(gold_studies)}")
        print(f"  Unlabeled studies:       {len(unlabeled_studies)}")

    # Load pseudo-labels
    pseudo_df = pd.read_csv(PSEUDO_CSV)

    # Map pred columns to standard names
    pred_cols = {f"pred_{c}": c for c in TARGET_COLUMNS}
    conf_cols = {f"conf_{c}": f"confidence_{c}" for c in TARGET_COLUMNS}

    # Filter by confidence level
    if CONFIG["confidence_filter"] == "HIGH":
        # Only keep studies where ALL 12 classes are HIGH confidence
        conf_mask = pd.Series(True, index=pseudo_df.index)
        for c in TARGET_COLUMNS:
            conf_mask &= (pseudo_df[f"conf_{c}"] == "HIGH")
        pseudo_df = pseudo_df[conf_mask].copy()
    elif CONFIG["confidence_filter"] == "HIGH_PLUS_MEDIUM":
        conf_mask = pd.Series(True, index=pseudo_df.index)
        for c in TARGET_COLUMNS:
            conf_mask &= (pseudo_df[f"conf_{c}"].isin(["HIGH", "MEDIUM"]))
        pseudo_df = pseudo_df[conf_mask].copy()

    # Build training labels: pseudo-labeled studies
    train_labels = pseudo_df[["StudyInstanceUID"]].copy()
    for c in TARGET_COLUMNS:
        train_labels[c] = pseudo_df[f"pred_{c}"]

    train_labels = train_labels.set_index("StudyInstanceUID")
    train_labels = train_labels.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    # Build validation labels: 58 gold studies
    val_labels = gold_studies[["StudyInstanceUID"] + label_cols_present].copy()
    val_labels = val_labels.set_index("StudyInstanceUID")
    for c in TARGET_COLUMNS:
        if c not in val_labels.columns:
            val_labels[c] = 0.0
    val_labels = val_labels.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    if rank == 0:
        print(f"  Training studies (pseudo): {len(train_labels)}")
        print(f"  Validation studies (gold): {len(val_labels)}")
        # Print class balance
        for c in TARGET_COLUMNS:
            train_pos = (train_labels[c] == 1).sum()
            val_pos = (val_labels[c] == 1).sum()
            print(f"    {c:<20s}  train_pos={train_pos:5d}  val_pos={val_pos:3d}")

    return train_labels, val_labels, series_meta


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════


def compute_auc(targets: np.ndarray, probs: np.ndarray) -> float:
    """Compute macro-averaged AUC ROC across all classes."""
    from sklearn.metrics import roc_auc_score

    aucs = []
    for c in range(targets.shape[1]):
        if targets[:, c].sum() == 0 or (targets[:, c] == 1).all():
            continue  # Skip classes with no positive or all positive
        try:
            aucs.append(roc_auc_score(targets[:, c], probs[:, c]))
        except Exception:
            continue

    return float(np.mean(aucs)) if aucs else 0.0


def compute_per_class_auc(targets: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute per-class AUC ROC."""
    from sklearn.metrics import roc_auc_score

    aucs = {}
    for i, c in enumerate(TARGET_COLUMNS):
        if targets[:, i].sum() == 0 or (targets[:, i] == 1).all():
            aucs[c] = float("nan")
            continue
        try:
            aucs[c] = roc_auc_score(targets[:, i], probs[:, i])
        except Exception:
            aucs[c] = float("nan")
    return aucs


# ═══════════════════════════════════════════════════════════════
# Training & Validation Loops
# ═══════════════════════════════════════════════════════════════


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    rank: int,
    device: torch.device,
) -> float:
    """Train one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    accum_steps = CONFIG["gradient_accumulation"]
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels) / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_norm"])
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

        if rank == 0 and batch_idx % CONFIG["log_interval"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader):4d} | "
                  f"loss={loss.item() * accum_steps:.4f} | lr={lr:.2e}",
                  flush=True)

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank: int = 0,
) -> dict:
    """Validate on gold labels. Returns loss, macro AUC, per-class AUC."""
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(images)
        total_loss += criterion(logits, labels).item()

        all_logits.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    targets_np = np.concatenate(all_labels)

    macro_auc = compute_auc(targets_np, logits_np)
    per_class = compute_per_class_auc(targets_np, logits_np)

    return {
        "loss": total_loss / len(loader),
        "macro_auc": macro_auc,
        "per_class_auc": per_class,
    }


# ═══════════════════════════════════════════════════════════════
# Main Training Entry Point
# ═══════════════════════════════════════════════════════════════


def main():
    # ── Setup ─────────────────────────────────────────────
    rank, world_size, is_main = setup_ddp()
    device = torch.device(f"cuda:{rank}" if rank >= 0 else "cpu")

    if is_main:
        print("=" * 60)
        print("DINOv2 + CNN Refiner — RSNA 2026 Knee MRI")
        print(f"GPUs: {world_size} | Device: {device}")
        print(f"Config: {CONFIG}")
        print("=" * 60)

    # ── Data ──────────────────────────────────────────────
    train_labels_df, val_labels_df, series_meta = load_and_merge_labels(rank)

    # Import dataset class (local module or inline)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datasets import Knee25DPseudoDataset

    train_ds = Knee25DPseudoDataset(
        series_df=series_meta,
        labels_df=train_labels_df,
        dicom_root=DICOM_ROOT,
        image_size=CONFIG["image_size"],
        slice_count=CONFIG["slice_count"],
        is_train=True,
    )

    val_ds = Knee25DPseudoDataset(
        series_df=series_meta,
        labels_df=val_labels_df,
        dicom_root=DICOM_ROOT,
        image_size=CONFIG["image_size"],
        slice_count=CONFIG["slice_count"],
        is_train=False,
    )

    if is_main:
        print(f"\nTrain samples: {len(train_ds):,}")
        print(f"Val samples:   {len(val_ds):,}")

    # DataLoaders with optional DDP sampler
    loader_kwargs = dict(
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if CONFIG["num_workers"] > 0 else False,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], sampler=val_sampler, **loader_kwargs)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────
    if is_main:
        print("\nBuilding model...")

    import timm
    from models import DINOv2Refiner
    from losses import FocalBCELoss

    # Load DINOv2 backbone
    dinov2_backbone = timm.create_model(
        CONFIG["dinov2_variant"],
        pretrained=True,
        num_classes=0,
    )

    model = DINOv2Refiner(
        dinov2_model=dinov2_backbone,
        spa_channels=64,
        cls_dim=CONFIG["cls_dim"],
        num_slices=CONFIG["slice_count"],
        num_classes=CONFIG["num_classes"],
        num_heads=4,
        slice_transformer_layers=2,
        dropout=CONFIG["dropout"],
        freeze_dinov2=True,
    ).to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # ── Loss & Optimizer ──────────────────────────────────
    criterion = FocalBCELoss(
        gamma=CONFIG["focal_gamma"],
        alpha=CONFIG["focal_alpha"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999),
    )

    # CosineAnnealingWarmRestarts scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CONFIG["lr_t0"],
        T_mult=CONFIG["lr_t_mult"],
        eta_min=CONFIG["lr_eta_min"],
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if CONFIG["mixed_precision"] and torch.cuda.is_available() else None

    # ── Training Loop ─────────────────────────────────────
    best_auc = 0.0
    patience_counter = 0
    train_start = time.time()

    # Create output directory
    ckpt_dir = KAGGLE_OUTPUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"\n{'='*60}")
        print(f"Starting training — {CONFIG['epochs']} epochs")
        print(f"{'='*60}\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            epoch=epoch, rank=rank, device=device,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device=device, rank=rank)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - train_start
        lr_now = optimizer.param_groups[0]["lr"]

        if is_main:
            # VRAM
            vram = torch.cuda.max_memory_allocated(device) / 1024**3
            torch.cuda.reset_peak_memory_stats(device)

            print(f"\n── Epoch {epoch:3d}/{CONFIG['epochs']} ────────────────────────")
            print(f"  Train Loss:     {train_loss:.4f}")
            print(f"  Val Loss:       {val_metrics['loss']:.4f}")
            print(f"  Val Macro AUC:  {val_metrics['macro_auc']:.4f}")
            print(f"  LR:             {lr_now:.2e}")
            print(f"  Time:           {epoch_time:.0f}s epoch | {elapsed_total/60:.0f}min total")
            print(f"  VRAM Peak:      {vram:.1f} GB")

            # Per-class AUC
            per_class = val_metrics.get("per_class_auc", {})
            if per_class:
                auc_strs = []
                for c, auc in per_class.items():
                    if not math.isnan(auc):
                        auc_strs.append(f"{c}={auc:.3f}")
                print(f"  Per-class:      {', '.join(auc_strs)}")

            # ── Checkpointing ──────────────────────────────
            current_auc = val_metrics["macro_auc"]

            if current_auc > best_auc + CONFIG["early_stop_min_delta"]:
                best_auc = current_auc
                patience_counter = 0
                ckpt_path = ckpt_dir / "best_model.pt"
                state = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": state,
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                    "config": CONFIG,
                }, ckpt_path)
                print(f"  ✓ Best model saved (AUC={best_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["early_stop_patience"]:
                    print(f"\n  Early stopping triggered at epoch {epoch}")
                    break

            # Periodic checkpoint
            if epoch % CONFIG["checkpoint_interval"] == 0:
                ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch}.pt"
                state = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": state,
                    "best_auc": best_auc,
                    "config": CONFIG,
                }, ckpt_path)

            print()

    # ── Final Summary ─────────────────────────────────────
    if is_main:
        total_time = time.time() - train_start
        print(f"\n{'='*60}")
        print(f"Training Complete")
        print(f"  Best Val Macro AUC: {best_auc:.4f}")
        print(f"  Total Time:         {total_time/3600:.1f} hours")
        print(f"  Checkpoints saved:  {ckpt_dir}")
        print(f"{'='*60}")

    cleanup_ddp()
    return best_auc


if __name__ == "__main__":
    main()
