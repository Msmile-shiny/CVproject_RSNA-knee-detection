"""
Standalone gold-study validation using a saved v3 checkpoint.

Usage (Kaggle):
    python validate_gold.py

Requires:
    - Trained checkpoint at /kaggle/working/checkpoints/best_model.pt
    - RSNA competition data at /kaggle/input/competitions/rsna-knee-abnormality-detection

Loads only gold-labeled studies (all 12 targets annotated), builds a minimal
uint8 cache, and computes per-class AUC.  Total wall time: ~2-3 min, GPU: ~10 s.
"""

from __future__ import annotations

import gc, math, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import pydicom
import cv2
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────
COMP_INPUT  = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CKPT_PATH   = Path("/kaggle/working/checkpoints/best_model.pt")
DICOM_ROOT  = COMP_INPUT / "train_series"

# ── Constants (mirrored from build_v3 config) ──────────────────────────────
TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

SLOTS = [
    ("SAG_FLUID_FS",   "Sagittal", True,  True),
    ("COR_FLUID_FS",   "Coronal",  True,  True),
    ("AX_FLUID_FS",    "Axial",    True,  True),
    ("SAG_FLUID_NOFS", "Sagittal", True,  False),
    ("COR_T1",         "Coronal",  False, False),
    ("SAG_T1",         "Sagittal", False, False),
]
N_SLOT = len(SLOTS)

SLOT_PRIORS = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5), "Effusion": (0, 2), "Synovitis": (0, 2),
    "Baker's": (0,), "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}

IMAGE_SIZE   = 224
CACHE_SLICES = 9
GROUP_SIZE   = 3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── SlotHead + Model (mirrors cells_v3/06_model.py) ────────────────────────

class SlotHead(nn.Module):
    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        prior = torch.zeros(n_out, n_slot)
        for target_name, slot_indices in SLOT_PRIORS.items():
            if target_name in TARGET_COLUMNS:
                prior[TARGET_COLUMNS.index(target_name), list(slot_indices)] = 0.55
        self.register_buffer("slot_prior", prior)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        attention = (
            torch.einsum("bsh,oh->bos", h, self.query) / math.sqrt(self.hidden)
            + self.slot_prior.unsqueeze(0)
        )
        attention = attention.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        context = self.drop(torch.einsum("bos,bsh->boh", attention, h))
        return (context * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class MultiViewModel(nn.Module):
    def __init__(self, dinov2_model, n_slots=6, cls_dim=384, n_classes=12,
                 slot_hidden=256, dropout=0.2, unfreeze_layers=6):
        super().__init__()
        self.n_slots = n_slots
        self.cls_dim = cls_dim
        self.feature_dim = cls_dim * 3
        self.unfreeze_layers = unfreeze_layers
        self.dinov2 = dinov2_model
        n_blocks = len(self.dinov2.blocks)
        if unfreeze_layers > 0:
            for p in self.dinov2.parameters():
                p.requires_grad = False
            unfreeze_start = max(0, n_blocks - unfreeze_layers)
            for block in self.dinov2.blocks[unfreeze_start:]:
                for p in block.parameters():
                    p.requires_grad = True
            if hasattr(self.dinov2, 'norm'):
                for p in self.dinov2.norm.parameters():
                    p.requires_grad = True
        self.head = SlotHead(dim=self.feature_dim, n_slot=n_slots, n_out=n_classes,
                             hidden=slot_hidden, p=dropout)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _extract_features(self, x_3ch):
        if self.unfreeze_layers > 0:
            features = self.dinov2.forward_features(x_3ch)
        else:
            with torch.no_grad():
                features = self.dinov2.forward_features(x_3ch)
        cls = features[:, 0, :]
        patches = features[:, 1:, :]
        mean_p = patches.mean(dim=1)
        k = max(1, patches.shape[1] // 8)
        focal = patches.topk(k, dim=1).values.mean(dim=1)
        return torch.cat([cls, mean_p, focal], dim=1)

    def forward(self, images, mask):
        B, S = images.shape[:2]
        x = images.reshape(B * S, 3, images.shape[-2], images.shape[-1])
        x = x.float().div_(255.0)
        x = (x - self.mean) / self.std
        features = self._extract_features(x)
        features = features.reshape(B, S, -1)
        return self.head(features, mask)


# ── DICOM helpers (mirrors cells_v3/05_dicom_io.py) ────────────────────────

PLANE_SORT_AXIS = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
_DICOM_TAGS = [
    (0x0020, 0x0032), (0x0020, 0x0013), (0x0028, 0x0002), (0x0028, 0x0004),
    (0x0028, 0x0010), (0x0028, 0x0011), (0x0028, 0x0100), (0x0028, 0x0101),
    (0x0028, 0x0102), (0x0028, 0x0103), (0x0028, 0x0030), (0x0028, 0x1052),
    (0x0028, 0x1053), (0x7FE0, 0x0010),
]


def _get_slice_position(ds, plane=None):
    try:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp and len(ipp) >= 3:
            return float(ipp[PLANE_SORT_AXIS.get(plane, 2)])
    except Exception:
        pass
    try:
        sl = getattr(ds, "SliceLocation", None)
        if sl is not None: return float(sl)
    except Exception:
        pass
    try:
        return float(getattr(ds, "InstanceNumber", 0))
    except Exception:
        return 0.0


def read_series_volume(series_dir, plane=None, image_size=224):
    series_dir = Path(series_dir)
    dcm_paths = sorted(series_dir.glob("*.dcm"))
    if not dcm_paths:
        dcm_paths = sorted(series_dir.glob("*"))
    if not dcm_paths:
        return None, None

    slices_info = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), force=True, specific_tags=_DICOM_TAGS)
            pos = _get_slice_position(ds, plane)
            img = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
            img = img * slope + intercept
            slices_info.append((pos, img))
        except Exception:
            continue

    if not slices_info:
        return None, None

    slices_info.sort(key=lambda x: x[0])
    images = np.stack([img for _, img in slices_info], axis=0)

    v_low = np.percentile(images, 1.0)
    v_high = np.percentile(images, 99.0)
    images = np.clip(images, v_low, v_high)
    denom = max(v_high - v_low, 1e-6)
    images = (images - v_low) / denom

    resized = []
    for img in images:
        r = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        resized.append(r)
    return np.stack(resized, axis=0).astype(np.float32), None


def sample_cache_slices(volume, n_cache=9, center_pct=(0.2, 0.8)):
    n_total = volume.shape[0]
    if n_total <= n_cache:
        indices = list(range(n_total))
        while len(indices) < n_cache:
            indices.append(indices[-1])
        return volume[np.array(indices)]
    low = int(center_pct[0] * (n_total - 1))
    high = int(center_pct[1] * (n_total - 1))
    if high <= low:
        low, high = 0, n_total - 1
    indices = np.unique(np.linspace(low, high, n_cache).astype(int))
    while len(indices) < n_cache:
        indices = np.append(indices, indices[-1])
    return volume[indices[:n_cache]]


# ── Slot matching (mirrors cells_v3/04_slot_matching.py) ───────────────────

def match_slots_for_study(study_series_df):
    slots_found = {}
    for slot_name, plane, fluid, fatsat in SLOTS:
        candidates = study_series_df[
            (study_series_df["Anatomical_Plane"] == plane)
            & (study_series_df["Fluid_Sensitive"] == (1 if fluid else 0))
            & (study_series_df["Fat_Suppression"] == (1 if fatsat else 0))
        ]
        if len(candidates) == 0 and not fluid:
            candidates = study_series_df[
                (study_series_df["Anatomical_Plane"] == plane)
                & (study_series_df["Fluid_Sensitive"] == 0)
            ]
        if len(candidates) > 0:
            best = candidates.sort_values("n_slices", ascending=False).iloc[0]
            slots_found[slot_name] = {
                "series_uid": best["SeriesInstanceUID"],
                "dir": best["dir"],
                "n_slices": int(best["n_slices"]),
                "plane": plane,
            }
        else:
            slots_found[slot_name] = None
    return slots_found


def build_study_slot_map(series_meta, dicom_root):
    df = series_meta.copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["SeriesInstanceUID"] = df["SeriesInstanceUID"].astype(str)
    dirs, n_slices_list = [], []
    for _, row in df.iterrows():
        d = str(dicom_root / row["StudyInstanceUID"] / row["SeriesInstanceUID"])
        dirs.append(d)
        if os.path.isdir(d):
            n_slices_list.append(len([f for f in os.listdir(d) if f.endswith(".dcm")]))
        else:
            n_slices_list.append(0)
    df["dir"] = dirs
    df["n_slices"] = n_slices_list

    slot_map, study_series_map = {}, {}
    for study_uid, grp in df.groupby("StudyInstanceUID"):
        study_series_map[study_uid] = grp
        slot_map[study_uid] = match_slots_for_study(grp)
    return slot_map, study_series_map


# ── Cache builder (gold studies only) ─────────────────────────────────────

def build_gold_cache(gold_uids, slot_map, dicom_root):
    """Build uint8 cache for gold studies only."""
    n_studies = len(gold_uids)
    cache_shape = (n_studies, N_SLOT, CACHE_SLICES, IMAGE_SIZE, IMAGE_SIZE)
    cache = np.zeros(cache_shape, dtype=np.uint8)
    mask_arr = np.zeros((n_studies, N_SLOT), dtype=np.float32)
    study_index = {}

    t0 = time.time()
    completed, failed = 0, 0

    for row_idx, study_uid in enumerate(sorted(gold_uids)):
        study_index[study_uid] = row_idx
        study_slots = slot_map.get(study_uid, {})

        for slot_idx, (slot_name, plane, _fluid, _fatsat) in enumerate(SLOTS):
            slot_info = study_slots.get(slot_name)
            if slot_info is None:
                continue
            series_dir = Path(slot_info["dir"]) if "dir" in slot_info else None
            if series_dir is None or not series_dir.exists():
                continue
            try:
                volume, _px = read_series_volume(str(series_dir), plane=plane,
                                                  image_size=IMAGE_SIZE)
                if volume is None or volume.shape[0] < 3:
                    continue
                sampled = sample_cache_slices(volume, n_cache=CACHE_SLICES)
                sampled_uint8 = (sampled * 255).clip(0, 255).round().astype(np.uint8)
                cache[row_idx, slot_idx] = sampled_uint8
                mask_arr[row_idx, slot_idx] = 1.0
            except Exception:
                failed += 1
                continue

        completed += 1
        if completed % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{completed:3d}/{n_studies}] {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Cache: {n_studies} studies, {int(mask_arr.sum())} series, "
          f"{cache.nbytes/1024**3:.1f} GB in {elapsed:.0f}s "
          f"(avg {mask_arr.sum()/n_studies:.1f} slots/study)")
    return cache, mask_arr, study_index


# ── Validation ─────────────────────────────────────────────────────────────

class GoldValDataset(Dataset):
    """Returns gold studies with ALL window positions for TTA.

    Each item returns all 7 possible 3-slice windows from 9 cached slices.
    The evaluate() function averages logits across windows (matching ref code TTA).
    """
    def __init__(self, gold_uids, cache, mask_arr, labels_df, study_index):
        self.gold_uids = gold_uids
        self.cache = cache
        self.mask_arr = mask_arr
        self.labels_df = labels_df
        self.study_index = study_index

    def __len__(self):
        return len(self.gold_uids)

    def __getitem__(self, idx):
        uid = self.gold_uids[idx]
        row_idx = self.study_index[uid]
        slots = torch.from_numpy(self.cache[row_idx].copy())    # [6, 9, 224, 224]
        mask = torch.from_numpy(self.mask_arr[row_idx].copy())   # [6]

        # Extract ALL 7 possible 3-slice windows from 9 cached slices
        n_windows = CACHE_SLICES - GROUP_SIZE + 1  # 7
        windows = []
        for w in range(n_windows):
            windows.append(slots[:, w:w + GROUP_SIZE])  # each [6, 3, 224, 224]
        all_windows = torch.stack(windows, dim=0)  # [7, 6, 3, 224, 224]

        labels = torch.tensor(
            [float(self.labels_df.loc[uid, c]) for c in TARGET_COLUMNS],
            dtype=torch.float32)
        return {"windows": all_windows, "mask": mask, "labels": labels, "study_uid": uid}


@torch.no_grad()
def evaluate(model, loader):
    """TTA inference: average logits across 7 window positions."""
    model.eval()
    all_logits, all_labels, all_uids = [], [], []
    for batch in loader:
        windows = batch["windows"].to(DEVICE, non_blocking=True)  # [B, 7, 6, 3, H, W]
        mask = batch["mask"].to(DEVICE, non_blocking=True)         # [B, 6]
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        B, W = windows.shape[0], windows.shape[1]
        flat = windows.reshape(B * W, windows.shape[2], windows.shape[3],
                               windows.shape[4], windows.shape[5])  # [B*7, 6, 3, H, W]
        flat_mask = mask.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1)  # [B*7, 6]
        logits = model(flat, flat_mask)                            # [B*7, 12]
        logits_reshaped = logits.reshape(B, W, -1)                 # [B, 7, 12]
        logits_avg = logits_reshaped.mean(dim=1)                   # [B, 12] — average across windows

        all_logits.append(logits_avg.cpu())
        all_labels.append(labels.cpu())
        all_uids.extend(batch["study_uid"])

    logits_all = torch.cat(all_logits, dim=0).numpy()
    labels_all = torch.cat(all_labels, dim=0).numpy()
    probs_all = 1.0 / (1.0 + np.exp(-logits_all))

    return logits_all, labels_all, probs_all, all_uids


def compute_aucs(labels, probs):
    aucs = {}
    for i, c in enumerate(TARGET_COLUMNS):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            aucs[c] = float("nan")
        else:
            try:
                aucs[c] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                aucs[c] = float("nan")
    valid = [v for v in aucs.values() if not math.isnan(v)]
    return aucs, float(np.mean(valid)) if valid else float("nan")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CKPT_PATH}")

    # 1. Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    ckpt_auc = ckpt.get('auc')
    if isinstance(ckpt_auc, float):
        print(f"  epoch={ckpt.get('epoch')}  checkpoint_auc={ckpt_auc:.4f}")
    else:
        print(f"  epoch={ckpt.get('epoch')}")

    # 2. Find gold studies
    train_meta = pd.read_csv(COMP_INPUT / "train.csv")
    train_meta["StudyInstanceUID"] = train_meta["StudyInstanceUID"].astype(str)
    label_cols = [c for c in TARGET_COLUMNS if c in train_meta.columns]
    has_label = train_meta[label_cols].notna().all(axis=1)
    gold_df = train_meta[has_label].copy()
    gold_uids = sorted(gold_df["StudyInstanceUID"].unique())
    print(f"Gold studies: {len(gold_uids)}")

    # Build gold labels
    gold_labels = gold_df[["StudyInstanceUID"] + label_cols].copy()
    gold_labels = gold_labels.set_index("StudyInstanceUID")
    for c in TARGET_COLUMNS:
        if c not in gold_labels.columns:
            gold_labels[c] = 0.0
    gold_labels = gold_labels.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    # 3. Build slot map (filter to gold studies only)
    series_meta = pd.read_csv(COMP_INPUT / "train_series.csv")
    series_meta["StudyInstanceUID"] = series_meta["StudyInstanceUID"].astype(str)
    series_meta["SeriesInstanceUID"] = series_meta["SeriesInstanceUID"].astype(str)
    gold_series = series_meta[series_meta["StudyInstanceUID"].isin(gold_uids)]
    print(f"Series rows: {len(series_meta):,} → {len(gold_series):,} (gold only)")

    slot_map, _ = build_study_slot_map(gold_series, DICOM_ROOT)

    # 4. Build cache
    valid_uids = [u for u in gold_uids if u in slot_map]
    print(f"Gold studies with DICOM: {len(valid_uids)}")
    cache, mask_arr, study_index = build_gold_cache(valid_uids, slot_map, DICOM_ROOT)

    # 5. Build model & load weights (handle DataParallel prefix)
    dinov2 = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=IMAGE_SIZE)
    model = MultiViewModel(dinov2, n_slots=N_SLOT, cls_dim=384, n_classes=12,
                           slot_hidden=256, dropout=0.2, unfreeze_layers=6).to(DEVICE)

    state_dict = ckpt["model"]
    # Strip 'module.' prefix if saved from DataParallel (Kaggle: 2 GPUs)
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        print("  Stripped DataParallel prefix from state_dict")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model.eval()

    # 6. Evaluate
    ds = GoldValDataset(valid_uids, cache, mask_arr, gold_labels, study_index)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    logits, labels, probs, uids = evaluate(model, loader)

    # 7. Report
    aucs, macro_auc = compute_aucs(labels, probs)
    print(f"\n{'='*60}")
    print(f"GOLD-STUDY VALIDATION  ({len(valid_uids)} studies)")
    print(f"{'='*60}")
    print(f"  {'Class':<20s} {'AUC':>7s} {'Pos':>5s}")
    print(f"  {'-'*20} {'-'*7} {'-'*5}")
    for i, c in enumerate(TARGET_COLUMNS):
        a = aucs[c]
        auc_str = f"{a:.4f}" if not math.isnan(a) else "  N/A  "
        print(f"  {c:<20s} {auc_str:>7s} {int(labels[:, i].sum()):5d}")
    print(f"  {'-'*20} {'-'*7} {'-'*5}")
    print(f"  Macro AUC: {macro_auc:.4f}")

    # Save
    out = Path("/kaggle/working")
    rows = []
    for i, uid in enumerate(uids):
        row = {"StudyInstanceUID": uid}
        for j, c in enumerate(TARGET_COLUMNS):
            row[f"true_{c}"] = int(labels[i, j])
            row[f"pred_{c}"] = float(probs[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "gold_validation_predictions.csv", index=False)

    auc_rows = [{"class": c, "auc": aucs[c], "n_pos": int(labels[:, i].sum())}
                for i, c in enumerate(TARGET_COLUMNS)]
    auc_df = pd.DataFrame(auc_rows)
    auc_df.loc["macro"] = ["macro_avg", macro_auc, ""]
    auc_df.to_csv(out / "gold_validation_auc.csv", index=False)
    print(f"\nSaved: gold_validation_predictions.csv, gold_validation_auc.csv")


if __name__ == "__main__":
    main()
