# ============================================================
# v3: Gold-study validation — 完全自包含，不依赖前面的 cell
# ============================================================
#
# 使用方式：在 Kaggle notebook 中直接运行本 cell 即可。
# 不需要先跑前面的 cell。本 cell 包含所有需要的定义。
#
# 前提条件：
#   1. Kaggle notebook 已添加 RSNA 竞赛数据 (train.csv, train_series.csv, DICOM)
#   2. Kaggle notebook 已添加 checkpoint dataset (easoncyy/rsna-knee-v3-checkpoint)
#      或者 checkpoint 在 /kaggle/working/checkpoints/best_model.pt（上次训练 session）
#
# 预期运行时间：~2-3 分钟
# ============================================================

# ---- 0. 环境准备 ----
!pip install -q timm pydicom opencv-python scikit-learn 2>/dev/null

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ---- 1. 配置 (mirrors cells_v3/03_config.py) ----
TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
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

# ---- 2. 模型定义 (mirrors cells_v3/06_model.py) ----

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


# ---- 3. DICOM 读取 (mirrors cells_v3/05_dicom_io.py) ----

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
    except Exception: pass
    try:
        sl = getattr(ds, "SliceLocation", None)
        if sl is not None: return float(sl)
    except Exception: pass
    try:
        return float(getattr(ds, "InstanceNumber", 0))
    except Exception: return 0.0

def read_series_volume(series_dir, plane=None, image_size=224):
    series_dir = Path(series_dir)
    dcm_paths = sorted(series_dir.glob("*.dcm"))
    if not dcm_paths: dcm_paths = sorted(series_dir.glob("*"))
    if not dcm_paths: return None, None
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
        except Exception: continue
    if not slices_info: return None, None
    slices_info.sort(key=lambda x: x[0])
    images = np.stack([img for _, img in slices_info], axis=0)
    v_low, v_high = np.percentile(images, 1.0), np.percentile(images, 99.0)
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
        while len(indices) < n_cache: indices.append(indices[-1])
        return volume[np.array(indices)]
    low, high = int(center_pct[0] * (n_total - 1)), int(center_pct[1] * (n_total - 1))
    if high <= low: low, high = 0, n_total - 1
    indices = np.unique(np.linspace(low, high, n_cache).astype(int))
    while len(indices) < n_cache: indices = np.append(indices, indices[-1])
    return volume[indices[:n_cache]]


# ---- 4. Slot 匹配 (mirrors cells_v3/04_slot_matching.py) ----

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
    for col in ["Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]:
        if col not in df.columns:
            raise KeyError(f"train_series.csv missing column: {col}")
    slot_map, study_series_map = {}, {}
    for study_uid, grp in df.groupby("StudyInstanceUID"):
        study_series_map[study_uid] = grp
        slot_map[study_uid] = match_slots_for_study(grp)
    return slot_map, study_series_map


# ============================================================
# 以下是实际验证逻辑
# ============================================================

print("=" * 60)
print("GOLD-STUDY VALIDATION")
print("=" * 60)

# ---- 5. 路径设置 ----
COMP_INPUT = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
DICOM_ROOT = COMP_INPUT / "train_series"

# ---- 6. 找所有 gold 研究 ----
train_meta = pd.read_csv(COMP_INPUT / "train.csv")
train_meta["StudyInstanceUID"] = train_meta["StudyInstanceUID"].astype(str)
label_cols = [c for c in TARGET_COLUMNS if c in train_meta.columns]
has_all_labels = train_meta[label_cols].notna().all(axis=1)
gold_df = train_meta[has_all_labels].copy()
gold_uids = sorted(gold_df["StudyInstanceUID"].unique())

gold_labels = gold_df[["StudyInstanceUID"] + label_cols].copy()
gold_labels = gold_labels.set_index("StudyInstanceUID")
for c in TARGET_COLUMNS:
    if c not in gold_labels.columns:
        gold_labels[c] = np.nan
gold_labels = gold_labels.apply(pd.to_numeric, errors="coerce")

print(f"Gold studies (all 12 labeled): {len(gold_uids)}")
n_pos_per_class = (gold_labels > 0).sum(axis=0)
print(f"Positives per class: min={int(n_pos_per_class.min())}, "
      f"max={int(n_pos_per_class.max())}, mean={n_pos_per_class.mean():.1f}")

# ---- 7. 构建 gold-only slot map ----
series_meta = pd.read_csv(COMP_INPUT / "train_series.csv")
series_meta["StudyInstanceUID"] = series_meta["StudyInstanceUID"].astype(str)
series_meta["SeriesInstanceUID"] = series_meta["SeriesInstanceUID"].astype(str)
gold_series = series_meta[series_meta["StudyInstanceUID"].isin(gold_uids)]
print(f"Series rows: {len(series_meta):,} -> {len(gold_series):,} (gold only)")

gold_slot_map, _ = build_study_slot_map(gold_series, DICOM_ROOT)
valid_gold_uids = sorted([u for u in gold_uids if u in gold_slot_map])
print(f"Gold studies with DICOM: {len(valid_gold_uids)}")

# ---- 8. 构建 gold-only 缓存 ----
n_gold = len(valid_gold_uids)
GOLD_CACHE = np.zeros(
    (n_gold, N_SLOT, CACHE_SLICES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
GOLD_MASK = np.zeros((n_gold, N_SLOT), dtype=np.float32)
gold_study_idx = {}

t0 = time.time()
completed, failed = 0, 0

for row_idx, study_uid in enumerate(valid_gold_uids):
    gold_study_idx[study_uid] = row_idx
    study_slots = gold_slot_map[study_uid]

    for slot_idx, (slot_name, plane, fluid, fatsat) in enumerate(SLOTS):
        slot_info = study_slots.get(slot_name)
        if slot_info is None:
            continue
        series_dir = Path(slot_info["dir"]) if "dir" in slot_info else None
        if series_dir is None or not series_dir.exists():
            continue
        try:
            volume, px = read_series_volume(str(series_dir), plane=plane, image_size=IMAGE_SIZE)
            if volume is None or volume.shape[0] < 3:
                continue
            sampled = sample_cache_slices(volume, n_cache=CACHE_SLICES)
            sampled_uint8 = (sampled * 255).clip(0, 255).round().astype(np.uint8)
            GOLD_CACHE[row_idx, slot_idx] = sampled_uint8
            GOLD_MASK[row_idx, slot_idx] = 1.0
        except Exception:
            failed += 1
            continue
    completed += 1
    if completed % 20 == 0:
        print(f"  [{completed:3d}/{n_gold}] {time.time()-t0:.0f}s", flush=True)

elapsed = time.time() - t0
n_series = int(GOLD_MASK.sum())
print(f"Gold cache: {n_gold} studies, {n_series} series, "
      f"{GOLD_CACHE.nbytes/1024**3:.2f} GB in {elapsed:.0f}s "
      f"(avg {n_series/max(n_gold,1):.1f} slots/study, {failed} failed)")
gc.collect()

# ---- 9. 加载 checkpoint ----
ckpt_candidates = [
    Path("/kaggle/working/checkpoints/best_model.pt"),             # 旧 session
    Path("/kaggle/input/rsna-knee-v3-checkpoint/best_model.pt"),   # 上传的 dataset
]
ckpt_path = None
for p in ckpt_candidates:
    if p.exists():
        ckpt_path = p
        break

if ckpt_path is None:
    raise FileNotFoundError(
        "Checkpoint not found!\n"
        f"  请确认以下之一存在:\n"
        f"  1. {ckpt_candidates[0]} (上次训练 session 还在)\n"
        f"  2. {ckpt_candidates[1]} (上传的 dataset)\n"
        "  如果是新 session，需在右侧 Add Input -> Dataset -> "
        "搜索 'easoncyy/rsna-knee-v3-checkpoint' 并添加。"
    )

print(f"\nCheckpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"  epoch={ckpt.get('epoch')}  saved_val_auc={ckpt.get('auc', 0):.4f}")

# ---- 10. 构建模型 & 加载权重 ----
dinov2 = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m", pretrained=True,
    num_classes=0, img_size=IMAGE_SIZE)

model = MultiViewModel(
    dinov2_model=dinov2, n_slots=N_SLOT, cls_dim=384, n_classes=12,
    slot_hidden=256, dropout=0.2, unfreeze_layers=6,
).to(DEVICE)

state_dict = ckpt["model"]
first_key = next(iter(state_dict))
if first_key.startswith("module."):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    print("  Stripped DataParallel prefix")

missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"  WARNING: {len(missing)} missing keys")
if unexpected:
    print(f"  WARNING: {len(unexpected)} unexpected keys")

model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model loaded: {n_params/1e6:.1f}M params")

# ---- 11. TTA 推理 ----

class GoldDataset(Dataset):
    def __init__(self, uids, cache_arr, mask_arr, labels_df, study_idx):
        self.uids = uids; self.cache = cache_arr; self.mask = mask_arr
        self.labels = labels_df; self.study_idx = study_idx
    def __len__(self): return len(self.uids)
    def __getitem__(self, idx):
        uid = self.uids[idx]; ri = self.study_idx[uid]
        slots = torch.from_numpy(self.cache[ri].copy())
        m = torch.from_numpy(self.mask[ri].copy())
        # 7 个 3-slice 窗口
        windows = torch.stack([slots[:, w:w+3] for w in range(CACHE_SLICES - 3 + 1)], dim=0)
        labs = torch.tensor(
            [float(self.labels.loc[uid, c]) if not pd.isna(self.labels.loc[uid, c]) else 0.0
             for c in TARGET_COLUMNS], dtype=torch.float32)
        return windows, m, labs, uid


ds = GoldDataset(valid_gold_uids, GOLD_CACHE, GOLD_MASK, gold_labels, gold_study_idx)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)


@torch.no_grad()
def tta_evaluate(model, loader):
    model.eval()
    logits_list, labels_list, uids_list = [], [], []
    for windows, mask, labels, uids in loader:
        windows = windows.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        B, W = windows.shape[0], windows.shape[1]
        flat = windows.reshape(B * W, *windows.shape[2:])
        flat_mask = mask.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1)
        logits = model(flat, flat_mask).reshape(B, W, -1).mean(dim=1)
        logits_list.append(logits.cpu()); labels_list.append(labels); uids_list.extend(uids)
    logits_arr = torch.cat(logits_list).numpy()
    labels_arr = torch.cat(labels_list).numpy()
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
    return logits_arr, labels_arr, probs_arr, uids_list


print("Running TTA inference (7 windows x {} studies)...".format(len(valid_gold_uids)))
t1 = time.time()
logits_arr, labels_arr, probs_arr, all_uids = tta_evaluate(model, loader)
print(f"  Done in {time.time()-t1:.1f}s")

# ---- 12. Per-class AUC ----
aucs = {}
for i, c in enumerate(TARGET_COLUMNS):
    yt, yp = labels_arr[:, i], probs_arr[:, i]
    n_pos = int(yt.sum()); n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        aucs[c] = float("nan")
    else:
        try: aucs[c] = float(roc_auc_score(yt, yp))
        except Exception: aucs[c] = float("nan")

valid_aucs = [v for v in aucs.values() if not math.isnan(v)]
macro = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

print(f"\n{'='*65}")
print(f"  GOLD VALIDATION -- {len(valid_gold_uids)} studies, 7-window TTA")
print(f"{'='*65}")
print(f"  {'Class':<20s} {'AUC':>7s} {'Pos':>5s} {'Neg':>5s}")
print(f"  {'-'*20} {'-'*7} {'-'*5} {'-'*5}")
for i, c in enumerate(TARGET_COLUMNS):
    a = aucs[c]
    n_pos_ = int(labels_arr[:, i].sum())
    n_neg_ = len(labels_arr) - n_pos_
    auc_str = f"{a:.4f}" if not math.isnan(a) else "  N/A  "
    print(f"  {c:<20s} {auc_str:>7s} {n_pos_:5d} {n_neg_:5d}")
print(f"  {'-'*20} {'-'*7} {'-'*5} {'-'*5}")
print(f"  {'Macro AUC':<20s} {macro:7.4f}")

# ---- 13. 保存结果 ----
out = Path("/kaggle/working")

rows = []
for i, uid in enumerate(all_uids):
    row = {"StudyInstanceUID": uid}
    for j, c in enumerate(TARGET_COLUMNS):
        row[f"true_{c}"] = int(labels_arr[i, j])
        row[f"prob_{c}"] = float(probs_arr[i, j])
    rows.append(row)
pd.DataFrame(rows).to_csv(out / "gold_validation_predictions.csv", index=False)

auc_rows = [{"class": c, "auc": aucs[c], "n_pos": int(labels_arr[:, i].sum())}
            for i, c in enumerate(TARGET_COLUMNS)]
auc_df = pd.DataFrame(auc_rows)
auc_df.loc["macro_avg"] = ["macro_avg", macro, ""]
auc_df.to_csv(out / "gold_validation_auc.csv", index=False)

print(f"\nSaved:")
print(f"  {out / 'gold_validation_predictions.csv'}")
print(f"  {out / 'gold_validation_auc.csv'}")
print(f"\nDone. Macro AUC = {macro:.4f} on {len(valid_gold_uids)} gold studies.")
