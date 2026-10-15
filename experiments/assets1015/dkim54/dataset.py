#!/usr/bin/env python3
"""RSNA Knee — dataset / preprocessing.

Builds a per-study 2.5D image stack from DICOM series, using the free per-series
metadata (Fluid_Sensitive, Fat_Suppression, Anatomical_Plane) to budget slices per plane.

Design (from the strategy doc, validated against real data 2026-09-30):
  - Studies laid out: data/train_series/<StudyUID>/<SeriesUID>/<slice>.dcm
  - All slices uncompressed (TS 1.2.840.10008.1.2.1) -> no special codec needed.
  - Pixel shapes vary 320x300..1008x1008 -> normalize with a physical-mm crop via PixelSpacing,
    then resize to a fixed size.
  - 2.5D: pack 3 neighbouring slices as RGB channels.
  - Per-plane slice budget so effusion/synovitis (fluid-sensitive) and OA (any) are all represented.

Labels come from the teacher soft-label CSV (study -> 12 probs). Gold 58 override softs.
"""
import os, csv, json, math, numpy as np

LABELS=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
        "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

DATA_ROOT = "/mnt/nvme/coding_data/Kndat/data"
SERIES_CSV = os.path.join(DATA_ROOT, "train_series.csv")
TRAIN_CSV  = os.path.join(DATA_ROOT, "train.csv")

# per-plane slice budget (sums to 64, matching the reference 0.924 model's stack size)
PLANE_BUDGET = {"Sagittal":32, "Coronal":20, "Axial":12}
CROP_MM = 140.0
OUT_SIZE = 384

def load_series_index():
    """study -> list of (series_uid, fluid, fat, plane)."""
    idx={}
    for r in csv.DictReader(open(SERIES_CSV,encoding='utf-8')):
        idx.setdefault(r["StudyInstanceUID"],[]).append(
            (r["SeriesInstanceUID"], r["Fluid_Sensitive"], r["Fat_Suppression"], r["Anatomical_Plane"]))
    return idx

SECONDARY_TIEBREAK = "slices"  # "slices" (default, unchanged) | "diverse" (prefer T2/STIR as secondary)
MAX_SERIES_PER_PLANE = 2  # 2 (default, unchanged): each plane contributes primary + secondary only.
                          # 3 splits the plane's slice budget over a third series when one exists.
                          # Measured: 3,579 of 24,371 series (14.7%) are currently discarded by the cap,
                          # and studies with >=8 series score 0.8997 vs 0.9332 overall. Needs a layout
                          # whose per-plane budget divides by 3 — see LAYOUT_V2_S3.
CROP_CENTRE = "frame"  # "frame" (default, unchanged): crop about the geometric centre of the image.
                       # "tissue": crop about the intensity-weighted tissue centroid of each slice.
                       # Measured offset between the two: median 13.2 mm, p90 30.2 mm — at a 140 mm FOV
                       # that is enough to push the joint line well off centre, and the training-time
                       # affine jitter is only +/-5%.
RESIZE_MODE = "area"  # "area" (default, unchanged): always cv2.INTER_AREA.
                      # "adaptive": INTER_AREA when shrinking, INTER_CUBIC when GROWING. cv2.INTER_AREA
                      # degenerates to nearest-neighbour on upscale (verified: integer upscale is
                      # bit-identical to INTER_NEAREST), and ~30% of our series have an effective crop
                      # smaller than 384 px, so they are currently fed to the backbone blocky/aliased.
NORM_MODE = "slice"   # "slice" (default, unchanged): 1-99 percentile per slice.
                      # "series": 1-99 percentile pooled over the series' selected slices (every public notebook
                      # normalises per series/volume, which keeps intensity consistent along the series).


def _interp(src_h, src_w, out):
    """Interpolation for a crop of src_h x src_w -> out x out, honouring RESIZE_MODE."""
    import cv2
    if RESIZE_MODE == "adaptive" and (src_h < out or src_w < out):
        return cv2.INTER_CUBIC          # growing: INTER_AREA would be nearest-neighbour
    return cv2.INTER_AREA


def _read_raw(path, want_mm=CROP_MM, out=OUT_SIZE):
    """Physical-mm centre crop + resize, NO intensity normalisation (float32, scanner units)."""
    import pydicom, cv2
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    ps = getattr(ds, "PixelSpacing", None)
    if ps:
        rmm, cmm = float(ps[0]), float(ps[1])
        rpx = int(round(want_mm / rmm)); cpx = int(round(want_mm / cmm))
        r0, c0 = _crop_origin(arr, rpx, cpx)
        arr = arr[r0:r0 + rpx, c0:c0 + cpx]
    return cv2.resize(arr, (out, out), interpolation=_interp(arr.shape[0], arr.shape[1], out))


def _norm_series(slices):
    """1-99 percentile pooled over all slices of one series -> [0,1] (subsampled for speed)."""
    st = np.stack(slices, 0)
    lo, hi = np.percentile(st[:, ::4, ::4], [1, 99])
    if hi <= lo:
        return [np.zeros_like(s) for s in slices]
    return [np.clip((s - lo) / (hi - lo), 0, 1) for s in slices]


def _tissue_centre(arr):
    """Intensity-weighted centroid of the imaged tissue, in pixels (row, col).

    Background in knee MRI is near-zero air, so thresholding at the slice's own Otsu-free midpoint
    (mean of the 60th and 99th percentile) and taking the centroid of what remains lands on the leg.
    Used only by CROP_CENTRE == "tissue": the geometric centre of the frame is a median 13 mm away
    from this point (p90 30 mm), which at a 140 mm field of view pushes the joint line off centre."""
    lo = np.percentile(arr, 60.0)
    hi = np.percentile(arr, 99.0)
    thr = 0.5 * (lo + hi)
    m = arr >= thr
    if not m.any():
        return (arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0
    w = np.where(m, arr, 0.0)
    tot = w.sum()
    if tot <= 0:
        return (arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0
    rows = (w.sum(1) * np.arange(arr.shape[0])).sum() / tot
    cols = (w.sum(0) * np.arange(arr.shape[1])).sum() / tot
    return float(rows), float(cols)


def _crop_origin(arr, rpx, cpx):
    """Top-left corner of an rpx x cpx crop, honouring CROP_CENTRE, clamped inside the frame."""
    h, w = arr.shape
    if CROP_CENTRE == "tissue":
        cy, cx = _tissue_centre(arr)
        r0 = int(round(cy - rpx / 2.0))
        c0 = int(round(cx - cpx / 2.0))
        return max(0, min(r0, max(0, h - rpx))), max(0, min(c0, max(0, w - cpx)))
    return max(0, (h - rpx) // 2), max(0, (w - cpx) // 2)


def _read_slice(path, want_mm=CROP_MM, out=OUT_SIZE):
    import pydicom
    ds=pydicom.dcmread(path)
    arr=ds.pixel_array.astype(np.float32)
    # physical-mm center crop using PixelSpacing (row,col mm/px). Falls back to full frame.
    ps=getattr(ds,"PixelSpacing",None)
    if ps:
        rmm,cmm=float(ps[0]),float(ps[1])
        rpx=int(round(want_mm/rmm)); cpx=int(round(want_mm/cmm))
        r0,c0=_crop_origin(arr,rpx,cpx)
        arr=arr[r0:r0+rpx, c0:c0+cpx]
    # robust intensity normalize (per-slice percentile)
    lo,hi=np.percentile(arr,[1,99])
    if hi>lo: arr=np.clip((arr-lo)/(hi-lo),0,1)
    else: arr=np.zeros_like(arr)
    # resize
    import cv2
    arr=cv2.resize(arr,(out,out),interpolation=_interp(*arr.shape[:2],out))
    return arr

def build_study_stack(study, series_idx, budget=PLANE_BUDGET):
    """Returns np array [C, H, W] where C = sum(budget) sampled slices, or None if unreadable."""
    series=series_idx.get(study,[])
    by_plane={}
    for suid,fluid,fat,plane in series:
        by_plane.setdefault(plane,[]).append(suid)
    chans=[]
    for plane,n in budget.items():
        suids=by_plane.get(plane,[])
        if not suids: 
            chans.extend([np.zeros((OUT_SIZE,OUT_SIZE),np.float32)]*n); continue
        # pick the series with the most slices for this plane (usually the main acquisition)
        best=None; best_files=[]
        for suid in suids:
            d=os.path.join(DATA_ROOT,"train_series",study,suid)
            fs=sorted(os.listdir(d)) if os.path.isdir(d) else []
            if len(fs)>len(best_files): best,best_files=suid,fs
        if not best_files:
            chans.extend([np.zeros((OUT_SIZE,OUT_SIZE),np.float32)]*n); continue
        # evenly sample n slice indices across the series
        idxs=np.linspace(0,len(best_files)-1,n).round().astype(int)
        d=os.path.join(DATA_ROOT,"train_series",study,best)
        for i in idxs:
            try: chans.append(_read_slice(os.path.join(d,best_files[i])))
            except Exception: chans.append(np.zeros((OUT_SIZE,OUT_SIZE),np.float32))
    return np.stack(chans,0)  # [64,384,384]

# ----------------------------------------------------------------------------------------------
# v2 stack (2026-09-30 19:20 UTC). v1 above has a real bug: files are sorted by FILENAME, and the
# filenames are anonymised SOPInstanceUIDs with random suffixes, so slices were SHUFFLED within each
# series (InstanceNumber in filename order: [13, 7, 16, 24, 4, ...]). That breaks the 3-slice 2.5D
# windows and the bi-LSTM, and turns "evenly spaced" sampling into a random subset.
# v2 fixes the order (patient-space position along the slice normal) and adopts the reference
# 0.924 layout: primary + secondary sequence for sagittal and coronal, plus axial = 64 slices.
# ----------------------------------------------------------------------------------------------
TRAIN_SERIES_ROOT = os.path.join(DATA_ROOT, "train_series")
# (plane, rank, n_slices); rank 0 = primary (fluid-sensitive preferred), 1 = secondary
LAYOUT_V2 = [("Sagittal", 0, 18), ("Sagittal", 1, 14), ("Coronal", 0, 12), ("Coronal", 1, 8), ("Axial", 0, 12)]
# v3 (2026-10-03, A/B only): same 64-slice budget, moved OUT of sagittal (32 -> 20) INTO coronal
# (20 -> 26) and axial (12 -> 18). Why: our weakest findings vs their labels are MCL and Lateral
# Meniscus. Collateral ligaments are best seen on coronal images (CoPAS, Nat Commun 2024), and a DL
# plane ablation found axial "markedly outperformed" sagittal/coronal for meniscal tears (Yang et
# al., QIMS 2025); MRNet's learned plane weights also favoured axial PD for meniscal tears (PLOS Med
# 2018). Counter-evidence: radiologists showed no significant plane difference (AJR 2004), hence an
# A/B, not a default. Picked series hold median 56 coronal / 36 axial / 60 sagittal slices, so v2
# samples coronal and axial most sparsely.
LAYOUT_V3 = [("Sagittal", 0, 12), ("Sagittal", 1, 8), ("Coronal", 0, 16), ("Coronal", 1, 10), ("Axial", 0, 18)]
# s3 (2026-10-06, A/B only): same 64-slice budget and the same plane split as v2, but each plane is
# divided over THREE ranks instead of two, so a study with a third usable series in a plane feeds it
# instead of discarding it (14.7% of all series are currently dropped by MAX_SERIES_PER_PLANE=2).
# Per plane the v2 totals are preserved exactly: sagittal 32 = 12+10+10, coronal 20 = 8+6+6, axial 12 = 6+3+3.
# A plane with fewer usable series than ranks collapses to the ranks it has (existing code path).
LAYOUT_V2_S3 = [("Sagittal", 0, 12), ("Sagittal", 1, 10), ("Sagittal", 2, 10),
                ("Coronal", 0, 8), ("Coronal", 1, 6), ("Coronal", 2, 6),
                ("Axial", 0, 6), ("Axial", 1, 3), ("Axial", 2, 3)]


def _slice_key(path):
    """Anatomical position of a slice: projection of ImagePositionPatient onto the slice normal,
    sign-normalised on the normal's dominant axis so the direction is consistent in patient space
    across scanners. Falls back to InstanceNumber."""
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if ipp is not None and iop is not None and len(iop) == 6:
        n = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
        ax = int(np.argmax(np.abs(n)))
        if n[ax] < 0:
            n = -n
        return (0, float(np.dot(n, np.asarray(ipp, float))))
    inst = getattr(ds, "InstanceNumber", None)
    return (1, float(inst) if inst is not None else 0.0)


def ordered_files(d):
    """Files of one series in anatomical order (header-only reads)."""
    keyed = []
    for f in sorted(os.listdir(d)):
        try:
            keyed.append((_slice_key(os.path.join(d, f)), f))
        except Exception:
            keyed.append(((2, 0.0), f))
    keyed.sort()
    return [f for _, f in keyed]


_SERIES_TAX = None


def _series_family(suid):
    """Sequence family for a series uid, from the prebuilt train/series_taxonomy.json (or None)."""
    global _SERIES_TAX
    if _SERIES_TAX is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "series_taxonomy.json")
        try:
            with open(p) as f:
                _SERIES_TAX = json.load(f)
        except Exception:
            _SERIES_TAX = {}
    return _SERIES_TAX.get(suid)


def _pick_series(study, cands, split_root):
    """cands: [(suid, fluid_flag)] for one plane -> [primary, secondary?] as (suid, is_fluid, n_files).
    Primary prefers a fluid-sensitive sequence (effusion, marrow oedema, tears), then most slices;
    secondary prefers the opposite fluid flag (anatomical sequence), then most slices.

    SECONDARY_TIEBREAK == "slices" (default, unchanged): the secondary is the opposite-flag series with
    the MOST SLICES. Measured consequence of that rule: the "fluid" flag is really a fat-sat flag, so the
    non-fat-sat slot is contested by T1, T2 and PD together; T1 usually has more slices, so T2 loses it
    31.3% of the time and 418 studies lose their T2 entirely. Those studies score -0.028 macro AUC versus
    studies that keep it (three independent 5-fold runs, confound-controlled within vendor and within
    cap-bites studies), concentrated on Lateral OA -0.080, PF OA -0.068, Fracture -0.053.
    SECONDARY_TIEBREAK == "diverse": among opposite-flag candidates, prefer a T2/STIR family over another
    T1, then fall back to slice count. Costs nothing: same number of series, same slice budget."""
    info = []
    for suid, fluid in cands:
        d = os.path.join(split_root, study, suid)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        if n:
            info.append((suid, str(fluid) == "1", n))
    if not info:
        return []
    info.sort(key=lambda t: (not t[1], -t[2]))
    prim, rest = info[0], info[1:]
    if SECONDARY_TIEBREAK == "diverse":
        # rank 0 = a fluid-ish family we would otherwise lose (T2/T2-FS/STIR), 1 = anything else
        def _div(t):
            fam = _series_family(t[0])
            return (t[1] == prim[1], 0 if fam in ("T2", "T2-FS", "STIR") else 1, -t[2])
        rest.sort(key=_div)
    else:
        rest.sort(key=lambda t: (t[1] == prim[1], -t[2]))
    return [prim] + rest[:MAX_SERIES_PER_PLANE - 1]


def build_study_stack_v2(study, series_idx, split_root=TRAIN_SERIES_ROOT, layout=LAYOUT_V2):
    """[64, OUT_SIZE, OUT_SIZE] float32 in [0,1]. Plane blocks in layout order (Sag, Cor, Ax), each
    series anatomically ordered and evenly sampled. A plane with one usable series gives it the whole
    plane budget; a missing plane is zero-filled (same convention as v1)."""
    by_plane = {}
    for suid, fluid, fat, plane in series_idx.get(study, []):
        by_plane.setdefault(plane, []).append((suid, fluid))
    planes = []
    for p, _, _ in layout:
        if p not in planes:
            planes.append(p)
    chans = []
    for plane in planes:
        budgets = [n for p, _, n in layout if p == plane]
        picks = _pick_series(study, by_plane.get(plane, []), split_root)
        if not picks:
            chans.extend([np.zeros((OUT_SIZE, OUT_SIZE), np.float32)] * sum(budgets))
            continue
        if len(picks) < len(budgets):
            # Fewer usable series than ranks: redistribute this plane's whole budget over the series we
            # DO have, largest share first. With 2 ranks and 1 series this is the documented v2 behaviour
            # (one series takes the plane); with 3 ranks and 2 series it must still be a 2-WAY split, not
            # a collapse onto one series, or a 3-rank layout would degrade the 2-series majority.
            tot, k = sum(budgets), len(picks)
            budgets = [tot // k + (1 if i < tot % k else 0) for i in range(k)]
        for (suid, _, _), n in zip(picks, budgets):
            d = os.path.join(split_root, study, suid)
            fs = ordered_files(d)
            idxs = np.linspace(0, len(fs) - 1, n).round().astype(int)
            memo = {}
            if NORM_MODE == "series":
                raw, bad = {}, set()
                for i in dict.fromkeys(idxs.tolist()):
                    try:
                        raw[i] = _read_raw(os.path.join(d, fs[i]))
                    except Exception:
                        bad.add(i)
                good = [i for i in raw]
                if good:
                    memo = dict(zip(good, _norm_series([raw[i] for i in good])))
                for i in bad:
                    memo[i] = np.zeros((OUT_SIZE, OUT_SIZE), np.float32)
            for i in idxs:
                if i not in memo:
                    try:
                        memo[i] = _read_slice(os.path.join(d, fs[i]))
                    except Exception:
                        memo[i] = np.zeros((OUT_SIZE, OUT_SIZE), np.float32)
                chans.append(memo[i])
    return np.stack(chans, 0)


def build_study_stack_v3(study, series_idx, split_root=TRAIN_SERIES_ROOT):
    """v2 builder with the coronal/axial-heavy LAYOUT_V3 (same [64, OUT_SIZE, OUT_SIZE] shape)."""
    return build_study_stack_v2(study, series_idx, split_root=split_root, layout=LAYOUT_V3)


# Native-adjacent 2.5D windows (2026-10-06). Our 2.5D window is built from channels that are ADJACENT
# IN THE DECIMATED STACK, not adjacent in the scanner. Measured: for single-series coronal/axial planes
# the sampled step is 2.9-3.0x the native slice gap, so a "3-slice window" spans ~30 mm of knee instead
# of ~11 mm and its three channels are not neighbouring anatomy. Every top solution samples the window
# CENTRES and takes NATIVE neighbours for the channels (RSNA-2022 cervical 1st, RSNA-2023 abdominal 1st,
# and the public 0.944 knee kernel). This builder keeps the same 64 window centres and the same layout,
# and emits [192, H, W] = 64 consecutive (j-1, j, j+1) native triplets. Model side: chan_mode="native".
# 96-slice layout (2026-10-06). The 3rd-series test (REJECTED 3:30 AM) showed the multi-series deficit
# cannot be fixed by SPLITTING a fixed 64-slice budget — it needs more capacity. This keeps the v2
# per-plane proportions (sag 50%, cor 31%, ax 19%) and the same primary/secondary split, scaled to 96:
# sagittal 27+21=48, coronal 18+12=30, axial 18. Evidence: Dread's measured ladder 44 slices 0.926 ->
# 64 wide 0.928 -> 80 slices 0.932, and the 0.944 kernel's coverage curve saturating near 31 windows
# per slot. Costs ~1.5x step time, so it is a capacity test, not a free win.
LAYOUT_V2_96 = [("Sagittal", 0, 27), ("Sagittal", 1, 21), ("Coronal", 0, 18), ("Coronal", 1, 12),
                ("Axial", 0, 18)]


def build_study_stack_v2_96(study, series_idx, split_root=TRAIN_SERIES_ROOT):
    """v2 builder at a 96-slice budget (LAYOUT_V2_96) -> [96, OUT_SIZE, OUT_SIZE]."""
    return build_study_stack_v2(study, series_idx, split_root=split_root, layout=LAYOUT_V2_96)


# 128-slice budget: same plane proportions as v2/v296 (sag 50% / cor 31% / ax 19%). Median series has ~30 slices,
# so 36+28 sagittal and 24+16 coronal now cover essentially every slice of the two series per plane.
LAYOUT_V2_128 = [("Sagittal", 0, 36), ("Sagittal", 1, 28), ("Coronal", 0, 24), ("Coronal", 1, 16),
                 ("Axial", 0, 24)]


def build_study_stack_v2_128(study, series_idx, split_root=TRAIN_SERIES_ROOT):
    """v2 builder at a 128-slice budget (LAYOUT_V2_128) -> [128, OUT_SIZE, OUT_SIZE]."""
    return build_study_stack_v2(study, series_idx, split_root=split_root, layout=LAYOUT_V2_128)


# 160-slice budget, same proportions (sag 50% / cor 31% / ax 19%). Above the median series length (~30) for every block,
# so most series are now UPSAMPLED along the slice axis; this is the step where the density lever should saturate.
LAYOUT_V2_160 = [("Sagittal", 0, 45), ("Sagittal", 1, 35), ("Coronal", 0, 30), ("Coronal", 1, 20),
                 ("Axial", 0, 30)]


def build_study_stack_v2_160(study, series_idx, split_root=TRAIN_SERIES_ROOT):
    """v2 builder at a 160-slice budget (LAYOUT_V2_160) -> [160, OUT_SIZE, OUT_SIZE]."""
    return build_study_stack_v2(study, series_idx, split_root=split_root, layout=LAYOUT_V2_160)


def build_study_stack_v2_native(study, series_idx, split_root=TRAIN_SERIES_ROOT, layout=LAYOUT_V2):
    """[3*sum(budget), OUT_SIZE, OUT_SIZE] float32 in [0,1]: for each sampled centre, its native
    (j-1, j, j+1) slices, clamped at series ends. Channel c of triplet t is 3*t + c."""
    by_plane = {}
    for suid, fluid, fat, plane in series_idx.get(study, []):
        by_plane.setdefault(plane, []).append((suid, fluid))
    planes = []
    for p, _, _ in layout:
        if p not in planes:
            planes.append(p)
    chans = []
    for plane in planes:
        budgets = [n for p, _, n in layout if p == plane]
        picks = _pick_series(study, by_plane.get(plane, []), split_root)
        if not picks:
            chans.extend([np.zeros((OUT_SIZE, OUT_SIZE), np.float32)] * (3 * sum(budgets)))
            continue
        if len(picks) < len(budgets):
            tot, k = sum(budgets), len(picks)
            budgets = [tot // k + (1 if i < tot % k else 0) for i in range(k)]
        for (suid, _, _), n in zip(picks, budgets):
            d = os.path.join(split_root, study, suid)
            fs = ordered_files(d)
            last = len(fs) - 1
            centres = np.linspace(0, last, n).round().astype(int)
            memo = {}
            for c in centres:
                for j in (c - 1, c, c + 1):
                    j = int(min(max(j, 0), last))      # clamp at the series ends
                    if j not in memo:
                        try:
                            memo[j] = _read_slice(os.path.join(d, fs[j]))
                        except Exception:
                            memo[j] = np.zeros((OUT_SIZE, OUT_SIZE), np.float32)
                for j in (c - 1, c, c + 1):
                    chans.append(memo[int(min(max(j, 0), last))])
    return np.stack(chans, 0)


def build_study_stack_v2_s3(study, series_idx, split_root=TRAIN_SERIES_ROOT):
    """v2 builder with three series ranks per plane (LAYOUT_V2_S3). Same [64, OUT_SIZE, OUT_SIZE] shape
    and the same per-plane slice totals as v2; only the within-plane split changes. Requires
    MAX_SERIES_PER_PLANE >= 3, which build_cache sets alongside this layout."""
    return build_study_stack_v2(study, series_idx, split_root=split_root, layout=LAYOUT_V2_S3)


LAT_THR_MM = 15.0   # |image-centre x| below this (some scanners re-centre on isocentre) => side unknown, no flip


def _series_center_x(path):
    """Patient-space x (DICOM LPS: +x = patient LEFT) of the centre of one slice, from header geometry only."""
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    ps = getattr(ds, "PixelSpacing", None)
    if ipp is None or iop is None or ps is None or len(iop) != 6:
        return None
    row, col = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    # IOP[0:3] = direction of increasing column index (step = PixelSpacing[1]); IOP[3:6] = increasing row (PixelSpacing[0])
    ctr = (np.asarray(ipp, float) + row * float(ps[1]) * int(ds.Columns) / 2.0
           + col * float(ps[0]) * int(ds.Rows) / 2.0)
    return float(ctr[0])


def study_side(study, series_idx, split_root=TRAIN_SERIES_ROOT, thr=LAT_THR_MM):
    """'L' / 'R' / None for which knee a study images, from scan geometry (the DICOM Laterality tag is empty
    in this dataset). Median over series of the image-centre x: knees sit ~60-120 mm off the midline.
    Measured on 251 training studies whose report names the side: 95.2% agreement at thr 0, 97.9% at 15 mm
    (18 undecided). Header reads only (one slice per series)."""
    xs = []
    for suid, _fluid, _fat, _plane in series_idx.get(study, []):
        d = os.path.join(split_root, study, suid)
        try:
            fs = sorted(os.listdir(d))
            x = _series_center_x(os.path.join(d, fs[len(fs) // 2])) if fs else None
        except Exception:
            x = None
        if x is not None:
            xs.append(x)
    if not xs:
        return None
    m = float(np.median(xs))
    return "L" if m > thr else "R" if m < -thr else None


def load_labels(soft_csv):
    """study -> np[12] soft labels. Gold rows (train.csv 0/1) override softs for those 58.

    The teacher labelled each UNIQUE report once, so labels_soft.csv has 4,273 rows for
    4,407 studies. The other 134 studies have a byte-identical report to a labelled one and
    inherit its labels (the teacher only ever saw the text, so the labels are identical by
    construction). Without this they would silently drop out of training."""
    soft={}
    for r in csv.DictReader(open(soft_csv,encoding='utf-8')):
        soft[r["StudyInstanceUID"]]=np.array([float(r[l]) for l in LABELS],np.float32)
    rows=list(csv.DictReader(open(TRAIN_CSV,encoding='utf-8')))
    by_report={r["Report"].strip(): soft[r["StudyInstanceUID"]]
               for r in rows if r["StudyInstanceUID"] in soft}
    for r in rows:
        if r["StudyInstanceUID"] not in soft and r["Report"].strip() in by_report:
            soft[r["StudyInstanceUID"]]=by_report[r["Report"].strip()].copy()
    gold={}
    for r in rows:
        if any(r[l].strip()!="" for l in LABELS):
            gold[r["StudyInstanceUID"]]=np.array([float(r[l]) for l in LABELS],np.float32)
    soft.update(gold)  # gold is truth, override
    return soft, set(gold)

if __name__=="__main__":
    # self-test the index without needing torch/pydicom fully wired
    idx=load_series_index()
    print("studies in series index:", len(idx))
    import collections
    planes=collections.Counter()
    for s,lst in idx.items():
        for _,_,_,p in lst: planes[p]+=1
    print("plane counts:", dict(planes))
    print("budget:", PLANE_BUDGET, "-> stack channels:", sum(PLANE_BUDGET.values()))
