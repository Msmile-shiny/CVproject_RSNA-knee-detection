"""RSNA Knee — offline inference library (runs inside the Kaggle notebook, internet OFF).

Mirrors training preprocessing EXACTLY:
  - stack  = dataset.build_study_stack_v2 (anatomically ordered, Sag 18+14 / Cor 12+8 / Ax 12)
  - quantise to uint8 exactly like the training cache, then /255 on the GPU
  - no resize when the model's train resolution equals the cache resolution (384)
Every study in test.csv gets a row; any failure falls back to the model output on a zero stack,
or 0.5 as a last resort, so the submission file is always complete.
fp16 autocast only — T4/P100 have no usable bf16.
"""
import os, sys, csv, json, time, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import (LABELS, OUT_SIZE, build_study_stack_v2, build_study_stack_v2_native, _pick_series,  # noqa: E402
                     LAYOUT_V2, LAYOUT_V2_96, LAYOUT_V2_128, study_side)

# Layouts the kernel can serve. Each model declares its "layout" in models.json (default "v2"); all models in
# one submission must share a layout because one stack is built per study. "sag" = sagittal block sizes in
# CHANNEL space, exactly what train.mirror_batch used (cache_v2_blocks.json / cache_v296_sagsizes.json /
# cache_v2nat_sagsizes.json), so the mirror below is identical to training. v2nat stores native (j-1,j,j+1)
# triplets -> 3 channels per window, 192 channels, and the model must be built with chan_mode="native".
LAYOUTS = {
    "v2":    {"layout": LAYOUT_V2,    "n": 64,  "sag": [18, 14],
              "build": lambda s, idx, root: build_study_stack_v2(s, idx, split_root=root)},
    "v296":  {"layout": LAYOUT_V2_96, "n": 96,  "sag": [27, 21],
              "build": lambda s, idx, root: build_study_stack_v2(s, idx, split_root=root, layout=LAYOUT_V2_96)},
    "v2128": {"layout": LAYOUT_V2_128, "n": 128, "sag": [36, 28],
              "build": lambda s, idx, root: build_study_stack_v2(s, idx, split_root=root, layout=LAYOUT_V2_128)},
    "v2nat": {"layout": LAYOUT_V2,    "n": 192, "sag": [54, 42],
              "build": lambda s, idx, root: build_study_stack_v2_native(s, idx, split_root=root)},
}


def n_sagittal_series(study, idx, split_root, layout="v2"):
    """How many sagittal series the builder used (same selection code) -> mirror block layout."""
    cands = [(suid, fluid) for suid, fluid, fat, plane in idx.get(study, []) if plane == "Sagittal"]
    budgets = [n for p, _, n in LAYOUTS[layout]["layout"] if p == "Sagittal"]
    return min(len(_pick_series(study, cands, split_root)), len(budgets))


def mirror_stack(x, n_sag, layout="v2"):
    """Label-preserving left-right mirror of a [B,N,H,W] stack (identical to train.mirror_batch at p=1):
    coronal + axial flipped horizontally; sagittal slice ORDER reversed per series block.
    v2: sag 18+14 (or 32 when one series), cor/ax from channel 32. v296: sag 27+21 (or 48), cor/ax from 48."""
    L = LAYOUTS[layout]
    sag = L["sag"]
    nsag = sum(sag)
    x = x.clone()
    x[:, nsag:] = x[:, nsag:].flip(-1)
    for b in range(x.shape[0]):
        if int(n_sag[b]) >= 2:
            o = 0
            for n in sag:
                x[b, o:o + n] = x[b, o:o + n].flip(0)
                o += n
        else:
            x[b, 0:nsag] = x[b, 0:nsag].flip(0)
    return x


def lat_norm_stack(x, n_sag, is_left, layout="v2"):
    """Laterality normalisation for models trained with --lat-norm: mirror only the LEFT knees
    (identical to train.lat_norm_batch). is_left: per-sample 0/1."""
    x = x.clone()
    for b in range(x.shape[0]):
        if int(is_left[b]) != 1:
            continue
        x[b:b + 1] = mirror_stack(x[b:b + 1], n_sag[b:b + 1], layout)
    return x


def find_dir(marker, root="/kaggle/input"):
    """First directory under root containing `marker` (mount paths differ between Kaggle images)."""
    hits = sorted(glob.glob(os.path.join(root, "**", marker), recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f"{marker} not found under {root}")
    return os.path.dirname(hits[0])


def load_index(series_csv):
    idx = {}
    for r in csv.DictReader(open(series_csv, encoding="utf-8")):
        idx.setdefault(r["StudyInstanceUID"], []).append(
            (r["SeriesInstanceUID"], r["Fluid_Sensitive"], r["Fat_Suppression"], r["Anatomical_Plane"]))
    return idx


def quantise(x):
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def make_loader(studies, idx, split_root, workers, bs, need_side=False, layouts=("v2",)):
    """One stack PER LAYOUT per study (a submission may mix 64-slice and 96-slice models). Returns
    ({layout: uint8 stack}, i, ok, {layout: n_sag}, left). ok=0 if ANY layout's stack failed."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    layouts = list(layouts)

    class DS(Dataset):
        def __len__(self):
            return len(studies)

        def __getitem__(self, i):
            ok, stacks, nss = 1, {}, {}
            for lay in layouts:
                L = LAYOUTS[lay]
                try:
                    stacks[lay] = torch.from_numpy(quantise(L["build"](studies[i], idx, split_root)))
                except Exception:
                    stacks[lay], ok = torch.zeros((L["n"], OUT_SIZE, OUT_SIZE), dtype=torch.uint8), 0
                try:
                    nss[lay] = n_sagittal_series(studies[i], idx, split_root, lay)
                except Exception:
                    nss[lay] = 2
            left = 0
            if need_side:                      # header reads only; unknown side -> 0 (left as is), as in training
                try:
                    left = int(study_side(studies[i], idx, split_root=split_root) == "L")
                except Exception:
                    left = 0
            return stacks, i, ok, nss, left

    return DataLoader(DS(), batch_size=bs, num_workers=workers, pin_memory=True)


def load_models(models_dir, dev):
    import torch
    from model import KneeModel
    spec = json.load(open(os.path.join(models_dir, "models.json")))
    models = []
    for m in spec["models"]:
        net = KneeModel(m["backbone"], pretrained=False, grad_ckpt=False, img_size=m.get("img"),
                        pool=m.get("pool", "lstm_attn"), chan_mode=m.get("chan_mode", "adjacent"))
        sd = torch.load(os.path.join(models_dir, m["weights"]), map_location="cpu")
        net.load_state_dict({k: v.float() for k, v in sd.items()})
        models.append((m, net.to(dev).eval()))
    return spec, models


def predict(comp_dir, models_dir, out_csv, split="test", workers=None, bs=2, limit=None):
    import torch
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    studies = [r["StudyInstanceUID"] for r in csv.DictReader(open(os.path.join(comp_dir, f"{split}.csv")))]
    if limit:
        studies = studies[:limit]
    idx = load_index(os.path.join(comp_dir, f"{split}_series.csv"))
    split_root = os.path.join(comp_dir, f"{split}_series")
    spec, models = load_models(models_dir, dev)
    print(f"[infer] {len(studies)} studies, {len(models)} models, device={dev}", flush=True)

    workers = workers if workers is not None else min(4, os.cpu_count() or 1)
    P = np.full((len(studies), len(LABELS)), np.nan, np.float32)
    failed = 0
    need_side = any(m.get("lat_norm") for m, _ in models)
    layouts = sorted({m.get("layout", "v2") for m, _ in models})
    for lay in layouts:
        assert lay in LAYOUTS, f"unknown layout {lay}"
    print(f"[infer] layouts={layouts} ({[LAYOUTS[l]['n'] for l in layouts]} slices)", flush=True)
    with torch.no_grad():
        for xs, i, ok, nss, left in make_loader(studies, idx, split_root, workers, bs, need_side=need_side,
                                                layouts=layouts):
            failed += int((ok == 0).sum())
            # per-layout plain / mirrored / lat-normalised views, built once per batch
            V = {}
            for lay in layouts:
                x = xs[lay].to(dev, non_blocking=True).float().div_(255.0)
                ns = nss[lay]
                xm = (mirror_stack(x, ns, lay)
                      if any(m.get("tta") and not m.get("lat_norm") and m.get("layout", "v2") == lay
                             for m, _ in models) else None)
                xl = lat_norm_stack(x, ns, left, lay) if need_side else None
                V[lay] = (x, xm, xl)
            nb = next(iter(xs.values())).shape[0]
            acc = torch.zeros(nb, len(LABELS), device=dev)
            for m, net in models:
                x, xm, xl = V[m.get("layout", "v2")]
                if m.get("lat_norm"):
                    views = [xl]                               # trained on right-knee-only inputs: no mirror TTA
                else:
                    views = [x, xm] if m.get("tta") else [x]   # "tta": true -> average plain + mirrored
                pv = 0
                for v in views:
                    xi = v
                    if m.get("img") and m["img"] != v.shape[-1]:
                        xi = torch.nn.functional.interpolate(xi, size=(m["img"], m["img"]), mode="bilinear",
                                                             align_corners=False, antialias=True)
                    with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                        logits, _ = net(xi)
                    pv = pv + torch.sigmoid(logits.float())
                acc += (pv / len(views)) * float(m.get("weight", 1.0))
            wsum = sum(float(m.get("weight", 1.0)) for m, _ in models)
            P[i.numpy()] = (acc / wsum).cpu().numpy()
    P = np.where(np.isnan(P), 0.5, P)

    header = ["StudyInstanceUID"] + LABELS
    ss = os.path.join(comp_dir, "sample_submission.csv")
    if os.path.exists(ss):
        header = next(csv.reader(open(ss, encoding="utf-8")))
    col = {l: j for j, l in enumerate(LABELS)}
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s, p in zip(studies, P):
            w.writerow([s] + [f"{float(p[col[h]]):.6f}" for h in header[1:]])
    print(f"[infer] wrote {out_csv}: {len(studies)} rows, stack failures={failed}, "
          f"{time.time() - t0:.1f}s", flush=True)
    return studies, P
