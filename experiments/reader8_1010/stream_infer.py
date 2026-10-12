"""Predict the 12 findings for every study straight from DICOM folders.

Apex Grandmaster V5: Multi-Scale Spatial Dual-Resolution (256px + 320px) +
Dual-Stride Interleaved Volumetric Sampling (Offset 0 and Offset +1) +
Bilateral Horizontal Mirror TTA for complete invariant 3D knee joint coverage.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from knee import LABELS, N_SLOTS, KneeNet, build_study_table, window_centres
from preprocess import MAX_SLICES, SIZE, load_series


class DicomStudyDataset(torch.utils.data.Dataset):
    """Loads every selected series of a study once; windows are cut per model setting later."""

    def __init__(self, studies, table, root, k, offset=0):
        self.studies, self.table, self.root, self.k, self.offset = list(studies), table, root, k, offset

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((N_SLOTS, self.k, 3, SIZE, SIZE), np.uint8)
        mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            try:
                vol, _ = load_series(f"{self.root}/{st}/{cands[0]}")
            except Exception:
                continue
            n = len(vol)
            if n < 3:
                continue
            c = window_centres(n, self.k, False)
            if self.offset:
                c = np.clip(c + self.offset, 1, max(1, n - 2))
            x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
            mask[s], pos[s] = True, c / max(n - 1, 1)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)


class PairedDicomStudyDataset(torch.utils.data.Dataset):
    """Decode each selected series once; reproduce both original offset views."""
    def __init__(self, studies, table, root, k):
        self.studies, self.table, self.root, self.k = list(studies), table, root, k

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((2, N_SLOTS, self.k, 3, SIZE, SIZE), np.uint8)
        mask = np.zeros((2, N_SLOTS), bool)
        pos = np.zeros((2, N_SLOTS, self.k), np.float32)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            try:
                vol, _ = load_series(f'{self.root}/{st}/{cands[0]}')
            except Exception:
                continue
            n = len(vol)
            if n < 3:
                continue
            centres = window_centres(n, self.k, False)
            for offset in (0, 1):
                c = centres if offset == 0 else np.clip(centres + 1, 1, max(1, n - 2))
                x[offset, s] = np.stack([vol[np.clip([j-1,j,j+1],0,n-1)] for j in c])
                mask[offset, s], pos[offset, s] = True, c / max(n-1,1)
        return tuple((torch.from_numpy(x[o]),torch.from_numpy(mask[o]),torch.from_numpy(pos[o])) for o in (0,1))


def load_models(ckpts, device="cuda"):
    models = []
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False, img_size=ck["args"]["res"])
        m.load_state_dict(ck["model"])
        models.append((m.to(device).to(memory_format=torch.channels_last).eval(), ck["args"]["res"]))
    return models


@torch.no_grad()
def run(data_dir, split, ckpts, k=16, studies=None, workers=4, bs=4):
    se = pd.read_csv(f"{data_dir}/{split}_series.csv")
    root = f"{data_dir}/{split}_series"
    if studies is None:
        studies = pd.read_csv(f"{data_dir}/{split}.csv").StudyInstanceUID.tolist()
    se = se[se.StudyInstanceUID.isin(set(studies))]
    n_files = {}
    for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
        d = f"{root}/{st}/{sr}"
        n_files[sr] = min(len(os.listdir(d)), MAX_SLICES) if os.path.isdir(d) else 0
    table = build_study_table(se, n_files)
    models = load_models(ckpts)
    
    # Dual-Stride Interleaved Volumetric Sampling (Offset 0 and Offset +1)
    dl = torch.utils.data.DataLoader(PairedDicomStudyDataset(studies, table, root, k), batch_size=bs,
                                     num_workers=workers, pin_memory=True)
    out = []
    for (x0, mask0, pos0), (x1, mask1, pos1) in dl:
        x0, mask0, pos0 = x0.cuda(non_blocking=True), mask0.cuda(), pos0.cuda()
        
        # Bilateral Horizontal Flip TTA
        x0_flip = torch.flip(x0, dims=[-1])

        with torch.autocast("cuda", dtype=torch.float16):
            # Stride 0 Multi-Scale (256px + 320px) with Bilateral TTA
            p0_256_orig = torch.stack([m(x0, mask0, pos0, res=256).float().sigmoid() for m, _ in models]).mean(0)
            p0_256_flip = torch.stack([m(x0_flip, mask0, pos0, res=256).float().sigmoid() for m, _ in models]).mean(0)
            p0_256 = 0.5 * p0_256_orig + 0.5 * p0_256_flip

            p0_320_orig = torch.stack([m(x0, mask0, pos0, res=320).float().sigmoid() for m, _ in models]).mean(0)
            p0_320_flip = torch.stack([m(x0_flip, mask0, pos0, res=320).float().sigmoid() for m, _ in models]).mean(0)
            p0_320 = 0.5 * p0_320_orig + 0.5 * p0_320_flip

            p0 = 0.5 * p0_256 + 0.5 * p0_320

            del x0, x0_flip, mask0, pos0
            x1, mask1, pos1 = x1.cuda(non_blocking=True), mask1.cuda(), pos1.cuda()
            x1_flip = torch.flip(x1, dims=[-1])
            # Stride 1 Multi-Scale (256px + 320px) with Bilateral TTA
            p1_256_orig = torch.stack([m(x1, mask1, pos1, res=256).float().sigmoid() for m, _ in models]).mean(0)
            p1_256_flip = torch.stack([m(x1_flip, mask1, pos1, res=256).float().sigmoid() for m, _ in models]).mean(0)
            p1_256 = 0.5 * p1_256_orig + 0.5 * p1_256_flip

            p1_320_orig = torch.stack([m(x1, mask1, pos1, res=320).float().sigmoid() for m, _ in models]).mean(0)
            p1_320_flip = torch.stack([m(x1_flip, mask1, pos1, res=320).float().sigmoid() for m, _ in models]).mean(0)
            p1_320 = 0.5 * p1_320_orig + 0.5 * p1_320_flip

            p1 = 0.5 * p1_256 + 0.5 * p1_320

            # Volumetric Dual-Stride Blend
            p = 0.52 * p0 + 0.48 * p1
        del x1, x1_flip, mask1, pos1
        out.append(p.cpu())
    return pd.DataFrame(torch.cat(out).numpy(), index=pd.Index(studies, name="StudyInstanceUID"), columns=LABELS)


if __name__ == "__main__":
    data_dir, split, out_csv, ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    run(data_dir, split, ckpts).to_csv(out_csv)
    print("wrote", out_csv)
