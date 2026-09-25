"""Offline fusion scan: 3-seed rank-mean base × spec expert (58 gold, CPU only).

5 scenarios:
  S1  baseline: pure 3-seed rank-mean (expect 0.8959)
  S2  current 08_infer_fusion config: 4-class 0.75/0.75 rank replacement
  S3  ACL-only replacement, weight scan w in {0.25, 0.5, 0.75, 1.0}
  S4  spec as 4th member, low-weight rank-mean over ALL 12 classes
  S5  per-class oracle: best w per class (upper bound, overfits 58 gold)

Note on weights (matches 08_infer_fusion.py semantics):
  final = w * base + w * spec_rank  ->  w=0.75 means both scaled 0.75.
  Rank-mean scale does not change AUC per class, only the mix ratio matters.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
SPEC_REPLACE_TARGETS = ["ACL", "MCL", "Lateral Meniscus", "Lateral OA"]

SEED_FILES = {
    "s42": RESULTS / "v5s1" / "gold_validation_predictions_s42.csv",
    "s142": RESULTS / "v5s2" / "gold_validation_predictions_s142.csv",
    "s242": RESULTS / "v5s3" / "gold_validation_predictions_s242.csv",
}
SPEC_FILE = RESULTS / "v5spec" / "gold_validation_predictions_spec.csv"


def load_member(path):
    df = pd.read_csv(path)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df = df.set_index("StudyInstanceUID")
    probs = np.stack([df[f"prob_{c}"].to_numpy() for c in TARGETS], axis=1)
    y = np.stack([df[f"true_{c}"].to_numpy() for c in TARGETS], axis=1).astype(int)
    return df.index.to_numpy(), y, probs


def macro_auc(y, probs):
    aucs = []
    for j in range(len(TARGETS)):
        yj = y[:, j]
        if len(np.unique(yj)) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(yj, probs[:, j]))
    return np.array(aucs)


def per_class_aucs(y, probs):
    return macro_auc(y, probs)


def main():
    uids_list, y_list, p_list = [], [], []
    for key, path in SEED_FILES.items():
        uids, y, p = load_member(path)
        uids_list.append(uids)
        y_list.append(y)
        p_list.append(p)
    spec_uids, y_spec, p_spec = load_member(SPEC_FILE)

    assert all((u == uids_list[0]).all() for u in uids_list[1:]), "seed UID mismatch"
    assert (spec_uids == uids_list[0]).all(), "spec UID mismatch"
    y = y_list[0]
    n, ncls = y.shape

    # ---- ranks per member (mean rank, ties averaged) ----
    def ranks(p):
        return np.stack([rankdata(p[:, j]) for j in range(ncls)], axis=1)

    seed_ranks = [ranks(p) for p in p_list]
    spec_rank = ranks(p_spec)
    base = np.mean(seed_ranks, axis=0)  # rank-mean of 3 seeds

    base_auc = per_class_aucs(y, base)
    spec_auc = per_class_aucs(y, spec_rank)
    print(f"gold studies: {n}")
    print(f"{'target':20s} {'base':>7s} {'spec':>7s} {'d(spec-base)':>12s}")
    for j, c in enumerate(TARGETS):
        print(f"{c:20s} {base_auc[j]:7.4f} {spec_auc[j]:7.4f} {spec_auc[j]-base_auc[j]:+12.4f}")
    print(f"{'MACRO':20s} {np.nanmean(base_auc):7.4f} {np.nanmean(spec_auc):7.4f}")

    # ---- S1 baseline ----
    print("\n=== S1 baseline (pure 3-seed rank-mean) ===")
    print(f"macro AUC = {np.nanmean(base_auc):.4f}")

    # ---- S2 current config: 4-class 0.75/0.75 replacement ----
    print("\n=== S2 current 08_infer_fusion config (4-class, w=0.75 both sides) ===")
    final = base.copy()
    for j, c in enumerate(TARGETS):
        if c in SPEC_REPLACE_TARGETS:
            final[:, j] = 0.75 * base[:, j] + 0.75 * spec_rank[:, j]
    s2_auc = per_class_aucs(y, final)
    for j, c in enumerate(TARGETS):
        if c in SPEC_REPLACE_TARGETS:
            print(f"  {c:20s} {base_auc[j]:.4f} -> {s2_auc[j]:.4f} ({s2_auc[j]-base_auc[j]:+.4f})")
    print(f"macro AUC = {np.nanmean(s2_auc):.4f} (delta {np.nanmean(s2_auc)-np.nanmean(base_auc):+.4f})")

    # ---- S3 ACL-only replacement, weight scan ----
    print("\n=== S3 ACL-only replacement, weight scan ===")
    j_acl = TARGETS.index("ACL")
    for w in [0.25, 0.5, 0.75, 1.0]:
        final = base.copy()
        final[:, j_acl] = w * base[:, j_acl] + w * spec_rank[:, j_acl]
        auc = per_class_aucs(y, final)
        print(f"  w={w:<4} ACL {base_auc[j_acl]:.4f} -> {auc[j_acl]:.4f} "
              f"({auc[j_acl]-base_auc[j_acl]:+.4f})  | macro {np.nanmean(auc):.4f} "
              f"({np.nanmean(auc)-np.nanmean(base_auc):+.4f})")

    # ---- S4 spec as 4th member, low-weight rank-mean over all 12 classes ----
    print("\n=== S4 spec as 4th member (rank-mean, weight scan on spec) ===")
    for w in [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]:
        final = (3.0 * base + w * spec_rank) / (3.0 + w)
        auc = per_class_aucs(y, final)
        print(f"  w={w:<4} macro {np.nanmean(auc):.4f} ({np.nanmean(auc)-np.nanmean(base_auc):+.4f})")

    # ---- S5 per-class oracle (upper bound; overfits 58 gold) ----
    print("\n=== S5 per-class oracle (best w per class, upper bound) ===")
    ws = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    oracle = base.copy()
    best_w = {}
    for j, c in enumerate(TARGETS):
        best = (0.0, base_auc[j])
        for w in ws:
            col = base[:, j] if w == 0.0 else w * (base[:, j] + spec_rank[:, j])
            a = roc_auc_score(y[:, j], col)
            if a > best[1]:
                best = (w, a)
        best_w[c] = best
        w = best[0]
        oracle[:, j] = base[:, j] if w == 0.0 else w * (base[:, j] + spec_rank[:, j])
    o_auc = per_class_aucs(y, oracle)
    for j, c in enumerate(TARGETS):
        w, a = best_w[c]
        print(f"  {c:20s} w={w:<4} {base_auc[j]:.4f} -> {o_auc[j]:.4f} ({o_auc[j]-base_auc[j]:+.4f})")
    print(f"oracle macro AUC = {np.nanmean(o_auc):.4f} (delta {np.nanmean(o_auc)-np.nanmean(base_auc):+.4f})")


if __name__ == "__main__":
    main()
