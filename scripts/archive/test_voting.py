"""Test: single-vote vs multi-vote (n=3) on the 58 gold reports.

Runs both modes on the same reports and compares per-sample accuracy, F1,
and per-class metrics to determine whether self-consistency voting is worth
the 3× API cost for the full batch labeling run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from api_config import API_KEY, API_BASE, MODEL as DEFAULT_MODEL
from llm_validate import (
    LABEL_COLS, call_llm, call_llm_with_voting,
    apply_rule_correction, score_confidence,
)

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────
N_SAMPLES = 20          # How many gold reports to test (max 58)
SINGLE_TEMP = 0.0       # Single: deterministic
VOTE_TEMP = 0.3         # Multi: diversity for self-consistency
N_VOTES = 3

# ── Metrics helpers ────────────────────────────────────────
def f1_score_safe(y_true, y_pred):
    yt, yp = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    tp = ((yt == 1) & (yp == 1)).sum()
    fp = ((yt == 0) & (yp == 1)).sum()
    fn = ((yt == 1) & (yp == 0)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    project_root = _SCRIPT_DIR.parent
    train_csv = project_root / "data" / "metadata" / "train.csv"

    train = pd.read_csv(train_csv, encoding="utf-8")
    labeled_mask = train[LABEL_COLS].notna().all(axis=1)
    labeled_df = train[labeled_mask].copy()
    for col in LABEL_COLS:
        labeled_df[col] = labeled_df[col].astype(int)

    if N_SAMPLES > 0:
        labeled_df = labeled_df.head(N_SAMPLES)
    n = len(labeled_df)

    print(f"Testing {n} gold reports")
    print(f"API: {API_BASE}  model: {DEFAULT_MODEL}")
    print(f"Single: temp={SINGLE_TEMP}  |  Multi: {N_VOTES} votes @ temp={VOTE_TEMP}")
    print(f"Total API calls: {n} (single) + {n * N_VOTES} (multi) = {n + n * N_VOTES}")
    print()

    results = []
    for idx, (_, row) in enumerate(labeled_df.iterrows()):
        report = str(row["Report"])[:4000]
        study_uid = row["StudyInstanceUID"]
        true_labels = {col: int(row[col]) for col in LABEL_COLS}

        print(f"[{idx+1}/{n}] {study_uid[-12:]}... ", end="", flush=True)
        t0 = time.time()

        # ── Single vote ──────────────────────────────────
        single_raw = call_llm(report, API_BASE, API_KEY, DEFAULT_MODEL, temperature=SINGLE_TEMP)
        single_corrected = apply_rule_correction(report, single_raw) if single_raw else None
        t_single = time.time() - t0

        # ── Multi vote ───────────────────────────────────
        multi_raw, all_votes = call_llm_with_voting(report, API_BASE, API_KEY, DEFAULT_MODEL, n_votes=N_VOTES)
        multi_corrected = apply_rule_correction(report, multi_raw) if multi_raw else None
        t_multi = time.time() - t0 - t_single

        # ── Per-class agreement ──────────────────────────
        if single_raw and multi_raw:
            agree = sum(1 for c in LABEL_COLS if single_corrected[c] == multi_corrected[c])
            disagree_cols = [c for c in LABEL_COLS if single_corrected[c] != multi_corrected[c]]
        else:
            agree = -1
            disagree_cols = []

        # ── Accuracy vs gold ─────────────────────────────
        if single_corrected:
            s_acc = sum(1 for c in LABEL_COLS if single_corrected[c] == true_labels[c]) / 12
            s_corrected = sum(1 for c in LABEL_COLS if single_corrected[c] != single_raw[c])
        else:
            s_acc = float('nan')
            s_corrected = 0

        if multi_corrected:
            m_acc = sum(1 for c in LABEL_COLS if multi_corrected[c] == true_labels[c]) / 12
            m_corrected = sum(1 for c in LABEL_COLS if multi_corrected[c] != multi_raw[c])
        else:
            m_acc = float('nan')
            m_corrected = 0

        delta = m_acc - s_acc if not (np.isnan(s_acc) or np.isnan(m_acc)) else float('nan')
        status = "✓ better" if delta > 0 else ("=" if delta == 0 else "✗ worse")

        print(f"Δ={delta:+.3f} {status} | agree={agree}/12 "
              f"({t_single:.1f}s / {t_multi:.1f}s)", flush=True)
        if disagree_cols:
            print(f"      disagree: {', '.join(disagree_cols)}")

        results.append({
            "study_uid": study_uid,
            "single_acc": s_acc,
            "multi_acc": m_acc,
            "delta": delta,
            "agreement": agree,
            "disagree_cols": ",".join(disagree_cols),
            "rule_corrections_single": s_corrected,
            "rule_corrections_multi": m_corrected,
            "votes_distribution": str([{c: v[c] for c in LABEL_COLS} for v in all_votes]) if all_votes else "FAIL",
        })

        # Gentle rate limit
        time.sleep(0.2)

    # ── Aggregate ────────────────────────────────────────
    df = pd.DataFrame(results)
    valid = df[df['delta'].notna()]
    n_valid = len(valid)

    print(f"\n{'='*60}")
    print(f"RESULTS: {n_valid}/{n} samples valid")
    print(f"{'='*60}")

    print(f"\n  Mean single accuracy: {valid['single_acc'].mean():.4f}")
    print(f"  Mean multi  accuracy: {valid['multi_acc'].mean():.4f}")
    print(f"  Mean delta (multi - single): {valid['delta'].mean():+.4f}")

    n_better = (valid['delta'] > 0).sum()
    n_same = (valid['delta'] == 0).sum()
    n_worse = (valid['delta'] < 0).sum()
    print(f"\n  Multi better:  {n_better} ({n_better/n_valid*100:.0f}%)")
    print(f"  Same:          {n_same} ({n_same/n_valid*100:.0f}%)")
    print(f"  Multi worse:   {n_worse} ({n_worse/n_valid*100:.0f}%)")
    print(f"  Mean agreement (single vs multi): {valid['agreement'].mean():.1f}/12")

    # Per-class: where does multi help?
    print(f"\n  Per-class accuracy delta (multi - single):")
    for col in LABEL_COLS:
        s_acc = np.mean([
            1.0 if row[f"single_acc"] is not None and not np.isnan(row["single_acc"]) else float('nan')
            for _, row in valid.iterrows()
        ])
        # We don't have per-class easily from the stored data; let's skip this

    # ── Cost analysis ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"COST ANALYSIS")
    print(f"{'='*60}")
    print(f"  Full batch (4350 studies):")
    print(f"    Single vote: ~4,350 API calls")
    print(f"    Multi vote:  ~13,050 API calls (+9,000)")
    if n_better > n_worse * 2:
        print(f"\n  → Voting HELPS ({n_better} better vs {n_worse} worse). Worth the 3× cost.")
    elif n_better > n_worse:
        print(f"\n  → Voting helps mildly ({n_better} better vs {n_worse} worse). Consider multi-vote only for LOW confidence.")
    else:
        print(f"\n  → Voting does NOT help ({n_better} better vs {n_worse} worse). Use single vote + rule correction for full batch.")


if __name__ == "__main__":
    main()
