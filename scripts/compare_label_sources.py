"""Compare an external report-label set with this project's training labels.

This script is deliberately read-only: it never replaces ``v5_labels.csv`` or
``pseudo_labels_calibrated.csv``.  It aligns studies by StudyInstanceUID and
writes diagnostics only.  External label CSVs may use either target names
directly (``ACL``) or the project's soft-label names (``prob_ACL``).

Example
-------
python scripts/compare_label_sources.py \
  --candidate /path/to/community_gpt56sol_labels.csv \
  --name gpt56sol

The report includes agreement on the 4,349 pseudo-labelled studies.  If the
candidate also contains the 58 gold studies, its AUC against the official
labels is reported separately.  That AUC is diagnostic only: do not tune a
prompt repeatedly against those same 58 studies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]
UID = "StudyInstanceUID"


def auc(y_true: pd.Series, score: pd.Series) -> float:
    """ROC-AUC from ranks; returns NaN when a target has one class only."""
    valid = y_true.notna() & score.notna()
    y = y_true[valid].astype(int).to_numpy()
    s = score[valid].astype(float).to_numpy()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={UID: str})
    if UID not in frame:
        raise ValueError(f"{path} has no {UID!r} column")
    if frame[UID].duplicated().any():
        raise ValueError(f"{path} has duplicate {UID} values")

    out = frame[[UID]].copy()
    missing: list[str] = []
    for target in TARGETS:
        candidates = [f"prob_{target}", target, f"pred_{target}"]
        source = next((column for column in candidates if column in frame), None)
        if source is None:
            missing.append(target)
            continue
        out[target] = pd.to_numeric(frame[source], errors="coerce")
    if missing:
        raise ValueError(f"{path} is missing targets: {missing}")
    return out.set_index(UID)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Community/external label CSV to evaluate")
    parser.add_argument("--name", default="candidate", help="Short output-file prefix")
    parser.add_argument("--baseline", type=Path,
                        default=Path("data/pseudo_labels_calibrated.csv"))
    parser.add_argument("--gold", type=Path, default=Path("data/metadata/train.csv"))
    parser.add_argument("--baseline-gold", type=Path,
                        default=Path("data/pseudo_labels_valid.csv"),
                        help="Existing DeepSeek predictions for gold reports")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/label_source_comparison"))
    args = parser.parse_args()

    candidate = load_scores(args.candidate)
    baseline = load_scores(args.baseline)
    gold = pd.read_csv(args.gold, dtype={UID: str}).set_index(UID)
    gold = gold[[target for target in TARGETS if target in gold]].apply(pd.to_numeric, errors="coerce")
    gold = gold[gold.notna().all(axis=1)]

    # Some project label files intentionally include the 58 official labels.
    # They are not pseudo-labels and must not enter an agreement comparison.
    common = candidate.index.intersection(baseline.index).difference(gold.index)
    if not len(common):
        raise ValueError("No shared StudyInstanceUID values between candidate and baseline")

    rows = []
    for target in TARGETS:
        a, b = candidate.loc[common, target], baseline.loc[common, target]
        valid = a.notna() & b.notna()
        a, b = a[valid], b[valid]
        rows.append({
            "target": target,
            "n_shared": len(a),
            "candidate_mean": a.mean(),
            "baseline_mean": b.mean(),
            "mean_abs_difference": (a - b).abs().mean(),
            "pearson": a.corr(b, method="pearson"),
            "spearman_rank": a.corr(b, method="spearman"),
            "hard_disagreement_rate": ((a >= 0.5) != (b >= 0.5)).mean(),
        })
    agreement = pd.DataFrame(rows)

    gold_rows = []
    candidate_gold = candidate.index.intersection(gold.index)
    for target in TARGETS:
        score = candidate.loc[candidate_gold, target]
        truth = gold.loc[candidate_gold, target]
        exact_rate = float(np.isclose(score, truth, equal_nan=False).mean()) if len(truth) else float("nan")
        gold_rows.append({"source": args.name, "target": target, "n_gold": len(truth),
                          "auc": auc(truth, score), "exact_match_rate": exact_rate})

    # The historical validation file contains hard DeepSeek predictions, which
    # is still useful as a baseline diagnostic even though it is not a fair
    # out-of-sample model-selection score.
    if args.baseline_gold.exists():
        historical = load_scores(args.baseline_gold)
        historical_gold = historical.index.intersection(gold.index)
        for target in TARGETS:
            truth = gold.loc[historical_gold, target]
            score = historical.loc[historical_gold, target]
            gold_rows.append({"source": "historical_deepseek", "target": target,
                              "n_gold": len(historical_gold), "auc": auc(truth, score),
                              "exact_match_rate": float(np.isclose(score, truth).mean())})
    gold_eval = pd.DataFrame(gold_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    agreement_path = args.out_dir / f"{args.name}_agreement.csv"
    gold_path = args.out_dir / f"{args.name}_gold_auc.csv"
    agreement.to_csv(agreement_path, index=False)
    gold_eval.to_csv(gold_path, index=False)

    print(f"Shared pseudo-labelled studies: {len(common):,}")
    print(f"Mean rank correlation: {agreement['spearman_rank'].mean():.4f}")
    print(f"Mean absolute label difference: {agreement['mean_abs_difference'].mean():.4f}")
    print(f"Wrote {agreement_path}")
    print(f"Wrote {gold_path}")
    print("\nCandidate gold AUC (diagnostic only):")
    print(gold_eval[gold_eval['source'] == args.name].to_string(index=False))
    if (gold_eval.loc[gold_eval['source'] == args.name, 'exact_match_rate'] > 0.98).any():
        print("WARNING: candidate has near-exact gold labels for at least one target; "
              "do not use its gold AUC as evidence of label quality.")


if __name__ == "__main__":
    main()
