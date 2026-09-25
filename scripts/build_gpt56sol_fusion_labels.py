"""Create a conservative DeepSeek/GPT-5.6-Sol fused soft-label file.

Only the targets selected in ``--replace-targets`` receive GPT-5.6-Sol labels.
All other probabilities, weights, masks, and historical columns are copied
unchanged from the DeepSeek-calibrated file.  GPT's hard 0/1 verdicts are
calibrated against the official 58 gold rows using the same shrinkage idea as
the existing calibration script; they are not treated as literal probabilities.

The output is a new file.  Existing label files are never overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]
DEFAULT_REPLACE = ["ACL", "Lateral Meniscus", "Effusion", "Baker's", "Contusion", "Fracture"]


def posterior_rate(successes: float, total: int, prior: float, strength: float) -> float:
    return float((successes + strength * prior) / (total + strength))


def fit_hard_mapping(labels: pd.Series, truth: pd.Series) -> dict[int, float]:
    """Estimate P(gold=1 | GPT verdict) with conservative beta shrinkage."""
    valid = labels.notna() & truth.notna() & labels.isin([0, 1])
    y, p = truth[valid].astype(int), labels[valid].astype(int)
    prevalence = posterior_rate(float(y.sum()), len(y), 0.5, 4.0)
    mapping: dict[int, float] = {}
    for verdict in (0, 1):
        selected = y[p.eq(verdict)]
        mapping[verdict] = np.clip(
            posterior_rate(float(selected.sum()), len(selected), prevalence, 8.0), 0.01, 0.99
        )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek", type=Path, default=Path("data/pseudo_labels_calibrated.csv"))
    parser.add_argument("--gpt", type=Path,
                        default=Path("data/external_labels/gpt56sol/report_labels_gpt56sol.csv"))
    parser.add_argument("--gold", type=Path, default=Path("data/metadata/train.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/pseudo_labels_deepseek_gpt56sol_fused.csv"))
    parser.add_argument("--replace-targets", nargs="+", default=DEFAULT_REPLACE,
                        choices=TARGETS)
    args = parser.parse_args()

    base = pd.read_csv(args.deepseek, dtype={UID: str})
    gpt = pd.read_csv(args.gpt, dtype={UID: str}).set_index(UID)
    gold = pd.read_csv(args.gold, dtype={UID: str}).set_index(UID)
    gold = gold[TARGETS].apply(pd.to_numeric, errors="coerce")
    gold = gold[gold.notna().all(axis=1)]

    if base[UID].duplicated().any() or gpt.index.duplicated().any():
        raise ValueError("Duplicate StudyInstanceUID found")
    missing = [target for target in args.replace_targets if target not in gpt]
    if missing:
        raise ValueError(f"GPT file missing targets: {missing}")

    output = base.copy()
    output_index = output.set_index(UID)
    report_rows: list[dict[str, object]] = []
    for target in args.replace_targets:
        gpt_gold = gpt.index.intersection(gold.index)
        verdict_gold = pd.to_numeric(gpt.loc[gpt_gold, target], errors="coerce")
        mapping = fit_hard_mapping(verdict_gold, gold.loc[gpt_gold, target])

        shared = output_index.index.intersection(gpt.index).difference(gold.index)
        verdict = pd.to_numeric(gpt.loc[shared, target], errors="coerce")
        new_probability = verdict.map(mapping)
        usable = new_probability.notna()
        output_index.loc[shared[usable], f"prob_{target}"] = new_probability[usable].astype(np.float32)

        # Preserve existing reliability weights and masks.  This makes the
        # experiment isolate the teacher-label change rather than silently
        # increasing a target's loss contribution.
        output_index.loc[shared[usable], f"pred_{target}"] = verdict[usable].astype(int)
        output_index.loc[shared[usable], f"conf_{target}"] = "GPT56SOL_CALIBRATED"
        report_rows.append({
            "target": target,
            "gold_mapping_for_gpt_0": mapping[0],
            "gold_mapping_for_gpt_1": mapping[1],
            "replaced_pseudo_rows": int(usable.sum()),
            "weight_policy": "preserved_from_deepseek",
        })

    output = output_index.reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    report = pd.DataFrame(report_rows)
    report_path = args.output.with_name(args.output.stem + "_report.csv")
    report.to_csv(report_path, index=False)
    print(f"Created {args.output} with {len(output):,} pseudo-labelled studies")
    print(f"Replaced targets: {', '.join(args.replace_targets)}")
    print(f"Calibration report: {report_path}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
