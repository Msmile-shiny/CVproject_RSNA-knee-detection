"""Calibrate LLM pseudo-labels with the 58 gold reports.

This step does not call an API.  It turns binary ``pred_*`` plus categorical
``conf_*`` values into per-class soft targets and reliability weights.  The
small validation set is protected with hierarchical beta-binomial shrinkage,
so rare confidence buckets cannot produce unjustified probabilities of 0/1.

Outputs
-------
data/pseudo_labels_calibrated.csv
    StudyInstanceUID, pred_*, conf_*, prob_*, weight_*, mask_*
data/nlp_calibration_report.csv
    Per-class prevalence, hard-label metrics, and calibrated Brier scores.
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
CONFIDENCE_ORDER = {"FAIL": 0, "LOW": 1, "REVIEW": 1, "MEDIUM": 2, "HIGH": 3}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float(np.mean(y_true == y_pred)),
    }


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann–Whitney AUC with average ranks for ties; no sklearn dependency."""
    y_true = y_true.astype(int)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _posterior_rate(successes: float, total: int, prior: float, strength: float) -> float:
    """Posterior mean of a Bernoulli rate with an interpretable prior."""
    return float((successes + strength * prior) / (total + strength))


def fit_mapping(valid: pd.DataFrame, target: str, strength: float = 8.0) -> dict[tuple[int, str], float]:
    """Fit P(gold=1 | LLM prediction, confidence) with hierarchical shrinkage."""
    true = pd.to_numeric(valid[f"true_{target}"], errors="coerce")
    pred = pd.to_numeric(valid[f"pred_{target}"], errors="coerce").fillna(-1).astype(int)
    conf = valid[f"conf_{target}"].fillna("FAIL").astype(str).str.upper()
    usable = true.notna() & pred.isin([0, 1])
    true, pred, conf = true[usable], pred[usable], conf[usable]
    prevalence = _posterior_rate(float(true.sum()), len(true), 0.5, 4.0)

    pred_prior: dict[int, float] = {}
    for value in (0, 1):
        mask = pred.eq(value)
        pred_prior[value] = _posterior_rate(float(true[mask].sum()), int(mask.sum()), prevalence, 6.0)

    mapping: dict[tuple[int, str], float] = {}
    observed_levels = sorted(set(conf) | set(CONFIDENCE_ORDER))
    for value in (0, 1):
        for level in observed_levels:
            mask = pred.eq(value) & conf.eq(level)
            probability = _posterior_rate(
                float(true[mask].sum()), int(mask.sum()), pred_prior[value], strength
            )
            mapping[(value, level)] = float(np.clip(probability, 0.01, 0.99))
    mapping[(-1, "FAIL")] = prevalence
    return mapping


def calibrate_frame(
    pseudo: pd.DataFrame,
    valid: pd.DataFrame,
    strength: float = 8.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pseudo.copy()
    reports: list[dict] = []

    for target in TARGETS:
        required = [f"true_{target}", f"pred_{target}", f"conf_{target}"]
        missing = [column for column in required if column not in valid.columns]
        if missing:
            raise ValueError(f"validation file is missing {missing}")
        if f"pred_{target}" not in pseudo or f"conf_{target}" not in pseudo:
            raise ValueError(f"pseudo-label file is missing columns for {target}")

        mapping = fit_mapping(valid, target, strength=strength)
        pred = pd.to_numeric(result[f"pred_{target}"], errors="coerce").fillna(-1).astype(int)
        conf = result[f"conf_{target}"].fillna("FAIL").astype(str).str.upper()
        prevalence = float(pd.to_numeric(valid[f"true_{target}"], errors="coerce").mean())

        probabilities = np.array([
            mapping.get((int(p), str(c)), mapping.get((int(p), "FAIL"), prevalence))
            if p in (0, 1) else prevalence
            for p, c in zip(pred, conf)
        ], dtype=np.float32)
        reliability = np.clip(2.0 * np.abs(probabilities - 0.5), 0.05, 1.0)
        confidence_factor = conf.map({"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.35,
                                      "REVIEW": 0.25, "FAIL": 0.0}).fillna(0.0).to_numpy()
        weights = (reliability * confidence_factor).astype(np.float32)

        result[f"prob_{target}"] = probabilities
        result[f"weight_{target}"] = weights
        result[f"mask_{target}"] = (weights >= 0.15).astype(np.uint8)

        y_true = pd.to_numeric(valid[f"true_{target}"], errors="coerce").to_numpy()
        y_pred = pd.to_numeric(valid[f"pred_{target}"], errors="coerce").fillna(0).to_numpy()
        v_conf = valid[f"conf_{target}"].fillna("FAIL").astype(str).str.upper()
        calibrated = np.array([
            mapping.get((int(p), str(c)), prevalence) if p in (0, 1) else prevalence
            for p, c in zip(y_pred, v_conf)
        ])
        valid_mask = np.isfinite(y_true)
        y_true, y_pred, calibrated = y_true[valid_mask], y_pred[valid_mask], calibrated[valid_mask]
        auc = binary_auc(y_true, calibrated)
        hard_metrics = binary_metrics(y_true, y_pred)
        reports.append({
            "class": target,
            "n_gold": len(y_true),
            "prevalence": float(np.mean(y_true)),
            **hard_metrics,
            "calibrated_auc_in_sample": auc,
            "hard_brier": float(np.mean((y_true - np.clip(y_pred, 0, 1)) ** 2)),
            "calibrated_brier_in_sample": float(np.mean((y_true - calibrated) ** 2)),
            "mean_training_weight": float(weights.mean()),
            "usable_pseudo_labels": int((weights >= 0.15).sum()),
        })

    return result, pd.DataFrame(reports)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo", type=Path, default=Path("data/pseudo_labels.csv"))
    parser.add_argument("--valid", type=Path, default=Path("data/pseudo_labels_valid.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/pseudo_labels_calibrated.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/nlp_calibration_report.csv"))
    parser.add_argument("--strength", type=float, default=8.0,
                        help="Larger values shrink small confidence buckets more strongly")
    args = parser.parse_args()

    pseudo = pd.read_csv(args.pseudo, dtype={"StudyInstanceUID": str})
    valid = pd.read_csv(args.valid, dtype={"StudyInstanceUID": str})
    if pseudo["StudyInstanceUID"].duplicated().any():
        raise ValueError("pseudo-label file contains duplicate StudyInstanceUID values")

    calibrated, report = calibrate_frame(pseudo, valid, strength=args.strength)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(args.output, index=False)
    report.to_csv(args.report, index=False)

    print(f"Saved {len(calibrated):,} calibrated studies to {args.output}")
    print(f"Saved calibration report to {args.report}")
    print(f"Macro hard F1: {report['f1'].mean():.4f}")
    print(f"Macro calibrated AUC (diagnostic, in-sample): {report['calibrated_auc_in_sample'].mean():.4f}")
    print(f"Mean hard Brier: {report['hard_brier'].mean():.4f}")
    print(f"Mean calibrated Brier (diagnostic, in-sample): {report['calibrated_brier_in_sample'].mean():.4f}")


if __name__ == "__main__":
    main()
