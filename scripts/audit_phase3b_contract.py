"""Audit immutable inputs for the Phase 3B geometry ablation.

This script does not train a model.  It records the exact label file, the
source cells, and the OOF archive structure used by the 140-mm run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase3b_contract_audit.json"
LABEL = ROOT / "data" / "processed" / "v5_labels.csv"
OOF = ROOT / "kaggle_dataset" / "archive" / "oof.npz"
TRAIN = ROOT / "data" / "metadata" / "train.csv"
CELLS = ROOT / "notebooks" / "cells_v5"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    labels = pd.read_csv(LABEL, dtype={"StudyInstanceUID": str})
    train = pd.read_csv(TRAIN, dtype={"StudyInstanceUID": str})
    z = np.load(OOF, allow_pickle=True)
    gold = train[TARGETS].notna().all(axis=1)

    expected_columns = ["StudyInstanceUID"] + [f"{prefix}{t}" for prefix in ("prob_", "weight_", "mask_") for t in TARGETS]
    missing = sorted(set(expected_columns) - set(labels.columns))
    if missing:
        raise RuntimeError(f"v5 labels missing columns: {missing}")
    if labels["StudyInstanceUID"].duplicated().any():
        raise RuntimeError("v5 labels contain duplicate StudyInstanceUID")
    ids = set(labels["StudyInstanceUID"])
    expected_ids = set(train["StudyInstanceUID"])
    if ids != expected_ids:
        raise RuntimeError("v5 label UIDs do not exactly match all training studies")

    payload = {
        "phase": "3B_geometry_only",
        "label_file": str(LABEL.relative_to(ROOT)),
        "label_sha256": sha256(LABEL),
        "label_rows": int(len(labels)),
        "label_gold_rows": int(labels["StudyInstanceUID"].isin(train.loc[gold, "StudyInstanceUID"]).sum()),
        "label_unlabeled_rows": int(labels["StudyInstanceUID"].isin(train.loc[~gold, "StudyInstanceUID"]).sum()),
        "label_columns_verified": expected_columns,
        "train_rows": int(len(train)),
        "gold_rows": int(gold.sum()),
        "unlabeled_rows": int((~gold).sum()),
        "oof_file": str(OOF.relative_to(ROOT)),
        "oof_sha256": sha256(OOF),
        "oof_keys": sorted(z.files),
        "oof_shapes": {key: list(z[key].shape) for key in z.files},
        "oof_gold_mask_count": int(z["gold_mask"].sum()),
        "cell_source_sha256": {
            path.name: sha256(path) for path in sorted(CELLS.iterdir()) if path.is_file()
        },
        "ooF_exclusion_evidence": (
            "The archive contains one prediction per study and a gold mask, but no per-study "
            "fold membership or member-training manifest. Its file structure alone cannot prove "
            "that each prediction excluded the corresponding study from training."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
