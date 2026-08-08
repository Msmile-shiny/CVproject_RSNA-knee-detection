"""Build mini_npy_index.csv — Series-level training index.

Input:
  - data/processed/mini_set_metadata.csv  (slice-level: 11,261 rows)
  - data/metadata/train.csv               (exam-level: 12 labels)

Output:
  - data/metadata/mini_npy_index.csv      (series-level: ~600 rows)

Each row = one series = one training sample (a volume of 2D slices).
"""

import pandas as pd
from pathlib import Path

SEED = 2026
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_COLS = [
    "ACL", "MCL",
    "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# ── 1. Load ──────────────────────────────────────────────────
meta = pd.read_csv(PROJECT_ROOT / "data/processed/mini_set_metadata.csv")
train = pd.read_csv(PROJECT_ROOT / "data/metadata/train.csv")

print(f"Slice-level rows: {len(meta):,}")
print(f"Exam-level rows:  {len(train):,}")

# ── 2. Aggregate to series level ─────────────────────────────
index = (
    meta.groupby(["StudyInstanceUID", "SeriesInstanceUID"], as_index=False)
    .agg(n_slices=("SOPInstanceUID", "nunique"))
)

print(f"\nBefore dedup: {len(index)} series from {index['StudyInstanceUID'].nunique()} studies")
print(f"Avg slices/series: {index['n_slices'].mean():.1f}")

# ── 3. Deduplicate: one series per study (keep the one with most slices) ──
index = index.sort_values("n_slices", ascending=False)
index = index.drop_duplicates(subset="StudyInstanceUID", keep="first")

print(f"After dedup:  {len(index)} series from {index['StudyInstanceUID'].nunique()} studies")

# ── 4. Clean & merge labels ──────────────────────────────────
for col in TARGET_COLS:
    train[col] = pd.to_numeric(train[col], errors="coerce").fillna(0).astype(int)

index = index.merge(
    train[["StudyInstanceUID"] + TARGET_COLS],
    on="StudyInstanceUID",
    how="left",
)

# ── 4. Add derived columns ───────────────────────────────────
index["patient_id"] = index["StudyInstanceUID"]  # One exam = one patient
index["n_labels"] = index[TARGET_COLS].sum(axis=1)

# ── 5. Rename to match dataset.py expectations ───────────────
index = index.rename(columns={"StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid"})

# ── 6. Validate ──────────────────────────────────────────────
n_abnormal = (index["n_labels"] > 0).sum()
n_normal = (index["n_labels"] == 0).sum()

print(f"\n── Validation ──")
print(f"Total rows:     {len(index)}")
print(f"Abnormal:       {n_abnormal}")
print(f"Normal:         {n_normal}")
print(f"Missing labels: {index[TARGET_COLS].isna().any(axis=1).sum()}")
print(f"Columns:        {list(index.columns)}")

# Per-class coverage check
print(f"\n── Per-class positive count ──")
for col in TARGET_COLS:
    n_pos = int(index[col].sum())
    print(f"  {col:20s}: {n_pos:4d}")

# ── 7. Save ──────────────────────────────────────────────────
out_dir = PROJECT_ROOT / "data/metadata"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "mini_npy_index.csv"
index.to_csv(out_path, index=False)

print(f"\n── Output ──")
print(f"Saved: {out_path}")
print(f"Size:  {out_path.stat().st_size / 1024:.1f} KB")
