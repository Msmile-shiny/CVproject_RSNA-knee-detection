# RSNA 2026 Knee Abnormality Detection

Kaggle **RSNA 2026 Knee Abnormality Detection** competition project.  The task is to predict 12 knee-MRI abnormalities from DICOM studies; evaluation is macro ROC-AUC.  This repository contains reproducible Kaggle inference/training notebooks, modular notebook cells, local validation utilities, and the experiment record.

## Current status

| Item | Current result / decision |
|---|---|
| Best reproduced public solution | **LB 0.936** — community ensemble reproduced on Kaggle |
| Earlier reproducible baseline | **LB 0.920** — replica of `amanatar/rsna-knee-super-ensemble` |
| In-house v5 ensemble | Gold macro-AUC **0.8959** / LB **0.886** (3-seed rank mean) |
| Failed supervision experiments | Stage 3A **0.818**, Stage 3C **0.807** public LB |
| Current work | Stage 3D trusted-ranking control, then Phase 4A OrthoFoundation independent-member probe |
| Admission rule | Add a member only after measuring both accuracy and ranking diversity against the 0.936 parent |

The current plan is [Phase 4 independent member](docs/plans/phase4_independent_member.md). See the [documentation index](docs/README.md) for earlier plans, research and experiment reports.

## Approach

The active pipeline is an ensemble-oriented 2.5D MRI workflow:

- DICOM header parsing, physical-mm centre crops, ordering, laterality normalization, and anatomical slot matching;
- DINOv2 and RadImageNet feature/model branches, with 5-slice stacks or multi-slot views;
- LLM/report-derived calibrated soft pseudo-labels for 4,349 non-gold studies; the 58 gold studies remain held out for evaluation;
- weighted soft BCE, EMA, TTA, and study-level diagnostic pooling;
- percentile-rank fusion, which is appropriate for the competition's ranking-based ROC-AUC metric.

`reference_code/` is retained for audited public-notebook replication and comparison.  It is not an importable production dependency.

## Repository map

| Path | Purpose |
|---|---|
| `notebooks/cells_v5/` | Modular source cells for the active DINOv2 v5 training notebook |
| `notebooks/cells_v6a/` | RadImageNet ResNet-50 diversity-member training cells |
| `notebooks/kernel_push_super/` | Verified, submission-ready 0.920 super-ensemble package |
| `notebooks/cells_super/` | Extra-member blend cell for stage 2 |
| `scripts/` | Validation, checkpoint conversion, label building, fusion scans, and smoke tests |
| `datasets/`, `models/`, `losses/` | Reusable local dataset/model/loss implementations |
| `data/` | Metadata and pseudo-label artifacts (large raw/cache data is ignored) |
| `docs/` | Indexed plans, experiment reports, research notes, and archived early designs |
| `reference_code/` | Downloaded public references and extracted source cells |

Notebook assembly scripts (for example `notebooks/build_v5.py`, `build_v6a.py`, and `build_super_ours.py`) generate the corresponding `.ipynb` artifacts from the cell directories.

## Validation and submission

Install the local Python dependencies (including a suitable PyTorch build) with:

```bash
pip install -r requirements.txt
```

Useful checks:

```bash
python scripts/verify_super_ensemble_package.py
python scripts/verify_checkpoints_local.py
python scripts/validate_gold.py
```

The canonical 0.920 package is `notebooks/kernel_push_super/`.  Its README documents the Kaggle upload/push procedure.  Training and leaderboard submissions are intended for Kaggle GPU; local scripts should remain CPU-safe.

## Data, weights, and reproducibility

The repository deliberately excludes DICOM caches, downloaded datasets, checkpoints, generated results, and private API configuration.  See `.gitignore` for the exact rules.  A Kaggle run needs the competition data plus the datasets/models listed in the package metadata; do not assume local ignored assets are available on a fresh clone.

The 58 gold studies are for validation only.  Do not use them to train weights or tune per-target blend coefficients.  Use the global macro-AUC decision rule documented in the stage-2 plan.

## Project conventions

- Preserve the 0.920 package as a fail-closed baseline.
- Prefer independent architectural diversity over more highly correlated random seeds.
- Validate additions with emitted gold predictions and a global blend scan before a leaderboard submission.
- Record material outcomes under `docs/experiments/` and update `docs/README.md` so the next experiment has an audit trail.
