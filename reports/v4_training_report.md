# RSNA Knee MRI — DINOv2 Multi-View v4 Training Report

**Date:** 2026-08-10 | **GPU:** 2× T4 | **Backbone:** DINOv2 ViT-S/14 | **Image:** 224²

## Core Metrics

| Metric | Value |
|--------|-------|
| Gold Macro AUC | **0.7659** |
| vs v3 (0.6264) | **+0.1395** |
| Epochs | 21 (early stop, patience=15) |
| Best Epoch | 6 (val AUC 0.6168 single-window) |
| Best Class | Effusion **0.932** |
| Weakest Class | Synovitis **0.624** |

> ⚠️ This run did NOT include laterality normalization (bug fixed post-run). Expected +0.01–0.03 on re-run.

## v3 → v4 Per-Class AUC

| Class | v3 AUC | v4 AUC | Δ | Pos |
|-------|--------|--------|----|-----|
| ACL | 0.540 | **0.801** | +0.261 | 24 |
| MCL | 0.550 | 0.667 | +0.117 | 9 |
| Medial Meniscus | 0.580 | 0.686 | +0.106 | 26 |
| Lateral Meniscus | 0.600 | **0.809** | +0.209 | 23 |
| Medial OA | 0.750 | **0.899** | +0.149 | 15 |
| Lateral OA | 0.620 | **0.822** | +0.202 | 11 |
| PF OA | 0.700 | 0.759 | +0.059 | 21 |
| Effusion | 0.720 | **0.932** | +0.212 | 35 |
| Synovitis | 0.600 | 0.624 | +0.024 | 27 |
| Baker's | 0.580 | 0.696 | +0.116 | 12 |
| Contusion | 0.640 | **0.808** | +0.168 | 19 |
| Fracture | 0.540 | 0.688 | +0.148 | 18 |
| **MACRO AVG** | **0.626** | **0.766** | **+0.139** | — |

## 7-Window TTA Impact

Training validation used single middle window. Gold validation used 7-window TTA + diagnostic pooling. The gap reveals TTA's massive contribution:

| Class | Single Window | 7-Window TTA | Δ |
|-------|--------------|-------------|----|
| Lateral OA | 0.538 | 0.822 | **+0.284** |
| Contusion | 0.543 | 0.808 | **+0.266** |
| ACL | 0.581 | 0.801 | **+0.221** |
| MCL | 0.465 | 0.667 | **+0.202** |
| Lateral Meniscus | 0.624 | 0.809 | +0.185 |
| Effusion | 0.763 | 0.932 | +0.169 |
| Baker's | 0.558 | 0.696 | +0.138 |
| Medial OA | 0.767 | 0.899 | +0.132 |
| PF OA | 0.744 | 0.759 | +0.015 |
| Fracture | 0.660 | 0.688 | +0.028 |

## Training Dynamics

| Metric | Epoch 1 | Best | Final (Epoch 21) |
|--------|---------|------|-------------------|
| Train Loss | 0.0203 | — | 0.0127 (−37%) |
| Val Loss | 0.6260 | 0.5777 (epoch 13) | 0.5820 |
| Val AUC (single) | 0.5346 | 0.6168 (epoch 6) | 0.6151 |

**Observations:**

- **Mild overfitting** — train loss drops continuously while val loss plateaus from ~epoch 8. Model learns pseudo-label distribution beyond what generalizes to gold labels.
- **Noisy early stopping** — single-window AUC fluctuates ±0.01; best at epoch 6. With 7-window TTA in training validation (now fixed), early stopping will be more reliable.
- **LR restart** — CosineAnnealingWarmRestarts (T₀=15) causes LR reset at epoch 16, visible as train loss bump (0.0128 → 0.0143).

## Bug Status

| Issue | Severity | Status |
|-------|----------|--------|
| Laterality normalization disabled (`lat=None`) | 🔴 Critical | ✅ Fixed `322c808` |
| Submission.csv only 3 studies (test_series.csv insufficient) | 🔴 Critical | ✅ Fixed `322c808` |
| Training validation single-window (no TTA) | 🟠 High | ✅ Fixed `322c808` |
| `infer_test_batch` early return skipped entire batch | 🟠 High | ✅ Fixed `322c808` |
| `best_model.pt` fallback to last epoch | 🟠 High | ✅ Fixed |
| `\` SyntaxError in slot_matching.py f-string | 🟠 High | ✅ Fixed |
| Self-distillation (v3→v4 pseudo re-labeling) not implemented | 🔵 Low | Known limitation |
| Test set laterality not detected | 🔵 Low | Known limitation |

## V4 Optimizations Applied

| Optimization | This Run |
|-------------|----------|
| Physical crop 160mm | ✅ |
| Spatial slice ordering | ✅ |
| Diagnostic-specific TTA pooling | ✅ |
| EMA weight averaging (0.999) | ✅ |
| Focal Loss (α=0.25, γ=2.0) | ✅ |
| Full pseudo-label training (~4000 studies) | ✅ |
| All gold → validation (58 studies) | ✅ |
| ThreadPoolExecutor parallel I/O | ✅ |
| Laterality normalization | ❌ (fixed post-run) |
