# Anchor942 audited reproduction

Kaggle Notebook: https://www.kaggle.com/code/easoncyy/rsna-anchor942-audited

This directory locks the public `maverickss26/rsna-knee-0942-restructured` source and builds a strict private reproduction. The inference recipe is unchanged. The audit patch rejects partial CoAt families and any residual-CoAt case fallback, then validates UID order, finite probabilities and branch receipts.

Version 1 finished the model pipeline but the added final check referenced the wrong root variable. Version 2 corrected only that audit-cell variable, completed successfully, and was submitted for scoring. The model computations and blend recipe did not change between those versions.

The visible three-study run produced `complete=true`, both `resgated_top3` and `d4_swa3`, and zero residual fallback studies. This establishes execution integrity, not model quality. The Kaggle Public score is the external result used for promotion.

Files:

- `upstream/`: downloaded source and Kaggle metadata;
- `build_anchor942.py`: deterministic strict-copy builder;
- `anchor942_build_receipt.json`: source and patched-cell hashes;
- `notebook/`: uploadable notebook and its input configuration;
- `run-receipts/`: small execution receipts; no model weights or hidden predictions.

`build_fracture_ablation.py` prepares the next controlled experiment from the hidden-robust parent that actually scored 0.942. It changes one line—Fracture is re-admitted to the RadImageNet blend while Baker's cyst remains excluded—so its leaderboard delta is interpretable.

Submission `56446116` later failed during Kaggle's larger hidden rerun. The visible run itself was complete, and Kaggle intentionally withholds the hidden traceback. The most likely failure source is the strict audit patch: it turned the upstream recipe's recoverable per-study/child fallbacks into fatal exceptions. `build_anchor942_hidden_robust.py` therefore starts again from the untouched upstream recipe, preserves all original fallback behavior, and adds only non-invasive final schema and degradation receipts.

The hidden-robust public run completed without degradation and scored **0.942** as submission `56454576` from Kaggle Notebook version 1. This is the current public-score baseline.

The Kaggle submission receipt is `anchor942_submission_receipt.json`. The Fracture notebook now inherits from this hidden-robust parent; its build receipt records the parent hash and the single changed code cell.

Fracture version 1 completed its three-study visible run with both CoAt family members, zero fallback, and a valid 12-column submission. It was submitted as `56514045`; the public score is pending.
