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

`build_fracture_ablation.py` prepares the next controlled experiment, but its promotion gate is strict: run it only if submission `56446116` reproduces at least 0.942. It changes one line—Fracture is re-admitted to the RadImageNet blend while Baker's cyst remains excluded—so its leaderboard delta is interpretable.
