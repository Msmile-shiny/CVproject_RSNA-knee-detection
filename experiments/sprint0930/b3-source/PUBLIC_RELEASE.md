# RSNA Knee B3 V47/V48 public deployment bundle

This is a deployment-only release of the five EfficientNet-B3 checkpoints used
by the audited V49/V52 inference chain. The checkpoints were trained entirely
from the competition training split and public report-label datasets. They are
provided because the competition expects inference weights to be available as
a public Kaggle input.

Included:

- five fold checkpoints;
- exact training and inference source;
- per-fold configuration/input-hash manifests;
- aggregate, non-row-level audit metrics.

Explicitly excluded:

- DICOM images or reconstructed image data;
- train or test UID tables;
- row-level labels, fold-label tables, or OOF prediction tables;
- hidden-test data, features, predictions, or submissions.

The checkpoints are competition-train-derived artifacts and are not relicensed
under an open-source software license. Use remains subject to the RSNA Knee
competition terms. The source files document their public-model lineage; the
upstream PyTorch, timm, and EfficientNet components retain their own licenses.

This package is not itself evidence of an official competition score.
