# R7 + S6: a fixed-weight ensemble improved Public AUC from 0.956 to 0.957

## Scope and evidence

This is a reproducible competition experiment built on public models, not a new architecture or a claim of first discovery.

| Experiment | Submission | Script version | Public score |
|---|---|---|---|
| Raptor384 + R7, rank weights 0.50 / 0.50 | 57144024 | 357976215 | 0.956 |
| Raptor384 + R7 + S6, rank weights 0.50 / 0.40 / 0.10 | 57257123 | 358727913 | 0.957 |

The improvement is +0.001 at displayed precision. These are two scored submissions, not a confidence interval, a repeated-seed study, or evidence of improved Private AUC. Hidden labels and per-case errors are unavailable. The three visible test cases verify execution only. A separate R6-based submission 57261794 already scored 0.958; this experiment is not our best submission or a state-of-the-art claim.

## Our contribution

We constructed and evaluated this fixed combination, reusing the published S6 member without rerunning its Raptor branch. We isolated outputs, pinned source and asset hashes, required complete study-level predictions, and independently verified ID alignment and the fusion formula. We did not train these backbones or create the underlying weights. Whether this exact combination was previously published has not been established.

For each finding independently, let rank mean average-tie percentile rank over the submitted studies:

`score = 0.50 * rank(Raptor384) + 0.40 * rank(R7) + 0.10 * rank(S6)`.

All prediction members must be present. Missing members do not silently fall back to the parent. This does not establish that every DICOM slice decoded successfully (`decode_coverage_verified=False`). The visible execution took 881.61 seconds; this is not a hidden-runtime estimate.

## Why not automatically transfer this to 0.958?

An additional model can correct one ensemble's ordering errors while duplicating or worsening another's. Our positive result concerns R7. The community reports a different R6 + S6 combination at 0.957 versus its 0.958 control; this is evidence against assuming universal improvement, not proof that every R6/S6 weight is harmful.

A future fixed test could replace 20% of the R6 contribution in each finding with S6, leaving the 0.958 parent Raptor weights unchanged: `w * RaptorRank + (1-w) * (0.8 * R6Rank + 0.2 * S6Rank)`. That is an unrun hypothesis, not the scored experiment above. It must be a separate candidate and should not be launched alongside the current published-v2 candidate without reviewing runtime and score evidence.

## Attribution and publication checklist

- R7 public parent: https://www.kaggle.com/code/jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble
- Heliosli compact R7 model: https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss
- nartaa S6 code: https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo
- nartaa S6 weights: https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights
- Raptor weights: https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007
- Preserve all original code credits, report-label acknowledgements, Dread Development credits, and OAI/NDA acknowledgements embedded below.
- Before public release, verify each upstream code/data licence and competition sharing rules. Public accessibility alone is not permission to relicense. Do not attach original patient data or report-label tables to this publication.
- This local draft has not been uploaded, rerun, or made public. Only markdown is added; scored code cells remain byte-for-byte equivalent as strings. A new published version is not itself the scored version.

Recommended description: **Public competition technical report / reproducible ensemble experiment**. Do not describe it as a peer-reviewed paper, a new foundation model, statistically significant improvement, or guaranteed medal result.
