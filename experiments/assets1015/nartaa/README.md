# RSNA Knee: S6 ConvNeXt, all data, 0.950 public

One learned checkpoint, `E_prior_s42_fall.pt`: Heliosli's S6 architecture (convnext_small, per-finding spatial attention + window-attention MIL) retrained from the timm ImageNet-22k init on our report-derived JEV-blend soft labels, 4,263 competition studies, gold-58 excluded. Scored **0.950 public** alone in [rsna-knee-s6fall-solo](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo). Recipe and hashes: `checkpoint-manifest.json`. Write-up: the discussion post linked from the notebook.

The scored notebook, [rsna-knee-s6fall-solo](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo), runs from this dataset plus Heliosli's public [rsna-orthofoundation-fold0-epoch7-20261004](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004) (SKM-TEA anatomy V-Net and data pipeline, attached, not copied here) and our public 0.949 weights. `s6_extras.bin` holds the few code files his dataset lacks: his public-notebook `compact_*` scripts and 16 MONAI 0.6.0 files, both Apache-2.0. `code_manifest.json` pins the sha256 of every code file of the scored run.

No OAI data was used for this model. No images, reports, label tables or predictions are included. Public scores do not establish clinical validity.

## Licence

CC BY-NC 4.0 for these weights: use, fine-tune and redistribute for non-commercial purposes with credit (nartaa, and Heliosli for the architecture). Third-party code and pretrained components keep their own licences.

## Credits

Heliosli (S6 architecture, training/inference code, anatomy checkpoint); Desai et al. (SKM-TEA); weak-label sources stevenleehans, pilkwang and riadmohamed42 (JEV labels); RSNA and the competition data contributors.
