"""Build the Stage-3A Kaggle notebook: v5 with only crop_mm changed to 140.

The generated notebook loads the separately uploaded fused-label dataset.  It
does not alter the original v5 cells or notebook, preserving a clean control.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CELLS_DIR = ROOT / "cells_v5"
OUTPUT = ROOT / "kaggle_train_v5_stage3a_140mm.ipynb"
LABEL_INPUT = "/kaggle/input/rsna-knee-stage3a-labels"
LABEL_FILE = "pseudo_labels_deepseek_gpt56sol_fused.csv"


def cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
        "outputs": [],
        "execution_count": None,
    }


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    cells: list[dict] = [cell("markdown", """# Stage 3A — v5 Geometry Ablation

**Only changed variable:** physical crop `130mm → 140mm`.

The model, six slots, nine cached slices, image size, seed, training schedule,
and loss are inherited from v5. Training labels are the independently stored
DeepSeek + GPT-5.6-Sol fused labels. Gold cases are validation only.
""")]

    for path in sorted(CELLS_DIR.iterdir()):
        if not path.is_file():
            continue
        source = read_source(path)
        if path.suffix == ".md":
            cells.append(cell("markdown", source))
            continue
        if path.name == "03_config.py":
            # Replace the defaults *inside* CFG so that both the config audit
            # and all downstream cells observe the experimental contract.
            # Appending an override after this cell printed CFG was confusing
            # and made a valid 140-mm run look like the 130-mm control.
            source = source.replace(
                "'label_input':  '/kaggle/input/datasets/easoncyy/rsna-knee-v5-labels',",
                f"'label_input':  {LABEL_INPUT!r},",
            )
            source = source.replace(
                "'crop_mm': 130.0,",
                "'crop_mm': 140.0,",
            )
            source += "\nCFG['experiment_name'] = 'stage3a_v5_140mm_gpt56sol_fused'\n"
        elif path.name == "09_load_data.py":
            source = source.replace("v5_label_file = label_input / 'v5_labels.csv'",
                                    f"v5_label_file = label_input / {LABEL_FILE!r}")
            source = source.replace(
                "upload data/processed/v5_labels.csv as a Kaggle Dataset",
                f"upload data/processed/{LABEL_FILE} as a Kaggle Dataset",
            )
        elif path.name == "17_submission.py":
            source += '''\n\n# Stage 3A provenance artifact\nimport json\nwith open(output_dir / 'stage3a_manifest.json', 'w', encoding='utf-8') as _fh:\n    json.dump({\n        'experiment': CFG.get('experiment_name'),\n        'crop_mm': CFG['crop_mm'],\n        'label_file': str(v5_label_file),\n        'seed': CFG['seed'],\n        'cache_slices': CFG['cache_slices'],\n        'image_size': CFG['image_size'],\n        'gold_macro_auc': gold_macro,\n    }, _fh, indent=2)\n'''
        cells.append(cell("code", source))

    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10.0"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
