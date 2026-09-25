"""Build Phase 3B: a 140-mm geometry-only v5 reproduction.

The historical v5 label file remains unchanged.  The only model-input change
from the v5 source cells is crop_mm: 130.0 -> 140.0.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CELLS_DIR = ROOT / "cells_v5"
OUTPUT = ROOT / "kaggle_train_v5_stage3b_140mm_v5labels.ipynb"
LABEL_INPUT = "/kaggle/input/rsna-knee-v5-labels"
LABEL_FILE = "v5_labels.csv"
LABEL_SHA256 = "TO_BE_FILLED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
        "outputs": [],
        "execution_count": None,
    }


def main() -> None:
    label_path = ROOT.parent / "data" / "processed" / LABEL_FILE
    label_sha = sha256(label_path)
    cells: list[dict] = [cell("markdown", """# Phase 3B — v5 140 mm geometry-only ablation

This run uses the historical `v5_labels.csv` unchanged.  The only altered
model-input setting is the physical crop: `130 mm -> 140 mm`.

The notebook verifies the exact label file hash before reading it.  Gold cases
remain validation-only, as in the original v5 training contract.
""")]

    for path in sorted(CELLS_DIR.iterdir()):
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        if path.suffix == ".md":
            cells.append(cell("markdown", source))
            continue
        if path.name == "03_config.py":
            source = source.replace(
                "'label_input':  '/kaggle/input/datasets/easoncyy/rsna-knee-v5-labels',",
                f"'label_input':  {LABEL_INPUT!r},",
            )
            source = source.replace("'crop_mm': 130.0,", "'crop_mm': 140.0,")
            source += "\nCFG['experiment_name'] = 'stage3b_v5labels_140mm'\n"
        elif path.name == "09_load_data.py":
            guard = f'''
# Phase 3B provenance guard: reject a Kaggle dataset version with a different label file.
import hashlib as _phase3b_hashlib
_phase3b_digest = _phase3b_hashlib.sha256(v5_label_file.read_bytes()).hexdigest()
_phase3b_expected = {label_sha!r}
if _phase3b_digest != _phase3b_expected:
    raise RuntimeError(
        f'Phase 3B requires the historical v5_labels.csv SHA256={{_phase3b_expected}}; '
        f'got {{_phase3b_digest}}. Update the mounted Kaggle Dataset version explicitly.'
    )
print(f'Phase 3B label contract verified: {{_phase3b_digest}}')
'''
            marker = "fused_df = pd.read_csv(v5_label_file)"
            if marker not in source:
                raise RuntimeError("v5 label load marker not found")
            source = source.replace(marker, guard + "\n" + marker)
        elif path.name == "17_submission.py":
            source += f'''

# Phase 3B provenance artifact
import json
with open(output_dir / 'phase3b_manifest.json', 'w', encoding='utf-8') as _fh:
    json.dump({{
        'experiment': CFG.get('experiment_name'),
        'crop_mm': CFG['crop_mm'],
        'label_file': str(v5_label_file),
        'label_sha256': {label_sha!r},
        'seed': CFG['seed'],
        'cache_slices': CFG['cache_slices'],
        'image_size': CFG['image_size'],
        'gold_macro_auc': gold_macro,
    }}, _fh, indent=2)
'''
        cells.append(cell("code", source))

    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10.0"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(cells)} cells); label SHA256={label_sha}")


if __name__ == "__main__":
    main()
