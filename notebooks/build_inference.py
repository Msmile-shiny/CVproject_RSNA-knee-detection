"""Build kaggle_inference_v4.ipynb — 纯推理 notebook，禁网可用.

Reads cells from cells_inference/ directory.
"""

import json
from pathlib import Path

CELLS_DIR = Path(__file__).resolve().parent / "cells_inference"
OUTPUT = Path(__file__).resolve().parent / "kaggle_inference_v4.ipynb"

SECTION_HEADERS = {
    "01_setup.py":          "## 1. Setup",
    "02_imports.py":        "## 2. Imports",
    "03_config.py":         "## 3. Configuration",
    "04_slot_matching.py":  "## 4. Slot Matching + Test DICOM Scan",
    "05_dicom_io.py":       "## 5. DICOM I/O",
    "06_model.py":          "## 6. Model Definition",
    "07_build_test_cache.py": "## 7. Build Test Cache",
    "08_load_model.py":     "## 8. Load Model Weights",
    "09_inference.py":      "## 9. Inference -> submission.csv",
}


def make_cell(cell_type, source):
    if isinstance(source, str):
        source = [line + "\n" for line in source.split("\n")]
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


def read_cell(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if filepath.suffix == ".md":
        return "markdown", content
    else:
        return "code", content


def main():
    cells = []

    # Header
    header = (
        "# DINOv2 Multi-View v4 — Inference Only\n\n"
        "纯推理 notebook：读取 best_model.pt + DINOv2 权重 → 推理 test 集 → submission.csv\n\n"
        "**前置条件：**\n"
        "1. DINOv2 权重已上传为 Kaggle Dataset (`rsna-dinov2-weights`，包含 `dinov2_vits14.pth`)\n"
        "2. best_model.pt 已上传为 Kaggle Dataset (`rsna-knee-v4-best-model`，包含 `best_model.pt`)\n"
        "3. 竞赛数据已挂载\n\n"
        "**不训练，不访问网络。**"
    )
    cells.append(make_cell("markdown", header))

    # Add all cell files in order
    cell_files = sorted(CELLS_DIR.glob("*.py"))
    for fp in cell_files:
        fname = fp.name
        if fname in SECTION_HEADERS and SECTION_HEADERS[fname]:
            cells.append(make_cell("markdown", SECTION_HEADERS[fname]))
        cell_type, source = read_cell(fp)
        cells.append(make_cell(cell_type, source))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Notebook written: {OUTPUT}")
    print(f"Cells: {len(cells)}")
    for i, c in enumerate(cells):
        src_preview = "".join(c["source"])[:100].replace("\n", " ").strip()
        print(f"  [{i:2d}] {c['cell_type']:9s} | {src_preview}...")


if __name__ == "__main__":
    main()
