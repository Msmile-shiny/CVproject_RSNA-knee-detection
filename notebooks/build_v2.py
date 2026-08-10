"""Assemble kaggle_train_v2_softlabel.ipynb from cells_v2/ directory.

Each cell is a standalone .py (code) or .md (markdown) file.
No quoting issues -- source code is read directly from files.
"""

import json
from pathlib import Path

CELLS_DIR = Path(__file__).resolve().parent / "cells_v2"
OUTPUT = Path(__file__).resolve().parent / "kaggle_train_v2_softlabel.ipynb"

# Section headers — injected before the named cell file.
# Files NOT listed here simply follow the previous section without a new header.
SECTION_HEADERS = {
    "01_setup.py":              "## 1. 环境安装",
    "02_imports.py":            "## 2. 导入与配置",
    "04_spa.py":                "## 3. 模型组件",
    "10_dicom_io.py":           "## 4. DICOM 读取",
    "11_dataset.py":            "## 5. 数据集 — 软标签支持",
    "12_load_data.py":          "## 6. 加载校准数据",
    "13_cache.py":              "## 7. 构建 RAM 缓存",
    "15_dataloaders.py":        "## 8. DataLoader",
    "16_build_model.py":        "## 9. 构建模型 — 部分解冻 + 分离学习率",
    "17_train_val_functions.py":"## 10. 训练与验证函数",
    "18_sanity_check.py":       "## 11. 管线检查",
    "19_training_loop.py":      "## 12. 训练循环",
    "20_threshold_md.md":       "## 13. 阈值决策",
}


def make_cell(cell_type, source):
    """Create a notebook cell."""
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
    """Read cell content from file. Returns (cell_type, source_text)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if filepath.suffix == ".md":
        return "markdown", content
    else:
        return "code", content


def main():
    cells = []

    # Get all cell files sorted by name
    cell_files = sorted(CELLS_DIR.glob("*"))

    for fp in cell_files:
        fname = fp.name

        # Inject section header if any
        if fname in SECTION_HEADERS and SECTION_HEADERS[fname]:
            cells.append(make_cell("markdown", SECTION_HEADERS[fname]))

        # Read and add the cell
        cell_type, source = read_cell(fp)
        cells.append(make_cell(cell_type, source))

    # Build notebook
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
        src_preview = "".join(c["source"])[:80].replace("\n", " ").strip()
        print(f"  [{i:2d}] {c['cell_type']:9s} | {src_preview}...")


if __name__ == "__main__":
    main()
