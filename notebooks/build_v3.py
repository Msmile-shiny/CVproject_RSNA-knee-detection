"""Assemble kaggle_train_v3_multiview.ipynb from cells_v3/ directory."""

import json
from pathlib import Path

CELLS_DIR = Path(__file__).resolve().parent / "cells_v3"
OUTPUT = Path(__file__).resolve().parent / "kaggle_train_v3_multiview.ipynb"

SECTION_HEADERS = {
    "01_setup.py":              "## 1. 环境安装",
    "02_imports.py":            "## 2. 导入与配置",
    "03_config.py":             "## 3. 配置",
    "04_slot_matching.py":      "## 4. Slot 匹配",
    "05_dicom_io.py":           "## 5. DICOM 读取",
    "06_model.py":              "## 6. 模型 — SlotHead + MultiViewModel",
    "07_loss.py":               "## 7. 损失函数",
    "08_dataset.py":            "## 8. 数据集",
    "09_load_data.py":          "## 9. 加载数据与标签",
    "10_cache.py":              "## 10. 构建 RAM 缓存",
    "11_dataloaders.py":        "## 11. DataLoader",
    "12_train_val.py":          "## 12. 训练与验证函数",
    "13_build_model.py":        "## 13. 构建模型与优化器",
    "14_sanity_check.py":       "## 14. 管线检查",
    "15_training_loop.py":      "## 15. 训练循环",
    "16_threshold_md.md":       "## 16. 阈值决策",
    "17_gold_validation.py":    "## 17. Gold 验证（全部 gold 研究 AUC）",
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
    cell_files = sorted(CELLS_DIR.glob("*"))

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
        src_preview = "".join(c["source"])[:80].replace("\n", " ").strip()
        print(f"  [{i:2d}] {c['cell_type']:9s} | {src_preview}...")


if __name__ == "__main__":
    main()
