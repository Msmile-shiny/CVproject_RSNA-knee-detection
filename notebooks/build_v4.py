"""Assemble kaggle_train_v4_multiview.ipynb from cells_v4/ directory.

v4 Improvements over v3:
  - Physical crop (160mm) matching reference code
  - Spatial slice ordering via ImagePositionPatient
  - Laterality normalization (right→left knee)
  - Diagnostic-specific TTA pooling (max for focal, top2 for ACL/MCL)
  - EMA weight averaging
  - Focal Loss instead of BCE
  - Full data training (~4000 pseudo-labeled studies)
  - All gold studies → validation
  - ThreadPoolExecutor for parallel DICOM I/O
  - Submission.csv generation for Kaggle competition
"""

import json
from pathlib import Path

CELLS_DIR = Path(__file__).resolve().parent / "cells_v4"
OUTPUT = Path(__file__).resolve().parent / "kaggle_train_v4_multiview.ipynb"

SECTION_HEADERS = {
    "01_setup.py":              "## 1. 环境安装",
    "02_imports.py":            "## 2. 导入",
    "03_config.py":             "## 3. 配置 — v4 全量优化",
    "04_slot_matching.py":      "## 4. Slot 匹配 + 侧性检测",
    "05_dicom_io.py":           "## 5. DICOM — 空间排序 + 物理裁剪 + 侧性归一化",
    "06_model.py":              "## 6. 模型 — SlotHead + MultiViewModel + 诊断池化",
    "07_loss.py":               "## 7. 损失函数 — FocalLoss",
    "08_dataset.py":            "## 8. 数据集",
    "09_load_data.py":          "## 9. 加载数据 — Gold→Val, 全量伪标签训练",
    "10_cache.py":              "## 10. 构建 RAM 缓存（并行 DICOM）",
    "11_dataloaders.py":        "## 11. DataLoader",
    "12_train_val.py":          "## 12. 训练与验证 — EMA + 诊断池化",
    "13_build_model.py":        "## 13. 构建模型 — FocalLoss + EMA",
    "14_sanity_check.py":       "## 14. 管线检查",
    "15_training_loop.py":      "## 15. 训练循环",
    "16_threshold_md.md":       "## 16. 阈值决策",
    "17_submission.py":         "## 17. Test 推理 + Submission.csv + Gold 验证",
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

    # Add header
    header = (
        "# DINOv2 Multi-View v4 — Full Data + Physical Crop + Laterality + Diag Pool + EMA + Focal\n\n"
        "### v3 → v4 核心升级\n\n"
        "| 优化 | v3 | v4 |\n"
        "|---|---|---|\n"
        "| 物理裁剪 | 无 | **160mm** 固定 FOV |\n"
        "| 切片排序 | 文件名 | **ImagePositionPatient** 空间排序 |\n"
        "| 侧性归一化 | 无 | **右膝→左膝** 水平翻转/反转 |\n"
        "| TTA 池化 | mean | **诊断特异性** max/top2/mean |\n"
        "| 损失函数 | BCE | **Focal Loss** (α=0.25, γ=2.0) |\n"
        "| 权重平均 | 无 | **EMA** (decay=0.999) |\n"
        "| 训练数据 | 500 子集 | **全量 ~4000** 伪标签 |\n"
        "| 验证集 | 11 (20% gold) | **全部 58 gold** |\n"
        "| DICOM I/O | 单线程 | **ThreadPoolExecutor** 并行 |\n"
        "| 输出 | 仅训练 | **submission.csv** 竞赛提交 |\n"
    )
    cells.append(make_cell("markdown", header))

    # Add all cell files in order
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
        src_preview = "".join(c["source"])[:100].replace("\n", " ").strip()
        print(f"  [{i:2d}] {c['cell_type']:9s} | {src_preview}...")


if __name__ == "__main__":
    main()
