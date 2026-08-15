"""Assemble kaggle_train_v5spec.ipynb from cells_v5spec/ directory.

v5spec = v5 的专家成员分支 (方向3 修订版): 结构与数据侧参数与 v5 完全同构,
唯一区别 = 标签源换成纯词表提取器分数 (report_labels_v2.csv, 规则型/无泄漏)。

动机 (scripts/text_blend_probe.py, 58 gold):
  词表在 ACL/MCL/Lateral Meniscus/Lateral OA 上显著强于图像成员
  (0.953/0.964/0.846/0.839 vs 0.885/0.924/0.753/0.811), 但测试集无报告文本
  → 文本信号只能走训练期蒸馏: 训练词表监督的图像专家, 推理时只 rank 替换这 4 类。
"""

import json
from pathlib import Path

CELLS_DIR = Path(__file__).resolve().parent / "cells_v5spec"
OUTPUT = Path(__file__).resolve().parent / "kaggle_train_v5spec.ipynb"

SECTION_HEADERS = {
    "01_setup.py":              "## 1. 环境安装",
    "02_imports.py":            "## 2. 导入",
    "03_config.py":             "## 3. 配置 — spec 专家成员 (词表标签) + 288px/130mm",
    "04_slot_matching.py":      "## 4. Slot 匹配 + 侧性检测",
    "05_dicom_io.py":           "## 5. DICOM — 空间排序 + 物理裁剪 + 侧性归一化",
    "06_model.py":              "## 6. 模型 — SlotHead + MultiViewModel + 诊断池化",
    "07_loss.py":               "## 7. 损失函数 — 置信度加权软 BCE",
    "08_dataset.py":            "## 8. 数据集",
    "09_load_data.py":          "## 9. 加载数据 — 词表软标签 (报告规则提取器), Gold→Val",
    "10_cache.py":              "## 10. 构建 RAM 缓存（并行 DICOM）",
    "11_dataloaders.py":        "## 11. DataLoader",
    "12_train_val.py":          "## 12. 训练与验证 — EMA + 诊断池化",
    "13_build_model.py":        "## 13. 构建模型 — WeightedSoftBCE + EMA",
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

    header = (
        "# DINOv2 Multi-View v5spec — 词表标签监督专家成员 (方向3 修订版)\n\n"
        "### 与 v5 的唯一区别\n\n"
        "| 项 | v5 (s1/s2/s3) | v5spec |\n"
        "|---|---|---|\n"
        "| 标签源 | 融合软标签 (文本×OOF 逻辑回归) | **纯词表分数** (report_labels_v2.csv, 规则提取器) |\n"
        "| 标签语义 | 概率 (calibrated) | 有序分级 0.04~0.97, prob=clip(score, 0.01, 0.99) |\n"
        "| 样本权重 | 融合置信度 | **max(词表 conf, 0.5)** (沉默保底, 强证据加权) |\n"
        "| 其余全部 | — | **完全同构** (模型/损失/288px/130mm/jitter/EMA/30ep) |\n\n"
        "### 为什么\n\n"
        "词表提取器 (规则型, 无训练无泄漏) 在 58 gold 上 ACL/MCL/外侧半月板/外侧OA "
        "显著强于图像成员 (探针 scripts/text_blend_probe.py: 这 4 类 rank 混合 w=0.5 → "
        "gold 0.8959→0.9160)。测试集无报告文本 → 文本信号只能蒸馏进图像模型。\n"
        "融合用法: 推理端 0.75·rank(三成员融合) + 0.75·rank(spec) 只替换这 4 类。\n\n"
        "### 运行前必读\n\n"
        "- **必须挂载词表标签**: 把本地 `data/processed/report_labels_v2.csv` 上传到 "
        "v5 标签 Kaggle Dataset 的新版本 (与 v5_labels.csv 同一目录), "
        "默认路径 `/kaggle/input/datasets/easoncyy/rsna-knee-v5-labels`\n"
        "- **加速器必须选 T4x2** (与 v5 同规格: 288px 缓存 ~19.7GB RAM)\n"
        "- 其余挂载同 v5: 竞赛数据 + rsna-dinov2-weights (timm 权重)\n\n"
        "### 预期输出\n\n"
        "- cell 9 打印词表自身 gold AUC (蒸馏上限, ACL≈0.953/MCL≈0.964/LM≈0.846/LOA≈0.839)\n"
        "- cell 17 打印模型 gold AUC: 与上限的差距 = 蒸馏折损, 决定融合权重\n"
        "- 产物 `best_model_spec.pt` + `gold_validation_predictions_spec.csv`\n"
    )
    cells.append(make_cell("markdown", header))

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
