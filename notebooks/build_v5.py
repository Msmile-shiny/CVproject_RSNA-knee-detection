"""Assemble kaggle_train_v5_multiview.ipynb from cells_v5/ directory.

v5 Improvements over v4 (v5 规划前三项):
  ① 标签系统重写 — v5 融合软标签 (本地 scripts/build_v5_labels.py 生成):
     文本提取器 score (text teacher) × 公开 20 成员集成 OOF (image teacher,
     4 seeds × 5 folds, 全 4407 研究无泄漏 OOF) 在 58 gold 上 per-finding
     逻辑回归融合; 置信度权重 (0.35+0.65·conf × text/oof 一致性);
     FocalBCE → WeightedSoftBCE (置信度加权软 BCE)。
  ② 分辨率 224px/160mm → 288px/130mm — 0.451mm/px 满足奈奎斯特采样定理
     (v4 0.714mm/px 不足以解析 1mm 半月板撕裂; 0.903 冠军同款 288px)。
  ③ 训练稳健性 — 墙钟预算保护 (max_train_minutes) + epochs 40→30。
  其余沿用 v4: 物理裁剪/侧性归一化/诊断池化/EMA/全部 gold 验证。
"""

import json
from pathlib import Path

CELLS_DIR = Path(__file__).resolve().parent / "cells_v5"
OUTPUT = Path(__file__).resolve().parent / "kaggle_train_v5_multiview.ipynb"

SECTION_HEADERS = {
    "01_setup.py":              "## 1. 环境安装",
    "02_imports.py":            "## 2. 导入",
    "03_config.py":             "## 3. 配置 — v5 融合标签 + 288px/130mm",
    "04_slot_matching.py":      "## 4. Slot 匹配 + 侧性检测",
    "05_dicom_io.py":           "## 5. DICOM — 空间排序 + 物理裁剪 + 侧性归一化",
    "06_model.py":              "## 6. 模型 — SlotHead + MultiViewModel + 诊断池化",
    "07_loss.py":               "## 7. 损失函数 — 置信度加权软 BCE",
    "08_dataset.py":            "## 8. 数据集",
    "09_load_data.py":          "## 9. 加载数据 — v5 融合软标签 (teacher-student), Gold→Val",
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

    # Add header
    header = (
        "# DINOv2 Multi-View v5 — Fused Soft Labels (Teacher-Student) + 288px/130mm Nyquist\n\n"
        "### v4 → v5 核心升级\n\n"
        "| 优化 | v4 | v5 |\n"
        "|---|---|---|\n"
        "| 标签 | 硬伪标签 (FocalBCE) | **v5 融合软标签** (text 提取器 × 公开集成 OOF teacher) |\n"
        "| 损失 | Focal Loss (α=0.25, γ=2) | **WeightedSoftBCE** (置信度加权软标签) |\n"
        "| 分辨率 | 224px @ 160mm (0.714mm/px) | **288px @ 130mm (0.451mm/px)** 满足奈奎斯特 |\n"
        "| 训练轮数 | 40 | **30** + 墙钟预算保护 (420min) |\n"
        "| 物理裁剪 | 160mm FOV | **130mm FOV** (膝关节实径, 无浪费像素) |\n"
        "| 侧性归一化 | 右膝→左膝 | 沿用 |\n"
        "| TTA 池化 | 诊断特异性 max/top2/mean | 沿用 |\n"
        "| EMA | decay=0.999 | 沿用 |\n"
        "| 验证集 | 全部 58 gold | 沿用 (与 v4 0.833 可比) |\n"
        "| 输出 | submission.csv | 沿用 |\n\n"
        "### 运行前必读\n\n"
        "- **必须挂载 v5 标签数据集**: 本地 `scripts/build_v5_labels.py` 生成 "
        "`data/processed/v5_labels.csv`, 上传为 Kaggle Dataset (根目录), "
        "挂载后路径与 CFG['label_input'] 一致 (默认 `/kaggle/input/datasets/easoncyy/rsna-knee-v5-labels`)\n"
        "- **加速器必须选 T4x2** (288px 缓存 ~19.7GB RAM; P100 的 16GB RAM 装不下, "
        "若只能用 P100 请把 CFG['image_size'] 改为 256)\n"
        "- 其余挂载同 v4: 竞赛数据 + rsna-dinov2-weights (timm 权重)\n"
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
