"""Assemble kaggle_train_v6a.ipynb from cells_v6a/ directory.

v6a = v6 异架构集成第一成员: RadImageNet ResNet50 冻结编码器 @224px/130mm,
与 v5 (DINOv2-small @288px) 构成架构+领域+分辨率三重多样性。
标签/损失/池化/EMA/TTA 管线与 v5 完全同构 (同为 v5 融合软标签)。
"""

import json
from pathlib import Path

CELLS_DIR = Path(__file__).resolve().parent / "cells_v6a"
OUTPUT = Path(__file__).resolve().parent / "kaggle_train_v6a.ipynb"

SECTION_HEADERS = {
    "01_setup.py":              "## 1. 环境安装",
    "02_imports.py":            "## 2. 导入",
    "03_config.py":             "## 3. 配置 — RadImageNet R50 冻结 @224px/130mm",
    "04_slot_matching.py":      "## 4. Slot 匹配 + 侧性检测",
    "05_dicom_io.py":           "## 5. DICOM — 空间排序 + 物理裁剪 + 侧性归一化",
    "06_model.py":              "## 6. 模型 — SlotHead + RadResNetModel + 诊断池化",
    "07_loss.py":               "## 7. 损失函数 — 置信度加权软 BCE",
    "08_dataset.py":            "## 8. 数据集",
    "09_load_data.py":          "## 9. 加载数据 — v5 融合软标签, Gold→Val",
    "10_cache.py":              "## 10. 构建 RAM 缓存（并行 DICOM）",
    "11_dataloaders.py":        "## 11. DataLoader",
    "12_train_val.py":          "## 12. 训练与验证 — EMA + 诊断池化",
    "13_build_model.py":        "## 13. 构建模型 — RadImageNet R50 冻结 + WeightedSoftBCE + EMA",
    "13b_cuda_probe.py":        "## 13b. CUDA 泄漏探针 + 训练配置决策",
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
        "# v6a — RadImageNet ResNet50 冻结编码器成员 (v6 异架构集成第一成员)\n\n"
        "### 与 v5 的区别\n\n"
        "| 项 | v5 (s1/s2/s3) | v6a |\n"
        "|---|---|---|\n"
        "| 编码器 | DINOv2-small (ViT-s/14) | **RadImageNet ResNet50** (医学影像预训练, 官方 h5 转换) |\n"
        "| 编码器训练 | 最后 6 层解冻 | **全冻结** (只训 SlotHead) |\n"
        "| 分辨率 | 288px@130mm (0.451mm/px, 奈奎斯特) | **224px@130mm (0.580mm/px)** — 分辨率多样性, FOV 相同 |\n"
        "| 归一化 | ImageNet mean/std | **x/127.5 − 1** (由 bn1.running_mean 反解确认的 RadImageNet 训练归一化) |\n"
        "| 特征 | cls+mean+focal 拼接 (1152d) | **GAP (2048d)** |\n"
        "| 其余全部 | — | **完全同构** (SlotHead/软标签/损失/jitter TTA/诊断池化/EMA/30ep) |\n\n"
        "### 为什么\n\n"
        "seed 集成三成员两两相关 0.959-0.966 (再堆 seed 无意义), 融合上限定为"
        "逐类 oracle +0.0011 → 增益只能来自结构多样性成员: 不同架构 (CNN vs ViT) + "
        "不同预训练领域 (放射影像 vs 自然图像) + 不同分辨率 → 期望成员相关显著下降, "
        "rank-mean 融合直接受益。\n\n"
        "### 运行前必读\n\n"
        "- **必须挂载 RadImageNet 权重**: 把本地 `datasets/radimagenet_raw/`\n"
        "  `radimagenet_resnet50_notop.pt` (94.3MB, 由 `scripts/convert_radimagenet_r50.py`\n"
        "  从官方 h5 转换 + bias 吸收) 上传为 Kaggle Dataset, 默认路径\n"
        "  `/kaggle/input/rsna-radimagenet-r50/radimagenet_resnet50_notop.pt`\n"
        "  (与上传 slug 不一致时改 cell 3 的 `rad_weights`)\n"
        "- **挂载 v5 标签数据集** (rsna-knee-v5-labels, 与 v5 训练共用)\n"
        "- **加速器 T4x2** (224px 缓存 ~11.9GB RAM; 单 T4 13GB 边缘)\n"
        "- **无需** DINOv2 权重数据集 (checkpoint 自含全部权重)\n\n"
        "### 预期输出\n\n"
        "- 训练更快: R50 前向 ~3.4× 便宜于 ViT-s@288, 30 epochs 预计远低于预算\n"
        "- cell 17 打印 gold AUC: 单成员预期略低于 v5 成员 (更粗分辨率 + 冻结编码器), "
        "价值在集成多样性, 不在单模型分数\n"
        "- 产物 `best_model_rad.pt` + `gold_validation_predictions_rad.csv`\n"
        "  (上传为推理数据集后, 用融合扫描脚本 scripts/fusion_scan_v6.py 决定融合权重)\n"
    )
    cells.append(make_cell("markdown", header))

    cell_files = sorted(p for p in CELLS_DIR.glob("*") if p.is_file())
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
