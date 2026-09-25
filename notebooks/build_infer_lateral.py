"""Assemble kaggle_inference_lateral_swap.ipynb from cells_infer_lateral/ directory.

推理专用 notebook (零训练): 加载 3 个 v5 seed 成员 checkpoint (288px DINOv2-small)
+ v6a rad 成员 checkpoint (224px RadImageNet R50), 跳过训练直接做 gold 验证 +
test 推理, base = 3 seed rank-mean, Lateral Meniscus / Lateral OA 换成 rad 的
test 池内 rank → submission.csv (lateral_swap 彩票提交)。
"""

import json
from pathlib import Path

CELLS_DIR = Path(__file__).resolve().parent / "cells_infer_lateral"
OUTPUT = Path(__file__).resolve().parent / "kaggle_inference_lateral_swap.ipynb"

SECTION_HEADERS = {
    "01_imports.py":       "## 1. 导入",
    "02_config.py":        "## 2. 配置 — v5 (288px) 与 rad (224px) 两套数据侧参数",
    "03_slot_matching.py": "## 3. Slot 匹配 + 侧性检测",
    "04_dicom_io.py":      "## 4. DICOM — 空间排序 + 物理裁剪 + 侧性归一化",
    "05_model.py":         "## 5. 模型 — MultiViewModel (v5) + RadResNetModel (rad) + 诊断池化",
    "06_load_gold.py":     "## 6. 加载 gold 研究 (仅验证, 无需 v5 标签数据集)",
    "07_cache_gold.py":    "## 7. 构建 gold 双分辨率缓存 (58 研究, ~4 分钟)",
    "08_checkpoints.py":   "## 8. Checkpoint 定位 + 配置交叉核对 + 推理模型构建",
    "09_test_cache.py":    "## 9. Test slot 匹配 + 双分辨率缓存",
    "10_infer_v5.py":      "## 10. v5 seed 成员推理 (gold + test)",
    "11_infer_rad.py":     "## 11. rad 成员推理 (gold + test) + 释放 gold 缓存",
    "12_fusion_swap.py":   "## 12. Lateral swap 融合 + Submission",
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
        "# Lateral Swap 推理专用 — 3 seed (v5) + rad (v6a) 集成提交 (不训练)\n\n"
        "### 用途\n\n"
        "加载 3 个 v5 seed 成员 checkpoint (`best_model_s42/s142/s242.pt`, 288px DINOv2-small)\n"
        "+ v6a rad 成员 checkpoint (`best_model_rad.pt`, 224px RadImageNet R50), 跳过训练直接做\n"
        "gold 验证 + test 推理, 然后 **lateral_swap 融合**: 10 类用 base (3 seed rank-mean),\n"
        "Lateral Meniscus / Lateral OA 用 rad 的 test 池内 rank → `submission.csv`。\n\n"
        "### 依据 (58 gold 融合扫描 oracle)\n\n"
        "- Lateral Meniscus: base 0.7528 → rad 0.8124 (**+0.060**)\n"
        "- Lateral OA:      base 0.8114 → rad 0.8704 (**+0.059**)\n"
        "- 其余类 rad 无杠杆 → 只换这两类; swap gold 期望 ≈ 0.9057 (base 0.8958)\n\n"
        "### 运行前必做\n\n"
        "1. **上传 4 个 checkpoint 为 Kaggle Dataset** (私有即可):\n"
        "   `results/v5s{1,2,3}/checkpoints/best_model_s{42,142,242}.pt` (各 132MB)\n"
        "   + `results/v6a/checkpoints/best_model_rad.pt` (98.6MB) — 可放同一数据集,\n"
        "   挂载到本 notebook (已有 `v5-seed-checkpoints` 数据集的话只需把 rad 加进去,\n"
        "   或另建数据集; notebook 会自动扫描 `/kaggle/input` 全部数据集找这两个文件名)\n"
        "2. 若挂载路径不是 `CFG_V5['ckpt_input']` 的默认值, 改 cell 2 的 `ckpt_input` —\n"
        "   或不改, 自动扫描会兜底\n"
        "3. **加速器 T4x2**\n"
        "4. **无需挂载** v5 标签数据集、无需挂载 DINOv2/RadImageNet 预训练权重\n"
        "   (checkpoint 已包含全部权重)\n\n"
        "### 预期输出 (对照读数)\n\n"
        "- 各成员 gold AUC 应与各自训练会话一致: s42≈0.8933 / s142≈0.8899 / s242≈0.8937 /\n"
        "  rad≈0.8137\n"
        "- BASE gold ≈ 0.896 (本地融合实测 0.8959) / SWAP ≈ 0.9057 / BLEND ≈ 0.901\n"
        "- `submission.csv` = ★ lateral_swap, 提交这个\n"
        "- `submission_base.csv` = 纯 3 seed rank-mean (掉分回退)\n"
        "- `submission_lateral_blend.csv` = 两类 0.5/0.5 半混 (保守中间档)\n\n"
        "### 运行时长\n\n"
        "gold 双分辨率解码 ~4 分钟 + test 解码 (随测试集大小) + 4 成员推理 "
        "(~1-2 分钟/成员), 不训练, 全程远低于 9h 限制。\n\n"
        "### 评分说明\n\n"
        "LB 在**非公开 test 集**上评分 — 提交时 Kaggle 会在评分环境重跑本 notebook "
        "并对抗 hidden test 数据; 本 notebook 的 test 解码/推理/融合代码全部按 "
        "test.csv 实际行数通用处理, 不假设 3 行占位。\n"
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
