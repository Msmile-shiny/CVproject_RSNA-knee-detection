"""Assemble kaggle_inference_v5_fusion.ipynb from cells_infer_v5/ directory.

推理专用 notebook: 跳过训练, 加载 3 个 seed 成员 checkpoint
(best_model_s42/s142/s242.pt, 由用户上传为 Kaggle Dataset),
gold + test 全量 TTA 推理 → rank-mean 融合 → submission.csv。
"""

import json
from pathlib import Path

CELLS_DIR = Path(__file__).resolve().parent / "cells_infer_v5"
OUTPUT = Path(__file__).resolve().parent / "kaggle_inference_v5_fusion.ipynb"

SECTION_HEADERS = {
    "01_imports.py":       "## 1. 导入",
    "02_config.py":        "## 2. 配置 — 与 v5 训练数据侧参数一致",
    "03_slot_matching.py": "## 3. Slot 匹配 + 侧性检测",
    "04_dicom_io.py":      "## 4. DICOM — 空间排序 + 物理裁剪 + 侧性归一化",
    "05_model.py":         "## 5. 模型 — SlotHead + MultiViewModel + 诊断池化",
    "06_load_gold.py":     "## 6. 加载 gold 研究 (仅验证, 无需 v5 标签数据集)",
    "07_cache_gold.py":    "## 7. 构建 gold 缓存 (58 研究, ~2 分钟)",
    "08_infer_fusion.py":  "## 8. Seed 集成推理 + rank-mean 融合 + Submission",
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
        "# DINOv2 v5 推理专用 — 3 seed 集成 + spec 专家融合 (不训练)\n\n"
        "### 用途\n\n"
        "加载 3 个 seed 成员 checkpoint (v5s1/s2/s3: "
        "`best_model_s42.pt` / `best_model_s142.pt` / `best_model_s242.pt`), "
        "跳过训练直接做 gold 验证 + test 推理, rank-mean 融合后产出 `submission.csv`。\n\n"
        "**spec 专家已归档 (2026-08-15)**: 离线扫描 `scripts/fusion_scan_spec.py` "
        "确认 4 类替换为负收益 (-0.0023), 逐类 oracle 上限也仅 +0.0011 (< +0.003 "
        "阈值) → 融合端禁用。即使扫描到 `best_model_spec.pt` 也只打印诊断 AUC, "
        "提交产物 = 纯 3 seed rank-mean, 与旧版行为一致。\n\n"
        "### 运行前必做\n\n"
        "1. **上传 3 个 checkpoint 为 Kaggle Dataset** (私有即可): 本地目录 "
        "`results/v5s{1,2,3}/checkpoints/` 里的 `best_model_s*.pt` (每个 ~132MB) "
        "打包为 Dataset 根目录, 挂载到本 notebook; "
        "spec 专家 `results/v5spec/checkpoints/best_model_spec.pt` 可同目录或另建数据集 "
        "(可跳过: 已归档不参与融合, 仅诊断)\n"
        "2. 若挂载路径不是 `CFG['ckpt_input']` 的默认值, 改 cell 2 的 "
        "`ckpt_input` — 或不改, notebook 会自动扫描 `/kaggle/input` 全部数据集找 "
        "`best_model_s*.pt`\n"
        "3. **加速器 T4x2** (test 缓存大小未知, 与训练会话同规格)\n"
        "4. **无需挂载** v5 标签数据集、无需挂载 DINOv2 预训练权重 "
        "(checkpoint 已包含全部权重)\n\n"
        "### 预期输出\n\n"
        "- 各成员 gold AUC 应与各自训练会话一致 (s42≈0.8933 / s142≈0.8899 / s242≈0.8937)\n"
        "- 纯 seed 融合 gold 宏 AUC ≈ 0.896 (58 研究本地融合实测 0.8959)\n"
        "- spec 若被扫到, 仅打印其诊断 AUC, FUSED = BASE\n"
        "- `submission.csv` = 最终融合预测, 可直接提交\n\n"
        "### 运行时长\n\n"
        "gold 解码 ~2 分钟 + test 解码 (随测试集大小) + 成员推理 (~1-2 分钟/成员), "
        "不训练, 全程远低于 9h 限制。\n"
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
