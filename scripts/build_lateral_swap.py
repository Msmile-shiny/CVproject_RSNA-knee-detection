"""Lateral 两类彩票提交: base (3-seed rank-mean) 的 Lateral Meniscus/Lateral OA
换成 rad (v6a RadImageNet) 的 test 池内 rank。零 GPU, 零训练。

依据 (fusion_scan_v6.py V6 oracle, 58 gold):
  Lateral Meniscus: base 0.7528 -> rad 0.8124 (+0.060)
  Lateral OA:      base 0.8114 -> rad 0.8704 (+0.059)
  → 纯换这两类的 gold 宏期望 ≈ 0.8958 + (0.0596+0.0590)/12 ≈ 0.9057
  (n=58 单类噪声风险真实存在; 社区 two-target student receipt 独立佐证了
   lateral 两类是共性弱项 — prvsiyan 学生模型 LM +0.063, bootstrap P>0 = 0.9524)

产物 (results/fusion_v5s1s2s3/):
  submission_lateral_swap.csv   纯换 (oracle 同款, 期望最高)
  submission_lateral_blend.csv  保守版: 两类 = 0.5·base_rank + 0.5·rad_rank

用法: python scripts/build_lateral_swap.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BASE_SUB = RESULTS / "fusion_v5s1s2s3" / "submission_fused.csv"
RAD_SUB = RESULTS / "v6a" / "submission.csv"
OUT_DIR = RESULTS / "fusion_v5s1s2s3"
SWAP = ("Lateral Meniscus", "Lateral OA")

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def load_sub(path):
    df = pd.read_csv(path)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    return df.set_index("StudyInstanceUID")


def main():
    base = load_sub(BASE_SUB)
    rad = load_sub(RAD_SUB)
    assert list(base.columns) == TARGETS, f"base columns: {list(base.columns)}"
    assert list(rad.columns) == TARGETS, f"rad columns: {list(rad.columns)}"
    assert base.index.equals(rad.index), "base/rad UID 不一致"

    rad_rank = pd.DataFrame(
        {c: rankdata(rad[c].to_numpy(), method="average") / len(rad)
         for c in TARGETS}, index=rad.index)
    # 保守: 两类各半混
    half = (base[TARGETS] + rad_rank) / 2

    swap = base[TARGETS].copy()
    blend = base[TARGETS].copy()
    for c in SWAP:
        swap[c] = rad_rank[c]
        blend[c] = half[c]

    swap_out = base.reset_index()[["StudyInstanceUID"]]
    blend_out = base.reset_index()[["StudyInstanceUID"]]
    for c in TARGETS:
        swap_out[c] = swap[c]
        blend_out[c] = blend[c]

    swap_out.to_csv(OUT_DIR / "submission_lateral_swap.csv", index=False)
    blend_out.to_csv(OUT_DIR / "submission_lateral_blend.csv", index=False)

    print(f"test studies: {len(base)}")
    for c in TARGETS:
        d_swap = float((swap[c] - base[c]).abs().sum())
        d_blend = float((blend[c] - base[c]).abs().sum())
        flag = " <== SWAPPED" if c in SWAP else ""
        print(f"  {c:20s} base_mean {base[c].mean():.3f} "
              f"|Δswap| {d_swap:6.2f} |Δblend| {d_blend:6.2f}{flag}")
    print(f"\nwritten: {OUT_DIR / 'submission_lateral_swap.csv'}")
    print(f"         {OUT_DIR / 'submission_lateral_blend.csv'}")
    print("裁决提示: LB 在非公开 test 集评分; 先投 swap, 掉分就回退 base。")


if __name__ == "__main__":
    main()
