"""Offline fusion scan v2: 3-seed base × 异架构成员 × 公开成员 (58 gold, CPU only).

用法:
  python scripts/fusion_scan_v6.py [rad_csv_path] [--ext oof_csv] [--spec] [--ext-name NAME]

成员池:
  s42/s142/s242   v5 3-seed rank-mean base (gold 0.8959, 已定)
  spec            v5spec 成员 (--spec 启用; spec 阶段曾被裁决弃用, 复检用)
  rad             v6a RadImageNet 成员 (默认 results/v6a/gold_validation_predictions_rad.csv)
  ext             外部公开全训练矩阵成员 (如 v52_oof.csv: StudyInstanceUID + 12 原始目标
                  + fold + is_gold; 自动取 is_gold==1 子集对齐 58 gold)
                  — 附 fold-z 形态 (fold 内 z 标准化, 防公开成员 fold 分布漂移)

场景:
  V1  baseline: 纯 3-seed rank-mean (期望 0.8959)
  V2  +rad 全局 w 扫描: (3·base + w·rad)/(3+w)
  V3  +ext 全局 w 扫描 (raw 与 fold-z 两种形态)
  V4  +rad+ext 2D w 扫描 (w ∈ [0.25,0.5,1,2] 网格)
  V5  贪心前向选择: 从 base 出发, 每步在剩余成员中扫 w, 提升 < MIN_GAIN 即停
  V6  逐类 oracle: 每类在 {base, rad, ext, spec} 凸包内取最优 (作弊天花板)
  D   多样性诊断: 成员两两 rank-Pearson + 单成员 macro AUC

裁决准则 (spec 经验, 2026-08-15):
  oracle 上限 < +0.003 → 融合无杠杆, 直接弃用;
  oracle 可观但逐类最优依赖单类噪声 (58 研究) → 优先全局简单权重 (V2-V5),
  只在该类成员 AUC 显著强于 base (≥+0.02) 时才考虑该类 mix。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

SEED_FILES = {
    "s42": RESULTS / "v5s1" / "gold_validation_predictions_s42.csv",
    "s142": RESULTS / "v5s2" / "gold_validation_predictions_s142.csv",
    "s242": RESULTS / "v5s3" / "gold_validation_predictions_s242.csv",
}
SPEC_FILE = RESULTS / "v5spec" / "gold_validation_predictions_spec.csv"

W_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
W2 = [0.25, 0.5, 1.0, 2.0]          # 2D 扫描网格 (16 组合)
MIN_GAIN = 0.003                     # gold 宏提升低于此值视为噪声


def load_member(path):
    """我们的 gold CSV: StudyInstanceUID + true_*/prob_* 交错列."""
    df = pd.read_csv(path)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df = df.set_index("StudyInstanceUID")
    probs = np.stack([df[f"prob_{c}"].to_numpy() for c in TARGETS], axis=1)
    y = np.stack([df[f"true_{c}"].to_numpy() for c in TARGETS], axis=1).astype(int)
    return df.index.to_numpy(), y, probs


def load_ext_member(path):
    """公开全训练矩阵: StudyInstanceUID + 12 原始目标 + fold + is_gold.
    返回 (full_probs, full_folds, gold_probs) — gold_probs 未对齐 (调用方 reindex)."""
    df = pd.read_csv(path)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    full_probs = df[TARGETS].to_numpy(dtype=np.float64)
    folds = df["fold"].to_numpy()
    gold_df = df[df["is_gold"].astype(int) == 1]
    return df["StudyInstanceUID"].to_numpy(), full_probs, folds, \
        gold_df["StudyInstanceUID"].to_numpy(), gold_df[TARGETS].to_numpy()


def fold_z(full_probs, folds):
    """fold 内 z 标准化 (每 fold 每类), 防 fold 分布漂移的成员形态."""
    z = np.empty_like(full_probs)
    for f in np.unique(folds):
        m = folds == f
        mu = full_probs[m].mean(0)
        sd = full_probs[m].std(0) + 1e-8
        z[m] = (full_probs[m] - mu) / sd
    return z


def per_class_aucs(y, probs):
    aucs = []
    for j in range(len(TARGETS)):
        yj = y[:, j]
        aucs.append(roc_auc_score(yj, probs[:, j]) if len(np.unique(yj)) > 1 else np.nan)
    return np.array(aucs)


def macro(y, probs):
    return np.nanmean(per_class_aucs(y, probs))


def ranks(p):
    return np.stack([rankdata(p[:, j]) for j in range(p.shape[1])], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rad_csv", nargs="?", default=None,
                    help="rad 成员 gold CSV (默认 results/v6a/gold_validation_predictions_rad.csv)")
    ap.add_argument("--ext", default=None, help="外部公开 OOF CSV (含 fold/is_gold 列)")
    ap.add_argument("--ext-name", default="ext", help="外部成员显示名")
    ap.add_argument("--spec", action="store_true", help="启用 v5spec 成员复检")
    args = ap.parse_args()

    # ---- 加载 base (3 seed) ----
    uids_list, y_list, p_list = [], [], []
    for key, path in SEED_FILES.items():
        uids, y, p = load_member(path)
        uids_list.append(uids)
        y_list.append(y)
        p_list.append(p)
    assert all((u == uids_list[0]).all() for u in uids_list[1:]), "seed UID mismatch"
    gold_uids = uids_list[0]
    y = y_list[0]
    n, ncls = y.shape
    seed_ranks = [ranks(p) for p in p_list]
    base = np.mean(seed_ranks, axis=0)
    base_macro = macro(y, base)
    print(f"gold studies: {n}")
    print(f"=== V1 baseline (3-seed rank-mean) ===  macro {base_macro:.4f}")

    pool = {}          # name -> rank matrix (已对齐 58 gold)
    pool_auc = {}      # name -> single-member macro

    # ---- rad ----
    rad_path = Path(args.rad_csv) if args.rad_csv else \
        RESULTS / "v6a" / "gold_validation_predictions_rad.csv"
    if rad_path.exists():
        rad_uids, y_rad, p_rad = load_member(rad_path)
        assert (rad_uids == gold_uids).all(), "rad UID mismatch"
        pool["rad"] = ranks(p_rad)
        pool_auc["rad"] = macro(y, p_rad)
    else:
        print(f"  [skip] rad CSV not found: {rad_path}")

    # ---- spec ----
    if args.spec:
        sp_uids, y_sp, p_sp = load_member(SPEC_FILE)
        assert (sp_uids == gold_uids).all(), "spec UID mismatch"
        pool["spec"] = ranks(p_sp)
        pool_auc["spec"] = macro(y, p_sp)

    # ---- ext (公开成员) ----
    ext_raw = ext_z = None
    if args.ext:
        uid_full, full_probs, folds, gold_sub, gold_probs = load_ext_member(args.ext)
        inter = np.intersect1d(gold_uids, gold_sub)
        assert len(inter) == n, \
            f"ext gold 子集与本地 gold 交集 {len(inter)} != {n}"
        idx = {u: i for i, u in enumerate(gold_sub)}
        order = [idx[u] for u in gold_uids]
        ext_raw = ranks(gold_probs[order])
        z_full = fold_z(full_probs, folds)
        z_gold = np.stack([z_full[uid_full == u].mean(0) for u in gold_uids])
        ext_z = ranks(z_gold)
        pool[args.ext_name] = ext_raw
        pool_auc[args.ext_name] = macro(y, gold_probs[order])
        print(f"  ext={args.ext_name}: 全矩阵 {len(uid_full)} 行, gold 子集 {len(gold_sub)}")
        print(f"    raw gold macro {pool_auc[args.ext_name]:.4f} | "
              f"fold-z gold macro {macro(y, z_gold):.4f}")

    # ---- D: 多样性诊断 ----
    print(f"\n=== D 多样性诊断 ===")
    members = {"base": base, **pool}
    names = list(members)
    print(f"{'member':>10s} {'macro':>7s} | " + " ".join(f"{m:>9s}" for m in names[1:]))
    corr = {}
    for a in names:
        row = []
        for b in names:
            if a == b:
                corr[(a, b)] = 1.0
            else:
                key = tuple(sorted((a, b)))
                if key not in corr:
                    corr[key] = np.mean([
                        np.corrcoef(members[a][:, j], members[b][:, j])[0, 1]
                        for j in range(ncls)])
                row.append(corr[key])
        ma = f"{pool_auc[a]:.4f}" if a in pool_auc else f"{base_macro:.4f}"
        print(f"{a:>10s} {ma:>7s} | " + " ".join(f"{r:9.3f}" for r in row[1:]))
    print(f"  (seed 成员间两两 0.959-0.966; 新成员与 base 越低 = 结构多样性越高)")

    # ---- 扫描工具 ----
    def scan_add(cur, cur_w, name, r):
        """在 cur(权重 cur_w) 上加成员 r, 扫 w. 返回 (best_w, best_auc, delta)."""
        best = (0.0, -1.0)
        for w in W_GRID:
            a = macro(y, (cur_w * cur + w * r) / (cur_w + w))
            if a > best[1]:
                best = (w, a)
        return best[0], best[1], best[1] - macro(y, cur)

    # ---- V2: +rad ----
    if "rad" in pool:
        print(f"\n=== V2 +rad 全局 w 扫描: (3·base + w·rad)/(3+w) ===")
        for w in W_GRID:
            a = macro(y, (3.0 * base + w * pool["rad"]) / (3.0 + w))
            print(f"  w={w:<5} macro {a:.4f} ({a - base_macro:+.4f})")

    # ---- V3: +ext (raw / fold-z) ----
    if ext_raw is not None:
        print(f"\n=== V3 +{args.ext_name} 全局 w 扫描 ===")
        for label, r in [("raw", ext_raw), ("fold-z", ext_z)]:
            parts = []
            for w in W_GRID:
                a = macro(y, (3.0 * base + w * r) / (3.0 + w))
                parts.append(f"w={w}:{a:.4f}({a - base_macro:+.4f})")
            print(f"  [{label}] " + "  ".join(parts))

    # ---- V4: +rad+ext 2D ----
    if "rad" in pool and ext_raw is not None:
        print(f"\n=== V4 +rad+{args.ext_name} 2D w 扫描 ===")
        combos = []
        for wr in W2:
            for we in W2:
                cand = (3.0 * base + wr * pool["rad"] + we * ext_raw) / (3.0 + wr + we)
                combos.append((macro(y, cand), wr, we))
        combos.sort(reverse=True)
        for a, wr, we in combos[:5]:
            print(f"  w_rad={wr:<5} w_ext={we:<5} macro {a:.4f} ({a - base_macro:+.4f})")

    # ---- V5: 贪心前向选择 ----
    print(f"\n=== V5 贪心前向选择 (提升 < {MIN_GAIN:+.3f} 停) ===")
    selected = {"base"}
    cur, cur_w = base.copy(), 3.0
    while True:
        best = None
        for name, r in pool.items():
            if name in selected:
                continue
            w, a, d = scan_add(cur, cur_w, name, r)
            if best is None or a > best[1]:
                best = (name, w, a, d)
        if best is None or best[3] < MIN_GAIN:
            break
        name, w, a, d = best
        cur = (cur_w * cur + w * pool[name]) / (cur_w + w)
        cur_w += w
        selected.add(name)
        print(f"  +{name} w={w:<5} macro {a:.4f} ({d:+.4f})")
    print(f"  最终: {sorted(selected)} | macro {macro(y, cur):.4f} "
          f"({macro(y, cur) - base_macro:+.4f})")

    # ---- V6: 逐类 oracle (凸包上限) ----
    print(f"\n=== V6 逐类 oracle (作弊天花板) ===")
    mix_a = pool.get("rad")
    mix_b = pool.get(args.ext_name)
    best_cols, best_w = [], {}
    base_auc = per_class_aucs(y, base)
    for j, c in enumerate(TARGETS):
        best = (base_auc[j], "base", {})
        for k, r in pool.items():
            a = roc_auc_score(y[:, j], r[:, j])
            if a > best[0]:
                best = (a, k, {"w": np.inf})
        if mix_a is not None and mix_b is not None:
            for wr in W2:
                for we in W2:
                    col = (3.0 * base[:, j] + wr * mix_a[:, j]
                           + we * mix_b[:, j]) / (3.0 + wr + we)
                    a = roc_auc_score(y[:, j], col)
                    if a > best[0]:
                        best = (a, "mix", {"wr": wr, "we": we})
        best_cols.append(best[0])
        best_w[c] = best
        if best[1] != "base":
            print(f"  {c:20s} {best[1]:>5s} {str(best[2]):>16s} "
                  f"{base_auc[j]:.4f} -> {best[0]:.4f} ({best[0] - base_auc[j]:+.4f})")
    o_macro = np.mean(best_cols)
    print(f"  oracle macro {o_macro:.4f} ({o_macro - base_macro:+.4f})")

    # ---- 裁决 ----
    print("\n=== 裁决 ===")
    if o_macro - base_macro < MIN_GAIN:
        print(f"oracle 上限 {o_macro - base_macro:+.4f} < {MIN_GAIN:+.3f} "
              f"→ 融合端无杠杆, 维持 base")
    else:
        strong = [c for j, c in enumerate(TARGETS)
                  if any(pool_auc.get(k, 0) - base_auc[j] >= 0.02 for k in pool)]
        print(f"oracle 上限 {o_macro - base_macro:+.4f} ≥ {MIN_GAIN:+.3f} → 有杠杆")
        print(f"成员显著强类 (成员 AUC ≥ base+0.02): {strong if strong else '无'}")
        print(f"建议: 58 gold 上优先取全局简单权重 (V2-V5 结果); "
              f"逐类 mix 仅当该类有成员显著强且相邻 w 皆正才考虑")


if __name__ == "__main__":
    main()
