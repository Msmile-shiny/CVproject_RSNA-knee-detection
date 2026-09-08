# 阶段 2 注入方案 — super-ensemble 加自研 v5 成员

> 目标：在 **0.92 基线**（super-ensemble 复刻）上测自研 v5 3-seed（gold 0.8959 / LB 0.886）作第 N+1 个成员。
> 裁决标准：58 gold 上**全局 macro AUC ≥ +0.003** 才启用（lateral-swap 教训：单类决策勿做）。

---

## 1. 注入点定位（已核实）

super-ensemble 复刻 notebook（`notebooks/kernel_push_super/kaggle_super_ensemble_replica.ipynb`）共 **6 cells**（0-indexed）：

| cell | 内容 | 关键产物 |
|---|---|---|
| 0 | markdown 标题 | — |
| 1 | 20×DINOv2 + 5×DINOv3 parent 推理 | `submission.csv`（parent）+ `submission_legacy_fold_blend.csv` |
| 2 | legacy 五折 fold blend | `submission_legacy_fold_blend.csv` |
| 3 | RadImageNet R50 分支 | rad 头 |
| **4** | **master rank blend** | **`submission_master.csv`** |
| 5 | rename `submission_master.csv` → `submission.csv` | 最终提交 |

**master blend 就在 cell 4**（[stage2.md](stage2.md) 里说的「cell 5」是 1-indexed），核心公式：

```python
_parent_rank = _parent[TARGETS].rank(pct=True)          # parent = 20DINOv2+5DINOv3+rad
_legacy_rank = _legacy[TARGETS].rank(pct=True)          # legacy 五折
_b3_rank     = _b3[TARGETS].rank(pct=True)              # 可选 B3

_parent_alpha = 1.0 - _legacy_alpha - _b3_alpha          # 默认 legacy=0.06, b3=0.05 → 0.89? 无 B3 时 =0.94
_master_arr = _parent_alpha*_parent_rank + _legacy_alpha*_legacy_rank + _b3_alpha*_b3_rank
```

- 默认 env（无 B3）：`MASTER_LEGACY_ALPHA=0.06`、`MASTER_B3_ALPHA=0.05`、`MASTER_FORCE_B3=0` → 无 B3 时 `0.94·parent + 0.06·legacy`（即 0.92 配方）。
- 安全护栏：`_legacy_alpha + _b3_alpha >= 0.5` 则 `raise ValueError`（保证 parent 永远主导）。
- 关键 globals（cell 1-3 已定义，注入 cell 直接可用）：`ASSET`、`ROOT`、`COMP`（`== ROOT`）、`DINO`、`CKPT`、`TARGETS`、`TIME_BUDGET`、`log`、`pd`、`np`。

---

## 2. 注入策略：追加 cell，不碰原 6 cell（保真）

和 v47 fork 同款：**原 6 cells 逐字保留**，只插入新 cell。三块：

| 插入位置 | 新 cell | 来源 |
|---|---|---|
| cell 3 后 / cell 4 前 | gold-emission（interactive-only，喂 α 扫描） | 复用 `build_fork_v47.build_emission_cell_source()` |
| 同上（emission 之后） | **OUR MEMBER**（v5 3-seed exec 隔离推理） | 复用 `build_fork_v47.build_ours_member_cell_source()` |
| cell 4 后 / cell 5 前 | **OUR BLEND**（fail-closed） | **新写** `notebooks/cells_super/ours_blend_cell.py` |

关键点：blend cell 必须插在 **cell 4 之后、cell 5 之前**——cell 4 产出 `submission_master.csv`，blend 覆盖它，cell 5 再把融合后的 master 重命名成 `submission.csv`。

---

## 3. 唯一需要新写的代码

队友 v47 的 `cells_fork_v47/ours_blend_cell.py` 几乎原样复用——super-ensemble 里 `COMP`/`TARGETS`/`log`/`pd`/`np` 全都有定义。**只改两处路径**：

| v47 版 | super-ensemble 版 |
|---|---|
| `_ours_primary = submission.csv` | `submission_master.csv` |
| `_ours_pres = submission_v47_theirs_only.csv` | `submission_master_pure920.csv` |

完整代码已写好：[notebooks/cells_super/ours_blend_cell.py](notebooks/cells_super/ours_blend_cell.py)。

---

## 4. 组装脚本（本机跑，产出注入版 notebook）

脚本已落地 [notebooks/build_super_ours.py](notebooks/build_super_ours.py)（依赖队友本机 `reference_code/gold_emission_cell.py`，缺了会自动跳过 emission 并告警；`--ckpt` / `--no-emission` / `--out` 可选，已实测跑通产出 8 cells）。核心拼装逻辑如下：

```python
import json, uuid
from pathlib import Path
import build_fork_v47 as bf47  # 复用 member + emission builder

HERE = Path(__file__).resolve().parent
BLEND = (HERE / 'cells_super' / 'ours_blend_cell.py').read_text(encoding='utf-8')
REPLICA = HERE / 'kernel_push_super' / 'kaggle_super_ensemble_replica.ipynb'
OUT = HERE / 'kaggle_super_ensemble_ours.ipynb'

def make_cell(src):
    return {'cell_type': 'code', 'metadata': {}, 'outputs': [],
            'execution_count': None, 'id': uuid.uuid4().hex,
            'source': [l + '\n' for l in src.split('\n')]}

nb = json.loads(REPLICA.read_text(encoding='utf-8'))
orig = nb['cells']
assert len(orig) == 6
cells = (list(orig[:4])                                    # 0-3 verbatim
         + [make_cell(bf47.build_emission_cell_source())]  # emission (interactive-only)
         + [make_cell(bf47.build_ours_member_cell_source())]  # our member (v5 3-seed)
         + [orig[4]]                                       # master blend (verbatim)
         + [make_cell(BLEND)]                              # our blend (new)
         + [orig[5]])                                      # rename (verbatim)
nb['cells'] = cells
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'written {OUT} ({len(cells)} cells)')
```

> 注意：member builder 默认把 checkpoint 路径换成 `'/kaggle/input/dinov5-multi-weights'`（v5 3-seed 权重数据集），
> cell 08 有 `/kaggle/input` 兜底扫描，slug 不同也能找到。上传前确认该数据集已挂到 super-ensemble notebook。

---

## 5. 两档提交 + α 扫描 workflow

| 档 | `OUR_MEMBER_ACTIVE` | `OUR_ALPHA` | 行为 |
|---|---|---|---|
| ① 纯复刻基线 | `False` | 0.10 | scoring 时 blend cell 零开销，`submission_master.csv` 原样 → 0.92 |
| ② 注入成员 | `True` | α（扫描定） | `submission_master.csv = (1-α)·master + α·ours_rank_mean` |

**裁决流程**（每步 [SOLO]，0 配额）：

1. 先跑 ① 确认 0.92 基线可复现（其实队友已提交命中 0.92，可跳过）。
2. 改 `OUR_MEMBER_ACTIVE=True` + 初始 `OUR_ALPHA=0.05~0.10`，**交互跑 3 行占位 test**（分钟级）→ 下载 `gold_members/` + `submission_ours_only.csv`。
3. 本地 58 gold 上做 **α 扫描**（`OUR_ALPHA` 从 0.02~0.15 网格）→ **只看全局 macro AUC**。
4. `Δ macro ≥ +0.003` → 用该 α `ACTIVE=True` 提交，记 LB；`< +0.003` → `ACTIVE=False` 保持纯复刻，转阶段 3。
5. 每次提交记录到 [stage2.md](stage2.md)「结果记录」。

**fail-closed 保证**：任何异常（schema drift / 非有限值 / 越界）→ `submission_master_pure920.csv` 回填 `submission_master.csv`，最终提交仍是纯 0.92，不会因为注入出错而掉分。

---

## 6. 实操清单

- [ ] 本机跑组装脚本产出 `kaggle_super_ensemble_ours.ipynb`（需队友 `reference_code/gold_emission_cell.py`）
- [ ] 确认 v5 3-seed 权重数据集已上传并挂到该 notebook（cell 08 会打印找到的 seeds）
- [ ] `OUR_MEMBER_ACTIVE=False` 交互冒烟 → 确认 `submission_master.csv` 原样、`ours_only` 正常产出
- [ ] `ACTIVE=True` + α 扫描 → 58 gold 全局 macro 裁决
- [ ] 通过 → 提交；记录 LB 到 [stage2.md](stage2.md)
