"""Assemble the super-ensemble + OUR v5-member injected notebook.

super-ensemble 原 6 cells 逐字保留 (复刻保真), 插入 3 个新 cell:
  1. gold-emission  (interactive-only, 喂 58-gold α 扫描) — 复用 build_fork_v47
  2. OUR MEMBER     (v5 3-seed inference, exec 隔离 namespace) — 复用 build_fork_v47
  3. OUR BLEND      (fail-closed) — cells_super/ours_blend_cell.py, 插在 cell 4 后 / cell 5 前

用法 (在 notebooks/ 目录下, 或任意目录):
  python notebooks/build_super_ours.py
  python notebooks/build_super_ours.py --ckpt /kaggle/input/<你的-v5-3seed-数据集>
  python notebooks/build_super_ours.py --no-emission    # 缺 reference_code 时显式跳过

依赖:
  - notebooks/cells_infer_lateral/*  (member body 组装源, 已入 git)
  - reference_code/gold_emission_cell.py  (队友本机, 未入 git — 缺了自动跳过 emission 并告警)
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # 使 `import build_fork_v47` 可解析
import build_fork_v47 as bf47

REPLICA = HERE / 'kernel_push_super' / 'kaggle_super_ensemble_replica.ipynb'
BLEND = HERE / 'cells_super' / 'ours_blend_cell.py'
OUTPUT = HERE / 'kaggle_super_ensemble_ours.ipynb'


def make_cell(source):
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]
    return {'cell_type': 'code', 'metadata': {}, 'source': source,
            'outputs': [], 'execution_count': None, 'id': uuid.uuid4().hex}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None,
                    help='v5 3-seed checkpoint 数据集 slug (默认 build_fork_v47 内置 '
                         '/kaggle/input/dinov5-multi-weights; cell 08 有 /kaggle/input 兜底扫描)')
    ap.add_argument('--no-emission', action='store_true',
                    help='跳过 gold-emission cell (reference_code 缺失时)')
    ap.add_argument('--out', default=str(OUTPUT), help='输出 notebook 路径')
    args = ap.parse_args()

    nb = json.loads(REPLICA.read_text(encoding='utf-8'))
    orig = nb['cells']
    assert len(orig) == 6, f'预期 6 cells, 实际 {len(orig)}'

    emission = None
    if args.no_emission:
        print('[skip] gold-emission cell (--no-emission)')
    else:
        try:
            emission = bf47.build_emission_cell_source()
        except FileNotFoundError as e:
            print(f'[warn] reference_code/gold_emission_cell.py 缺失, 跳过 emission: {e}')
            print('       (可加 --no-emission 显式跳过; 缺 emission 则无法交互产 gold_members/)')

    member_kwargs = {'ckpt_input': args.ckpt} if args.ckpt else {}
    member = bf47.build_ours_member_cell_source(**member_kwargs)
    blend = BLEND.read_text(encoding='utf-8')

    cells = list(orig[:4])                 # 0-3 verbatim (md + parent/legacy/rad)
    if emission is not None:
        cells.append(make_cell(emission))  # emission (interactive-only)
    cells.append(make_cell(member))        # our member (v5 3-seed)
    cells.append(orig[4])                  # master blend (verbatim cell 4)
    cells.append(make_cell(blend))         # our blend (new, before rename)
    cells.append(orig[5])                  # rename (verbatim cell 5)

    nb['cells'] = cells
    out = Path(args.out)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')

    print(f'\nwritten: {out}')
    print(f'cells: {len(cells)} (原 6 verbatim + {"emission + " if emission else ""}member + blend)')
    for i, c in enumerate(cells):
        first = (''.join(c['source']).strip().splitlines() or ['(empty)'])[0]
        print(f'  [{i:2d}] {c["cell_type"]:9s} | {first[:70]}')


if __name__ == '__main__':
    main()
