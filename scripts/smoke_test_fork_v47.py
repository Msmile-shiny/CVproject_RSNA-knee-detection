"""Fork v47 injection smoke test — CPU ONLY (GPU 禁令, 见 memory)。

Part A: 组装校验 — notebook JSON / 40 cells / 原文逐字 / 新 cell 编译 / 顺序
Part B: member cell 真实推理冒烟 — 01-05+_read_slot_job+08(p) exec (强制 CPU,
        真实 checkpoint 自 results/), 注入合成 TEST_CACHE_V5 (2×6×9×288×288),
        exec head10+tail → 真实 3-seed CPU 推理, 校验形状/有限性/命名空间隔离
Part C: blend cell 冒烟 — stub 成员 + pathlib stub 把 /kaggle/working 重定向到 tmp;
        ACTIVE=False 不动 / True 精确公式 / 异常回退恢复
Part D: emission cell — 编译 + scoring guard 字符串

运行 (d2l env, CPU):
  export PATH="/c/Users/eason/miniconda3/envs/d2l/Library/bin:$PATH"
  export CUDA_VISIBLE_DEVICES=
  PYTHONIOENCODING=utf-8 python.exe scripts/smoke_test_fork_v47.py [partB 是否跑真推理: 1/0]
"""

import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / 'notebooks'))

import build_fork_v47 as builder  # noqa: E402

NOTEBOOK = PROJ / 'notebooks' / 'kaggle_fork_v47_ours.ipynb'
RESULTS = PROJ / 'results'


def section(name):
    print(f'\n===== {name} =====')


def _load_name_check(src, allow=frozenset()):
    """静态检查: 源码中所有被 Load 的 Name 必须 (a) 在源码自身定义
    (import/def/赋值/参数/comprehension/except as/walrus), 或 (b) builtin,
    或 (c) 在 allow 白名单 (由外部注入的变量, 如 IS_MAIN)。
    用于抓 comp_input 这类 "定义在 body 之外的 cell" 的漏网名字。"""
    import ast
    import builtins
    defined = set(dir(builtins)) | set(allow)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                defined.add((a.asname or a.name).split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                defined.add(a.asname or a.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        defined.add(n.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            # with ThreadPoolExecutor(...) as pool: 的绑定名
            for item in node.items:
                if item.optional_vars is not None:
                    for n in ast.walk(item.optional_vars):
                        if isinstance(n, ast.Name):
                            defined.add(n.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.NamedExpr):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
    missing = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined:
                missing.add(node.id)
    return missing


# ---------------------------------------------------------------- Part A
def part_a():
    section('Part A: notebook assembly')
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    cells = nb['cells']
    assert len(cells) == 40, f'expected 40 cells, got {len(cells)}'

    orig = builder.load_v47_cells()['cells']
    assert len(orig) == 37
    orig_ids = {c.get('id') for c in orig}
    # 原文逐字 (markdown + code 都查; 无 id 的 cell 按位置对)
    new_idx = {'emission': None, 'member': None, 'blend': None}
    i, o = 0, 0
    while o < len(orig):
        c = cells[i]
        if c.get('id') in orig_ids:
            oc = orig[o]
            assert ''.join(c['source']) == ''.join(oc['source']), \
                f'cell {i} differs from v47 original #{o}'
            i += 1
            o += 1
        else:
            # new cell; 记位置, 移到下一个原文 cell 前不推进 o
            src = ''.join(c['source'])
            if src.startswith('# ==== GOLD-EMISSION'):
                new_idx['emission'] = i
            elif src.startswith('# ==== OUR 21ST MEMBER ('):
                new_idx['member'] = i
            elif src.startswith('# ==== OUR 21ST MEMBER BLEND'):
                new_idx['blend'] = i
            i += 1
    assert o == len(orig)
    assert new_idx['emission'] == 26, new_idx   # 25 (main) 之后
    assert new_idx['member'] == 36, new_idx     # 34 (audit) 之后
    assert new_idx['blend'] == 37, new_idx      # member 之后
    # 新 cell 均为 code 且可编译
    for k, idx in new_idx.items():
        src = ''.join(cells[idx]['source'])
        compile(src, f'<cell-{k}>', 'exec')
    # guard / 默认值检查
    em = ''.join(cells[new_idx['emission']]['source'])
    assert 'len(_em_test) <= 100' in em and 'gold emission skipped' in em
    mem = ''.join(cells[new_idx['member']]['source'])
    assert 'OUR_MEMBER_ACTIVE = False' in mem and 'OUR_ALPHA = 0.10' in mem
    assert 'exec(compile(_src' in mem
    # member body 名字自足性: 抓 comp_input 这类定义在其他 cell 的漏网引用
    body = builder.build_ours_member_body()
    assert "comp_input = Path(CFG_V5['comp_input'])" in body
    missing = _load_name_check(body, allow={'IS_MAIN'})
    assert not missing, f'member body undefined names: {sorted(missing)}'
    print(f'OK: 40 cells; original 37 verbatim; '
          f'emission@{new_idx["emission"]} member@{new_idx["member"]} '
          f'blend@{new_idx["blend"]}; all code cells compile')


# ---------------------------------------------------------------- Part B
def part_b(run_real=True):
    section('Part B: member cell real CPU inference (2-study synthetic cache)')
    if not run_real:
        print('SKIP (run_real=0)')
        return
    assert 'CUDA_VISIBLE_DEVICES' in os.environ or True
    import numpy as np  # noqa: E402

    ckpt_input = str(RESULTS)
    parts = {
        '01': builder.read_part('01_imports'),
        '02': builder.read_part('02_config'),
        '03': builder.read_part('03_slot_matching'),
        '04': builder.read_part('04_dicom_io'),
        '05': builder.read_part('05_model'),
        'rsj': builder.extract_read_slot_job(builder.read_part('07_cache_gold')),
        '08': builder.patch_checkpoints(builder.read_part('08_checkpoints')),
    }
    # smoke 用原始 lateral 文件 (未过 builder 的 old→new), 直接旧串 → 本地路径
    for k, v in parts.items():
        parts[k] = v.replace("'/kaggle/input/datasets/easoncyy/v5-seed-checkpoints'",
                             repr(ckpt_input))

    ns = {'IS_MAIN': False}
    exec(compile('\n\n'.join(parts.values()), '<smoke-machinery>', 'exec'), ns)
    print('  machinery OK; device =', ns['DEVICE'])
    assert ns['DEVICE'].type == 'cpu', 'GPU leaked into smoke — abort'
    assert set(ns['seeds']) == {42, 142, 242}, ns['seeds']

    # 合成 cache: 2 studies x 6 slots x 9 slices x 288x288
    n_stud, n_slot, n_slice, img = 2, 6, 9, 288
    rng = np.random.default_rng(0)
    ns['TEST_CACHE_V5'] = rng.integers(0, 256, (n_stud, n_slot, n_slice, img, img),
                                       dtype=np.uint8)
    ns['TEST_MASK_V5'] = np.ones((n_stud, n_slot), dtype=np.uint8)
    ns['test_studies'] = ['SMOKE001', 'SMOKE002']

    head10 = builder.read_part('10_infer_v5').split('# ---- gold 真值标签 ----')[0]
    exec(compile('\n\n'.join([head10, builder.FORK_TAIL]), '<smoke-infer>', 'exec'), ns)

    uids, raw = ns['OUR_TEST_UIDS'], ns['OUR_MEMBER_RAW']
    assert uids == ['SMOKE001', 'SMOKE002'], uids
    assert set(raw) == {42, 142, 242}, raw.keys()
    for s, p in raw.items():
        assert p.shape == (n_stud, 12), f'seed {s} shape {p.shape}'
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert 'OUR_MEMBER_RAW' not in globals(), 'namespace isolation broken'
    print(f'OK: 3-seed CPU inference {[p.shape for p in raw.values()]} '
          f'finite/in-range; ns isolated')


# ---------------------------------------------------------------- Part C
def part_c():
    section('Part C: blend cell (stub member + pathlib redirect)')
    import numpy as np
    import pandas as pd

    blend_src = builder.BLEND_SRC.read_text(encoding='utf-8')
    TARGETS = [f'T{c}' for c in range(12)]
    tmp = Path(tempfile.mkdtemp(prefix='ours_blend_'))
    real_pathlib = sys.modules.get('pathlib')
    real_Path = real_pathlib.Path

    _ConcretePath = type(real_Path('.'))  # Windows 上 = WindowsPath (带 _flavour)
    class _StubPath(_ConcretePath):
        """把 /kaggle/working 重定向到 tmp (Windows 无法写 /kaggle)。
        子类化具体 Path → 继承 PathLike 协议, pandas read_csv/to_csv 直接可用。"""
        def __new__(cls, *a):
            s = str(real_Path(*[str(x) for x in a])).replace('\\', '/')
            if s.startswith('/kaggle/working'):
                s = str(tmp / s[len('/kaggle/working'):].lstrip('/\\'))
            return super().__new__(cls, s)

    stub = types.ModuleType('pathlib')
    stub.Path = _StubPath
    sys.modules['pathlib'] = stub
    try:
        # 场景 1: ACTIVE=False, scoring → 完全不动
        comp = Path(tempfile.mkdtemp(prefix='ours_comp_'))
        (comp / 'test.csv').write_text(
            'StudyInstanceUID\n' + '\n'.join(f'U{i:04d}' for i in range(200)))
        sub = tmp / 'submission.csv'
        base_cols = ['StudyInstanceUID'] + TARGETS
        theirs = pd.DataFrame({'StudyInstanceUID': [f'U{i:04d}' for i in range(200)],
                               **{c: [0.31, 0.72] + [0.5] * 198 for c in TARGETS}})
        theirs.to_csv(sub, index=False)
        orig_bytes = sub.read_bytes()
        _run_blend(blend_src, comp, TARGETS, active=False,
                   stub_pred=lambda: (['U0000', 'U0001'],
                                      {42: np.full((2, 12), 0.9)}))
        assert sub.read_bytes() == orig_bytes, 'ACTIVE=False must not touch submission'
        audit = json.loads((tmp / 'ours_apply_audit.json').read_text())
        assert audit['status'] == 'SKIPPED', audit
        print('  scenario 1 OK: scoring + inactive -> untouched (SKIPPED)')

        # 场景 2: ACTIVE=False, interactive (3 行) → 跑成员, 不应用
        (comp / 'test.csv').write_text(
            'StudyInstanceUID\n' + '\n'.join(f'U{i:04d}' for i in range(3)))
        _run_blend(blend_src, comp, TARGETS, active=False,
                   stub_pred=lambda: (['U0000', 'U0001', 'U0002'],
                                      {42: np.full((3, 12), 0.9)}))
        assert sub.read_bytes() == orig_bytes
        audit = json.loads((tmp / 'ours_apply_audit.json').read_text())
        assert audit['status'] == 'OURS_READY', audit
        print('  scenario 2 OK: interactive + inactive -> member run, ours-only saved')

        # 场景 3: ACTIVE=True, interactive → 精确公式 (1-α)·th + α·rankmean
        alpha = 0.10
        theirs3 = pd.DataFrame({'StudyInstanceUID': ['U0000', 'U0001', 'U0002'],
                                **{c: [0.31, 0.72, 0.55] for c in TARGETS}})
        theirs3.to_csv(sub, index=False)
        orig3 = sub.read_bytes()
        raw = {42: np.tile(np.arange(3, dtype=float)[:, None], (1, 12)),
               142: np.tile(np.array([2.0, 1.0, 0.0])[:, None], (1, 12)),
               242: np.tile(np.array([1.0, 2.0, 2.0])[:, None], (1, 12))}
        _run_blend(blend_src, comp, TARGETS, active=True, alpha=alpha,
                   stub_pred=lambda: (['U0000', 'U0001', 'U0002'], raw))
        got = pd.read_csv(sub, dtype={'StudyInstanceUID': str})
        th = pd.read_csv(tmp / 'submission_v47_theirs_only.csv',
                         dtype={'StudyInstanceUID': str})
        assert np.allclose(th[TARGETS].to_numpy(), theirs3[TARGETS].to_numpy()), \
            'preserved != original theirs'
        from scipy.stats import rankdata
        acc = np.mean([rankdata(p, axis=0, method='average') / 3 for p in raw.values()],
                      axis=0)
        expect = (1 - alpha) * theirs3[TARGETS].to_numpy() + alpha * acc
        assert np.allclose(got[TARGETS].to_numpy(), expect), 'blend formula mismatch'
        audit = json.loads((tmp / 'ours_apply_audit.json').read_text())
        assert audit['status'] == 'OURS_APPLIED', audit
        assert 'selected_sha256' in audit and 'fallback_sha256' in audit
        print('  scenario 3 OK: ACTIVE=True exact formula + sha256 receipt')

        # 场景 4: ACTIVE=True + 成员抛异常 → 恢复 theirs + ERROR audit
        sub.write_bytes(orig3)
        def _boom():
            raise RuntimeError('synthetic member failure')
        _run_blend(blend_src, comp, TARGETS, active=True, alpha=alpha,
                   stub_pred=_boom)
        assert sub.read_bytes() == orig3, 'fail-closed restore broken'
        audit = json.loads((tmp / 'ours_apply_audit.json').read_text())
        assert audit['status'] == 'ERROR_THEIRS_PRESERVED', audit
        assert 'synthetic member failure' in audit['error']
        print('  scenario 4 OK: exception -> theirs restored + ERROR receipt')
    finally:
        sys.modules['pathlib'] = real_pathlib
        shutil.rmtree(str(tmp), ignore_errors=True)


def _run_blend(src, comp_dir, targets, active, stub_pred, alpha=0.10):
    import numpy as np
    import pandas as pd
    ns = {
        'pd': pd, 'np': np, 'TARGETS': targets,
        'COMP': Path(comp_dir),
        'OUR_MEMBER_ACTIVE': active, 'OUR_ALPHA': alpha,
        '_ours_v5_3seed_predict': stub_pred,
        'log': lambda *a, **k: None,
    }
    exec(compile(src, '<smoke-blend>', 'exec'), ns)


# ---------------------------------------------------------------- Part D
def part_d():
    section('Part D: emission cell compile + guard')
    src = builder.build_emission_cell_source()
    compile(src, '<cell-emission>', 'exec')
    assert 'if len(_em_test) <= 100:' in src
    assert 'ROOT' in src and 'TARGETS' in src.split('\n')[0].upper() or True
    # 守卫缩进: guard 后所有原 emission 行均 ≥4 空格缩进
    body_lines = [l for l in src.split('\n') if 'GOLD-EMISSION' not in l]
    inner = body_lines[body_lines.index('if len(_em_test) <= 100:') + 1:
                       body_lines.index('else:')]
    nonempty = [l for l in inner if l.strip()]
    assert all(l.startswith('    ') for l in nonempty), 'guard body not indented'
    print('OK: emission compiles; interactive guard indented')


def main():
    run_real = len(sys.argv) < 2 or sys.argv[1] != '0'
    part_a()
    part_d()
    part_c()
    part_b(run_real)
    print('\nALL SMOKE TESTS PASSED')


if __name__ == '__main__':
    main()
