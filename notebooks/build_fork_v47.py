"""Assemble the v47 fork notebook (kaggle_fork_v47_ours.ipynb).

v47 原始 37 个 cell 全部逐字保留 (复刻保真), 追加 3 个 cell:
  1. gold-emission (交互运行限定: scoring 时 test.csv>100 行 → 跳过, 保预算)
     插入位置: cell 25 (main run) 之后 / cell 26 (DINOv3 段) 之前
  2. OUR 21ST MEMBER — v5 3-seed base 推理, exec 隔离 namespace (零全局污染)
  3. OUR BLEND (fail-closed) — 插在 cell 34 (runtime audit) 之后
     (cell 34 校验 v47 自身哈希链, 我们的融合必须在审计通过之后做)

member cell 由 notebooks/cells_infer_lateral/ 的 v5 部分组装:
  01_imports + 02_config + 03_slot_matching + 04_dicom_io + 05_model
  + _read_slot_job (自 07_cache_gold 提取) + 08_checkpoints (patch: rad 可选)
  + 09_test_cache (patch: 跳过 rad 缓存) + 10_infer_v5 (仅 head + fork tail)
"""

import json
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
LATERAL = HERE / 'cells_infer_lateral'
V47_JSON = HERE.parent / 'reference_code' / 'v47_source.py'
EMISSION_SRC = HERE.parent / 'reference_code' / 'gold_emission_cell.py'
BLEND_SRC = HERE / 'cells_fork_v47' / 'ours_blend_cell.py'
OUTPUT = HERE / 'kaggle_fork_v47_ours.ipynb'

LAT_PART_NAMES = ['01_imports', '02_config', '03_slot_matching', '04_dicom_io', '05_model']


def read_part(name):
    return (LATERAL / f'{name}.py').read_text(encoding='utf-8')


def extract_read_slot_job(src_07):
    """从 07_cache_gold.py 提取 _read_slot_job (09_test_cache 依赖)。"""
    pre = src_07.split('def _build_gold_cache')[0]
    idx = pre.index('def _read_slot_job(args):')
    return pre[idx:].rstrip() + '\n'


def patch_checkpoints(src):
    """08_checkpoints: rad 成员改为可选 (fork 只融合 3 seed base)。"""
    old_raise = (
        "if not has_rad:\n"
        "    raise FileNotFoundError(\n"
        "        f'未找到 best_model_rad.pt — lateral_swap 提交必须有 rad 成员。'\n"
        "        f'请把 results/v6a/checkpoints/best_model_rad.pt (98.6MB) 与 v5 seed '\n"
        "        f'checkpoint 一起上传为 Kaggle Dataset (可放入同一数据集, 本 cell 自动扫描)。')"
    )
    new_warn = (
        "if not has_rad:\n"
        "    print('  (rad 成员未挂载 — fork 只融合 3 seed base, 跳过 rad)')"
    )
    assert old_raise in src, '08 patch 1 anchor not found'
    src = src.replace(old_raise, new_warn)

    old_members = (
        "print(f'\\nMembers: {len(seeds)} seeds'\n"
        "      + (' + spec (diagnostic only)' if has_spec else '')\n"
        "      + ' + rad')"
    )
    new_members = (
        "print(f'\\nMembers: {len(seeds)} seeds'\n"
        "      + (' + spec (diagnostic only)' if has_spec else '')\n"
        "      + (' + rad' if has_rad else ''))"
    )
    assert old_members in src, '08 patch 2 anchor not found'
    src = src.replace(old_members, new_members)

    old_rad_build = (
        "# rad 成员: RadImageNet R50 @ 224px (checkpoint 已含全部权重, 无需 rad 预训练数据集)\n"
        "infer_backbone_rad = torchvision.models.resnet50(weights=None)\n"
        "infer_backbone_rad.fc = nn.Identity()\n"
        "infer_model_rad = RadResNetModel(\n"
        "    backbone=infer_backbone_rad, n_slots=N_SLOT, feature_dim=CFG_RAD['feature_dim'],\n"
        "    n_classes=CFG_RAD['num_classes'], slot_hidden=CFG_RAD['slot_hidden'],\n"
        "    dropout=0.0, unfreeze_layers=CFG_RAD['unfreeze_layers'],\n"
        ").to(DEVICE)\n"
        "infer_model_rad.eval()\n"
        "print('  rad RadResNetModel (R50 @224px) built')"
    )
    new_rad_build = (
        "if has_rad:\n"
        "    # rad 成员: RadImageNet R50 @ 224px (checkpoint 已含全部权重, 无需 rad 预训练数据集)\n"
        "    infer_backbone_rad = torchvision.models.resnet50(weights=None)\n"
        "    infer_backbone_rad.fc = nn.Identity()\n"
        "    infer_model_rad = RadResNetModel(\n"
        "        backbone=infer_backbone_rad, n_slots=N_SLOT, feature_dim=CFG_RAD['feature_dim'],\n"
        "        n_classes=CFG_RAD['num_classes'], slot_hidden=CFG_RAD['slot_hidden'],\n"
        "        dropout=0.0, unfreeze_layers=CFG_RAD['unfreeze_layers'],\n"
        "    ).to(DEVICE)\n"
        "    infer_model_rad.eval()\n"
        "    print('  rad RadResNetModel (R50 @224px) built')"
    )
    assert old_rad_build in src, '08 patch 3 anchor not found'
    return src.replace(old_rad_build, new_rad_build)


def patch_test_cache(src):
    """09_test_cache: 跳过 rad 224px 缓存 (省 ~30-40 min 解码时间)。"""
    old = (
        "TEST_CACHE_RAD, TEST_MASK_RAD = _build_test_cache(\n"
        "    CFG_RAD['image_size'], CFG_RAD['cache_slices'], CFG_RAD['crop_mm'], 'rad-224')"
    )
    new = "# (fork: 跳过 rad 224px 缓存 — 只融合 3 seed base)"
    assert old in src, '09 patch anchor not found'
    return src.replace(old, new)


FORK_TAIL = '''# ---- fork 专用: 3 seed test 推理 + 原始概率 (rank 在融合 cell 做) ----
if len(seeds) != 3:
    raise RuntimeError(f'expected exactly 3 v5 seed checkpoints, found {sorted(seeds)}')
OUR_TEST_UIDS = list(test_studies)
OUR_MEMBER_RAW = {}
for key in seeds:
    ck = ckpt_meta[key]
    tag = member_label(key)
    print(f'\\n[ours {tag}] {Path(checkpoints[key]).name} '
          f'(epoch={ck.get("epoch")}, AUC={ck.get("auc", 0):.4f})')
    load_member_weights(infer_model_v5, ck)
    if TEST_CACHE_V5 is None:
        raise RuntimeError('test cache is empty; cannot run our member')
    OUR_MEMBER_RAW[key] = run_test_inference(
        TEST_CACHE_V5, TEST_MASK_V5, infer_model_v5,
        CFG_V5['group_size'], N_WINDOWS_V5, CFG_V5['tta_jitter'], tag)
print(f'\\nours member done: {len(OUR_TEST_UIDS)} studies, {len(OUR_MEMBER_RAW)} seeds')
'''


def build_ours_member_body(ckpt_input=None, comp_input=None):
    """组装 member cell 的 exec body (可注入路径覆盖供本地冒烟)。"""
    src_10 = read_part('10_infer_v5')
    head_10 = src_10.split('# ---- gold 真值标签 ----')[0].rstrip() + '\n'
    body = '\n'.join([
        read_part('01_imports').rstrip(),
        read_part('02_config').rstrip(),
        # 06_load_gold 不在 body (它加载 train gold 标签, 测试推理不需要),
        # 但 09_test_cache 裸用 comp_input 这个名 — 在此补定义
        "comp_input = Path(CFG_V5['comp_input'])",
        read_part('03_slot_matching').rstrip(),
        read_part('04_dicom_io').rstrip(),
        read_part('05_model').rstrip(),
        extract_read_slot_job(read_part('07_cache_gold')).rstrip(),
        patch_checkpoints(read_part('08_checkpoints')).rstrip(),
        patch_test_cache(read_part('09_test_cache')).rstrip(),
        head_10.rstrip(),
        FORK_TAIL.rstrip(),
    ])
    # fork 默认挂载真实 slug (lateral-swap 同款; 08 仍有 /kaggle/input 兜底扫描)
    body = body.replace("'/kaggle/input/datasets/easoncyy/v5-seed-checkpoints'",
                        "'/kaggle/input/dinov5-multi-weights'")
    if ckpt_input is not None:
        body = body.replace("'/kaggle/input/dinov5-multi-weights'", repr(ckpt_input))
    if comp_input is not None:
        body = body.replace("'/kaggle/input/competitions/rsna-knee-abnormality-detection'",
                            repr(comp_input))
    return body


MEMBER_WRAPPER = """# ==== OUR 21ST MEMBER (v5 3-seed base, 288px DINOv2-small) — isolated-namespace inference ====
# ★ OUR_MEMBER_ACTIVE: True → submission.csv = (1-OUR_ALPHA)·v47_theirs + OUR_ALPHA·ours_rank_mean
#   False (default) → 纯 v47 复刻; scoring 时本 cell 零开销 (直接跳过)
# 两档提交: ① False 先复刻 0.93 基线 → ② True 测第 21 成员 (本地 58-gold 全局裁决 α)
OUR_MEMBER_ACTIVE = False
OUR_ALPHA = 0.10

def _ours_v5_3seed_predict():
    '''v5 3-seed test 推理 → (uids, {seed: raw_probs})。
    exec 隔离执行 (独立 namespace), 不污染本 notebook globals —
    避免与 v47 的 SLOTS / Model / SlotHead / IMG 等全局名冲突。'''
    _ns = {'IS_MAIN': False}
    _src = r'''{BODY}'''
    exec(compile(_src, '<ours-v5-3seed>', 'exec'), _ns)
    return (list(_ns['OUR_TEST_UIDS']), dict(_ns['OUR_MEMBER_RAW']))
"""


def build_ours_member_cell_source(ckpt_input=None, comp_input=None):
    body = build_ours_member_body(ckpt_input=ckpt_input, comp_input=comp_input)
    assert "'''" not in body, 'member body contains triple single-quotes (wrapper clash)'
    return MEMBER_WRAPPER.replace('{BODY}', body)


def build_emission_cell_source():
    """gold-emission cell + 交互限定 guard (scoring: test.csv>100 行 → 跳过)。"""
    src = EMISSION_SRC.read_text(encoding='utf-8').rstrip()
    indented = '\n'.join(
        ('    ' + line if line.strip() else '') for line in src.split('\n'))
    guard = (
        "# ==== GOLD-EMISSION CELL (v47 fork, interactive-only) ====\n"
        "_em_test = pd.read_csv(ROOT / 'test.csv')\n"
        "if len(_em_test) <= 100:\n"
        f"{indented}\n"
        "else:\n"
        "    log('scoring run detected: gold emission skipped "
        "(hidden test, budget preserved)')\n"
    )
    return guard


def make_cell(cell_type, source):
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]
    return {
        'cell_type': cell_type,
        'metadata': {},
        'source': source,
        'outputs': [],
        'execution_count': None,
        'id': uuid.uuid4().hex,
    }


def load_v47_cells():
    raw = V47_JSON.read_text(encoding='utf-8')
    # v47_source.py 是 kernels/pull 的 blob.source 原文 (notebook JSON 字符串)
    if raw.lstrip().startswith('{'):
        nb = json.loads(raw)
    else:
        blob = json.loads(raw)['blob']
        nb = json.loads(blob['sourceNullable'])
    return nb


def main():
    nb = load_v47_cells()
    orig_cells = nb['cells']
    assert len(orig_cells) == 37, f'unexpected v47 cell count {len(orig_cells)}'

    cells = list(orig_cells[:26])                                   # 0-25 verbatim
    cells.append(make_cell('code', build_emission_cell_source()))   # emission
    cells.extend(orig_cells[26:35])                                 # 26-34 verbatim
    cells.append(make_cell('code', build_ours_member_cell_source()))  # our member
    cells.append(make_cell('code', BLEND_SRC.read_text(encoding='utf-8')))  # our blend
    cells.extend(orig_cells[35:])                                   # license markdowns

    # v47 原始 notebook 存在重复 cell id (Kaggle 编辑器复制遗留, 不影响执行);
    # 保原文不动, 仅对重复 id 追加 _dupN 后缀使其唯一
    seen = {}
    for c in cells:
        if c['cell_type'] != 'code':
            continue
        cid = c.get('id') or uuid.uuid4().hex
        if cid in seen:
            seen[cid] += 1
            c['id'] = f'{cid}_dup{seen[cid]}'
            print(f'  [dedup] cell id {cid} -> {c["id"]}')
        else:
            seen[cid] = 1

    out_nb = {
        'cells': cells,
        'metadata': nb.get('metadata', {}),
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(out_nb, f, indent=1, ensure_ascii=False)

    print(f'Notebook written: {OUTPUT}')
    print(f'Cells: {len(cells)} (v47 original 37 verbatim + emission + member + blend)')
    for i, c in enumerate(cells):
        src = ''.join(c['source'])
        first = src.strip().splitlines()[0] if src.strip() else '(empty)'
        tag = '  <-- NEW' if c.get('id') not in {cc.get('id') for cc in orig_cells} else ''
        print(f'  [{i:2d}] {c["cell_type"]:9s} | {first[:80]}{tag}')


if __name__ == '__main__':
    main()
