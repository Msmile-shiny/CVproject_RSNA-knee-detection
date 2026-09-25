# ============================================================
# ★ 集成推理 + 融合 (跳过训练, 纯推理)
#
# 成员: 3 个 seed (best_model_s{42,142,242}.pt)
#       + 可选 1 个 spec 专家 (best_model_spec.pt, 词表标签监督; 仅诊断, 不融合)
# 流程:
#   1. 找到全部成员 checkpoint (best_model_s*.pt)
#   2. 逐个与训练配置交叉核对 (数据侧参数不一致 → 立即报错)
#   3. gold 验证: 每成员 7 窗口 TTA + jitter + 诊断池化 → 逐类 AUC
#   4. test 推理: 每成员同款 TTA
#   5. 融合: 3 seed rank-mean (spec 成员已归档: 离线扫描 5 方案 ≤ +0.001,
#      当前 4 类替换为负 -0.0023 → 不参与融合, 只打印诊断 AUC)
#   6. submission.csv
# ============================================================

print('=' * 60)
print('ENSEMBLE INFERENCE (3 seeds rank fusion; spec = diagnostic only)')
print('=' * 60)

output_dir = Path(CFG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Part 0: 定位成员 checkpoint
# ============================================================

def _iter_pt_files(root, max_depth=2):
    """限深度扫描 best_model_s*.pt — 避免 rglob 遍历竞赛数据集的数万目录。"""
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for p in entries:
            try:
                if p.is_file() and p.name.startswith('best_model_s') and p.name.endswith('.pt'):
                    yield p
                elif p.is_dir() and depth < max_depth:
                    stack.append((p, depth + 1))
            except OSError:
                continue


def find_checkpoints():
    """扫描 ckpt_input 数据集 (+ Kaggle 全 input 兜底), 返回 [(key, path)]。

    key: int seed (s42/s142/s242) 或字符串 'spec' (词表专家, 按文件名识别)。
    seed 升序在前, spec 最后。
    """
    roots = [Path(CFG['ckpt_input'])]
    kg_input = Path('/kaggle/input')
    if kg_input.exists():  # Kaggle 环境: 兜底扫描全部挂载数据集
        roots += [p for p in kg_input.iterdir() if p.is_dir()]

    found = {}
    for root in roots:
        if not root.exists():
            continue
        for fp in sorted(_iter_pt_files(root, max_depth=2)):
            fp_str = str(fp)
            if fp_str in found.values():
                continue
            if fp.name == 'best_model_spec.pt':
                key = 'spec'
            else:
                m = re.match(r'best_model_s(\d+)\.pt$', fp.name)
                if not m:
                    continue
                key = int(m.group(1))
            try:
                ck = torch.load(fp_str, map_location='cpu', weights_only=False)
                # 配置一致性: 文件名的 seed 必须与 checkpoint 内 config 一致
                cfg_seed = (ck.get('config') or {}).get('seed')
                if key != 'spec' and cfg_seed != key:
                    print(f'  skip {fp_str}: filename seed {key} != config seed {cfg_seed}')
                    continue
                found[key] = fp_str
                print(f'  found: {key} <- {fp_str}')
            except Exception as e:
                print(f'  skip {fp_str}: {type(e).__name__}')
    order = sorted(k for k in found if k != 'spec') + \
            (['spec'] if 'spec' in found else [])
    return [(k, found[k]) for k in order]

checkpoints = find_checkpoints()
if not checkpoints:
    raise FileNotFoundError(
        f'未找到任何 best_model_s*.pt。请把训练产物 '
        f'(results/v5s{{1,2,3}}/checkpoints/best_model_s{{42,142,242}}.pt, '
        f'可选 results/v5spec/checkpoints/best_model_spec.pt) '
        f'上传为 Kaggle Dataset 并挂载到本 notebook (CFG["ckpt_input"])。')
print(f'\nMembers: {len(checkpoints)} | keys: {[k for k, _ in checkpoints]}')

# ============================================================
# Part 1: 配置交叉核对 (训练侧 vs 推理侧, 不一致立即停)
# ============================================================

# 数据侧参数必须与推理配置逐项一致; 训练侧独有参数 (lr/batch 等) 不检查
CHECK_KEYS = ['image_size', 'crop_mm', 'cache_slices', 'group_size',
              'center_pct', 'dinov2_variant', 'cls_dim', 'slot_hidden',
              'num_classes', 'unfreeze_layers', 'tta_jitter']

def member_label(key):
    """成员显示名/文件名后缀: int seed → s42, 'spec' → spec。"""
    return f's{key}' if key != 'spec' else 'spec'


ckpt_meta = {}
print('\n--- Checkpoint 交叉核对 ---')
for key, path in checkpoints:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg_ck = ck.get('config') or {}
    mismatches = [k for k in CHECK_KEYS
                  if k in cfg_ck and cfg_ck[k] != CFG.get(k)]
    if ck.get('targets') != TARGET_COLUMNS:
        mismatches.append('targets')
    if ck.get('slots') != SLOTS:
        mismatches.append('slots')
    if mismatches:
        raise ValueError(
            f'{member_label(key)} ({path}) 与推理配置不一致: {mismatches}。\n'
            f'  checkpoint 侧: ' + ' '.join(
                f'{k}={cfg_ck.get(k)}' for k in mismatches if k in cfg_ck) +
            f'\n  推理配置侧:    ' + ' '.join(
                f'{k}={CFG.get(k)}' for k in mismatches if k in CFG) +
            '\n  → 该 checkpoint 不属于本 v5 管线 (288px/130mm/jitter)，'
            '请检查上传的权重。')
    ema_ok = bool(ck.get('ema') and ck['ema'].get('shadow'))
    ckpt_meta[key] = ck
    print(f'  {member_label(key):>5s}: epoch={ck.get("epoch")}, AUC={ck.get("auc", 0):.4f}, '
          f'EMA={"OK" if ema_ok else "MISSING"}')

# ============================================================
# Part 2: 构建推理模型 (与训练 cell 17 同款: 权重全部来自 checkpoint)
# ============================================================

print('\nBuilding inference model (DINOv2-small @ 288px)...')
infer_backbone = timm.create_model(
    CFG['dinov2_variant'], pretrained=False, num_classes=0,
    img_size=CFG['image_size'])

infer_model = MultiViewModel(
    dinov2_model=infer_backbone, n_slots=N_SLOT, cls_dim=CFG['cls_dim'],
    n_classes=CFG['num_classes'], slot_hidden=CFG['slot_hidden'],
    dropout=0.0, unfreeze_layers=CFG['unfreeze_layers'],
).to(DEVICE)
infer_model.eval()


def load_member_weights(ck):
    """加载成员权重 (处理 DataParallel 前缀 + EMA shadow), 返回打印信息。"""
    state_dict = ck['model']
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    if ck.get('ema') and ck['ema'].get('shadow'):
        for name in state_dict:
            if name in ck['ema']['shadow']:
                state_dict[name] = ck['ema']['shadow'][name]
        ema_note = 'EMA'
    else:
        ema_note = 'raw'
    infer_model.load_state_dict(state_dict, strict=False)
    return ema_note


# ============================================================
# Part 3: gold 验证推理 (7 窗口 TTA + jitter + 诊断池化)
# ============================================================

print('\n--- Gold Validation (58 studies) ---')

gold_rows = []
for uid in gold_studies:
    ri = gold_study_index[uid]
    slots_all = torch.from_numpy(GOLD_CACHE[ri].copy())  # [6, 9, H, W]
    m = torch.from_numpy(GOLD_MASK[ri].copy())            # [6]
    windows = torch.stack(
        [slots_all[:, w:w + CFG['group_size']] for w in range(N_WINDOWS)], dim=0)
    gold_rows.append((windows, m, uid))

@torch.no_grad()
def infer_gold_batch(windows_batch, mask_batch, model):
    """TTA + 诊断池化（jitter 视图平均 → per-target 窗口池化）"""
    B = windows_batch.shape[0]
    W = N_WINDOWS
    flat = windows_batch.reshape(B * W, *windows_batch.shape[2:]).to(DEVICE)  # B-major
    flat_mask = mask_batch.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1).to(DEVICE)
    if CFG.get('tta_jitter', False):
        flat = torch.cat([flat, tta_jitter(flat)], dim=0)   # [2*B*W, ...] 原始块在前
        flat_mask = flat_mask.repeat(2, 1)
        n_orig = W
    else:
        n_orig = None
    logits = model(flat, flat_mask)  # [V*B*W, 12]
    logits_v = stack_views(logits, B, W, n_orig)  # [B, V*W, 12]
    return diagnostic_pool(logits_v.cpu(), n_orig=n_orig)  # [B, 12]


def run_gold_inference(model):
    """全 58 gold 推理 → [58, 12] probs。"""
    probs_list, uids_list = [], []
    for start in range(0, len(gold_rows), 8):
        batch = gold_rows[start:start + 8]
        windows_batch = torch.stack([r[0] for r in batch])
        mask_batch = torch.stack([r[1] for r in batch])
        uids = [r[2] for r in batch]
        probs = infer_gold_batch(windows_batch, mask_batch, model)
        probs_list.append(probs)
        uids_list.extend(uids)
    return torch.cat(probs_list).numpy(), uids_list


# gold 真值标签
gold_labels_arr = np.zeros((len(gold_studies), N_CLASSES))
for i, uid in enumerate(gold_studies):
    for j, c in enumerate(TARGET_COLUMNS):
        raw = gold_labels.loc[uid, c] if uid in gold_labels.index else np.nan
        gold_labels_arr[i, j] = float(raw) if not pd.isna(raw) else 0.0


def compute_gold_aucs(probs):
    aucs = {}
    for i, c in enumerate(TARGET_COLUMNS):
        yt, yp = gold_labels_arr[:, i], probs[:, i]
        n_pos = int(yt.sum())
        if n_pos > 0 and n_pos < len(yt):
            try:
                aucs[c] = float(roc_auc_score(yt, yp))
            except Exception:
                aucs[c] = float('nan')
        else:
            aucs[c] = float('nan')
    valid = [v for v in aucs.values() if not math.isnan(v)]
    macro = float(np.mean(valid)) if valid else float('nan')
    return aucs, macro


# ============================================================
# Part 4: test 集 slot 匹配 + 缓存 (训练 cell 17 同款)
# ============================================================

print('\n--- Test Set Slot Matching ---')

test_df = pd.read_csv(comp_input / 'test.csv')
test_df['StudyInstanceUID'] = test_df['StudyInstanceUID'].astype(str)
test_dicom_root = comp_input / 'test_series'


def _find_dicom_files(series_dir):
    """列出目录中的 DICOM 文件（不依赖扩展名，竞赛 test 集 DICOM 无 .dcm 后缀）。"""
    all_files = sorted([f for f in series_dir.iterdir() if f.is_file()])
    dcm = [f for f in all_files if f.suffix == '.dcm']
    return dcm if dcm else [f for f in all_files if not f.name.startswith('.')]


def _scan_test_dicoms(dicom_root):
    """扫描测试集 DICOM 目录，从 header 推断 plane / fluid / fatsat。"""
    rows = []
    root = Path(dicom_root)
    if not root.exists():
        return rows
    for study_dir in sorted(root.iterdir()):
        if not study_dir.is_dir():
            continue
        study_uid = study_dir.name
        for series_dir in sorted(study_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            series_uid = series_dir.name
            dcm_files = _find_dicom_files(series_dir)
            if not dcm_files:
                continue
            try:
                ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)

                iop = getattr(ds, 'ImageOrientationPatient', None)
                plane = 'Axial'
                if iop is not None and len(iop) >= 6:
                    try:
                        row_cos = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
                        col_cos = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
                        normal = np.cross(row_cos, col_cos)
                        dominant = int(np.argmax(np.abs(normal)))
                        plane = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}[dominant]
                    except Exception:
                        pass

                desc = str(getattr(ds, 'SeriesDescription', '')).lower()
                seq_name = str(getattr(ds, 'SequenceName', '')).lower()
                scan_opts = str(getattr(ds, 'ScanOptions', '')).upper()

                fs_kw = ['fs', 'fatsat', 'fat sat', 'stir', 'spair', 'spir', 'we',
                         'water excit', 'tirm', 'fatsup']
                has_fs = any(kw in desc for kw in fs_kw)
                has_fs = has_fs or any(kw in scan_opts for kw in ['FS', 'FATSAT', 'SPAIR', 'SPIR'])

                t1_kw = ['t1', 't1w']
                is_t1 = any(kw in desc or kw in seq_name for kw in t1_kw)
                is_t2 = any(kw in desc or kw in seq_name for kw in ['t2', 't2w'])
                is_pd = any(kw in desc for kw in ['pd', 'pdw', 'proton', 'dp', 'dens'])
                has_fluid = (is_t2 or is_pd) and not is_t1

                rows.append({
                    'StudyInstanceUID': study_uid,
                    'SeriesInstanceUID': series_uid,
                    'Anatomical_Plane': plane,
                    'Fluid_Sensitive': 1 if has_fluid else 0,
                    'Fat_Suppression': 1 if has_fs else 0,
                })
            except Exception:
                continue
    return rows


test_slot_map = {}
test_series_path = comp_input / 'test_series.csv'

if test_series_path.exists():
    test_series = pd.read_csv(test_series_path)
    test_series['StudyInstanceUID'] = test_series['StudyInstanceUID'].astype(str)
    test_series['SeriesInstanceUID'] = test_series['SeriesInstanceUID'].astype(str)
    print(f'test_series.csv: {len(test_series)} series, '
          f'{test_series["StudyInstanceUID"].nunique()} studies')

    test_slot_map, _ = build_study_slot_map(test_series, test_dicom_root)
    csv_studies = len(test_slot_map)

    if csv_studies < max(10, len(test_df) * 0.5):
        print(f'CSV coverage ({csv_studies}/{len(test_df)}) insufficient, '
              f'scanning DICOM headers...')
        dicom_rows = _scan_test_dicoms(test_dicom_root)
        if dicom_rows:
            test_series = pd.DataFrame(dicom_rows)
            test_slot_map, _ = build_study_slot_map(test_series, test_dicom_root)
            print(f'DICOM scan: {len(test_series)} series, '
                  f'{test_series["StudyInstanceUID"].nunique()} studies → '
                  f'{len(test_slot_map)} studies matched')
        else:
            print(f'DICOM scan returned 0 rows, keeping CSV results ({csv_studies} studies)')
else:
    print('test_series.csv not found, scanning DICOM headers...')
    dicom_rows = _scan_test_dicoms(test_dicom_root)
    if dicom_rows:
        test_series = pd.DataFrame(dicom_rows)
        test_slot_map, _ = build_study_slot_map(test_series, test_dicom_root)
        print(f'DICOM scan: {len(test_series)} series, '
              f'{len(test_slot_map)} studies matched')

test_studies = sorted(test_slot_map.keys())
print(f'Test studies with slot match: {len(test_studies)}/{len(test_df)}')

# ---- 测试缓存 (gold 缓存很小, 无需释放) ----
TEST_CACHE, TEST_MASK, test_study_idx = None, None, {}
if len(test_studies) > 0:
    n_test = len(test_studies)
    TEST_CACHE = np.zeros((n_test, N_SLOT, CFG['cache_slices'],
                           CFG['image_size'], CFG['image_size']), dtype=np.uint8)
    TEST_MASK = np.zeros((n_test, N_SLOT), dtype=np.float32)

    for row_idx, study_uid in enumerate(test_studies):
        test_study_idx[study_uid] = row_idx

    t0 = time.time()
    jobs = []
    for row_idx, study_uid in enumerate(test_studies):
        study_slots = test_slot_map[study_uid]
        for slot_idx, (slot_name, plane, fluid, fatsat) in enumerate(SLOTS):
            slot_info = study_slots.get(slot_name)
            if slot_info is not None:
                jobs.append((row_idx, slot_idx, slot_name, plane, slot_info, None))

    print(f'Decoding {len(jobs)} test slot-series...')
    completed, failed = 0, 0
    with ThreadPoolExecutor(max_workers=CFG['pix_threads']) as pool:
        for row_idx, slot_idx, result in pool.map(_read_slot_job, jobs):
            completed += 1
            if result is not None:
                TEST_CACHE[row_idx, slot_idx] = result
                TEST_MASK[row_idx, slot_idx] = 1.0
            else:
                failed += 1
            if completed % 1000 == 0:
                print(f'  [{completed}/{len(jobs)}] {time.time()-t0:.0f}s', flush=True)

    print(f'Test cache: {n_test} studies in {time.time()-t0:.0f}s ({failed} failed)')


@torch.no_grad()
def infer_test_batch(indices, model):
    windows_list, masks_list, empty_mask = [], [], []
    for idx in indices:
        windows_list.append(torch.stack([
            torch.from_numpy(TEST_CACHE[idx, :, w:w + CFG['group_size']].copy())
            for w in range(N_WINDOWS)
        ], dim=0))
        masks_list.append(torch.from_numpy(TEST_MASK[idx].copy()))
        empty_mask.append(TEST_MASK[idx].sum() == 0)

    windows_batch = torch.stack(windows_list)
    mask_batch = torch.stack(masks_list)

    B, W = windows_batch.shape[0], N_WINDOWS
    flat = windows_batch.reshape(B * W, *windows_batch.shape[2:]).to(DEVICE)  # B-major
    flat_mask = mask_batch.unsqueeze(1).expand(B, W, -1).reshape(B * W, -1).to(DEVICE)
    if CFG.get('tta_jitter', False):
        flat = torch.cat([flat, tta_jitter(flat)], dim=0)   # [2*B*W, ...] 原始块在前
        flat_mask = flat_mask.repeat(2, 1)
        n_orig = W
    else:
        n_orig = None
    logits = model(flat, flat_mask)
    probs = diagnostic_pool(
        stack_views(logits, B, W, n_orig).cpu(), n_orig=n_orig)  # [B, C]

    # Studies with no slots → fill 0.5
    for i, is_empty in enumerate(empty_mask):
        if is_empty:
            probs[i] = 0.5
    return probs


def run_test_inference(model):
    """全 test 推理 → [n_test, 12] probs (或 None)。"""
    if TEST_CACHE is None:
        return None
    n_test = TEST_CACHE.shape[0]
    test_probs = np.zeros((n_test, N_CLASSES), dtype=np.float32)
    t1 = time.time()
    for start in range(0, n_test, 8):
        idx = list(range(start, min(start + 8, n_test)))
        test_probs[idx] = infer_test_batch(idx, model).numpy()
        if start % 200 == 0:
            print(f'  [{start}/{n_test}] {time.time()-t1:.0f}s', flush=True)
    return test_probs

# ============================================================
# Part 5: 成员循环 — 逐成员 gold + test 推理
# ============================================================

print('\n--- Member inference ---')
member_gold_probs = {}   # key -> [58, 12]
member_test_probs = {}   # key -> [n_test, 12] (或 None)
member_aucs = {}         # key -> (per-class dict, macro)

for key, path in checkpoints:
    ck = ckpt_meta[key]
    tag = member_label(key)
    print(f'\n[{tag}] Loading {Path(path).name} ...')
    ema_note = load_member_weights(ck)
    print(f'  epoch={ck.get("epoch")}, AUC={ck.get("auc", 0):.4f}, weights={ema_note}')

    t0 = time.time()
    gold_probs, gold_uids = run_gold_inference(infer_model)
    aucs, macro = compute_gold_aucs(gold_probs)
    member_gold_probs[key] = gold_probs
    member_aucs[key] = (aucs, macro)
    print(f'  Gold ({len(gold_uids)} studies, {N_WINDOWS}-window TTA+jitter + diag pool): '
          f'Macro AUC {macro:.4f} ({time.time()-t0:.0f}s)')

    # 保存成员 gold 预测 (与训练产物同格式, 可逐位对比)
    gold_rows_out = []
    for i, uid in enumerate(gold_uids):
        row = {'StudyInstanceUID': uid}
        for j, c in enumerate(TARGET_COLUMNS):
            row[f'true_{c}'] = int(gold_labels_arr[i, j])
            row[f'prob_{c}'] = float(gold_probs[i, j])
        gold_rows_out.append(row)
    pd.DataFrame(gold_rows_out).to_csv(
        output_dir / f'gold_validation_predictions_{tag}.csv', index=False)
    auc_rows = [{'class': c, 'auc': aucs[c], 'n_pos': int(gold_labels_arr[:, i].sum())}
                for i, c in enumerate(TARGET_COLUMNS)]
    pd.DataFrame(auc_rows + [{'class': 'macro_avg', 'auc': macro, 'n_pos': 0}]
                 ).to_csv(output_dir / f'gold_validation_auc_{tag}.csv', index=False)

    if TEST_CACHE is not None:
        test_probs = run_test_inference(infer_model)
        member_test_probs[key] = test_probs
        print(f'  Test inference: {len(test_probs)} studies')

# ============================================================
# Part 6: rank-mean 融合 + 提交
# ============================================================

print('\n--- Fusion ---')

seeds = sorted(k for k in member_gold_probs if k != 'spec')
has_spec = 'spec' in member_gold_probs
if not has_spec:
    print('spec member not found — pure rank-mean over seed members')

# ★ spec 融合已归档 (2026-08-15 离线扫描 scripts/fusion_scan_spec.py, 58 gold):
#   S2 当前 4 类 0.75/0.75 替换 = -0.0023 (MCL -0.0295 拖累)
#   S3 仅 ACL 替换 = +0.0004 (w 任意 — rank 混合整体缩放不改变排序)
#   S4 spec 作第 4 成员 (最佳 w=0.15) = +0.0003
#   S5 逐类 oracle 上限 = +0.0011 —— 连作弊上限都低于 +0.003 阈值
#   → 蒸馏失败 (spec 宏 0.8606), 融合端禁用; 扫到 spec 文件只打印诊断 AUC
SPEC_REPLACE_TARGETS = []
SPEC_W = 0.75


def rank_mean(probs_list):
    """逐类 rankdata 平均 (rank 空间融合, 与 0.91 方案同构)。"""
    acc = np.zeros_like(probs_list[0], dtype=np.float64)
    for p in probs_list:
        r = rankdata(p, axis=0, method='average') / len(p)
        acc += r
    return acc / len(probs_list)


def rank_of(p):
    return rankdata(p, axis=0, method='average') / len(p)


def ensemble_fused(probs_by_member):
    """返回 (final, base): base = seed 成员 rank-mean;
    final = spec 在词表强类上按 0.75+0.75 重投票后的结果。"""
    if len(seeds) > 1:
        base = rank_mean([probs_by_member[s] for s in seeds])
    else:
        base = rank_of(probs_by_member[seeds[0]]) if seeds else None
    if base is None:
        return None, None
    if not has_spec:
        return base, base
    spec_rank = rank_of(probs_by_member['spec'])
    final = base.copy()
    for c in SPEC_REPLACE_TARGETS:
        j = TARGET_COLUMNS.index(c)
        final[:, j] = SPEC_W * base[:, j] + SPEC_W * spec_rank[:, j]
    return final, base


# gold 融合
fused_gold, base_gold = ensemble_fused(member_gold_probs)
print(f'gold: {len(seeds)} seed members rank-mean'
      + ('; spec member found but archived (not fused)' if has_spec else ''))

aucs_f, macro_f = compute_gold_aucs(fused_gold)
print(f'\n{"member":<12s} {"Macro AUC":>10s}')
for s in seeds:
    print(f's{s:<11d} {member_aucs[s][1]:10.4f}')
if has_spec:
    print(f'{"spec":<12s} {member_aucs["spec"][1]:10.4f}')
print(f'{"BASE":<12s} {compute_gold_aucs(base_gold)[1]:10.4f}'
      + ('  (纯 seed rank-mean)' if has_spec else ''))
print(f'{"FUSED":<12s} {macro_f:10.4f}'
      + ('  (= BASE, spec 已归档不参与)' if has_spec else ''))
if has_spec and SPEC_REPLACE_TARGETS:
    print(f'\n  替换类 per-class (base → fused):')
    base_aucs, _ = compute_gold_aucs(base_gold)
    for c in SPEC_REPLACE_TARGETS:
        print(f'    {c:<18s} {base_aucs[c]:.3f} → {aucs_f[c]:.3f} '
              f'({aucs_f[c] - base_aucs[c]:+.3f})')
print(f'\n★ 对照: 各成员读数应与各自训练会话一致 '
      f'(s42≈0.8933 / s142≈0.8899 / s242≈0.8937)。差 >0.005 说明产物错配。'
      + ('\n  spec 只作诊断 (蒸馏上限对照), 融合端已归档不参与。'
         if has_spec else ''))

# 保存融合 gold 预测 + AUC
def gold_rows_of(probs):
    rows = []
    for i, uid in enumerate(gold_studies):
        row = {'StudyInstanceUID': uid}
        for j, c in enumerate(TARGET_COLUMNS):
            row[f'true_{c}'] = int(gold_labels_arr[i, j])
            row[f'prob_{c}'] = float(probs[i, j])
        rows.append(row)
    return rows


pd.DataFrame(gold_rows_of(fused_gold)).to_csv(
    output_dir / 'gold_validation_predictions_fused.csv', index=False)
auc_rows_f = [{'class': c, 'auc': aucs_f[c], 'n_pos': int(gold_labels_arr[:, i].sum())}
              for i, c in enumerate(TARGET_COLUMNS)]
pd.DataFrame(auc_rows_f + [{'class': 'macro_avg', 'auc': macro_f, 'n_pos': 0}]
             ).to_csv(output_dir / 'gold_validation_auc_fused.csv', index=False)
if has_spec and SPEC_REPLACE_TARGETS:
    pd.DataFrame(gold_rows_of(base_gold)).to_csv(
        output_dir / 'gold_validation_predictions_fused_base.csv', index=False)

# ---- test 融合 → submission ----
submission_rows = []
if TEST_CACHE is not None:
    fused_test, _ = ensemble_fused(member_test_probs)
    for row_idx, study_uid in enumerate(test_studies):
        row = {'StudyInstanceUID': study_uid}
        for j, c in enumerate(TARGET_COLUMNS):
            row[c] = float(fused_test[row_idx, j])
        submission_rows.append(row)
else:
    print('No test DICOMs found — filling all studies with 0.5')

submission_df = pd.DataFrame(submission_rows)
full_submission = test_df[['StudyInstanceUID']].merge(
    submission_df, on='StudyInstanceUID', how='left')
for c in TARGET_COLUMNS:
    full_submission[c] = full_submission[c].fillna(0.5)

submission_path = output_dir / 'submission.csv'
full_submission.to_csv(submission_path, index=False)
print(f'\nSubmission saved: {submission_path}')
print(f'  Studies: {len(full_submission)} (expected: {len(test_df)})')
print(f'  Mean prob: {full_submission[TARGET_COLUMNS].values.mean():.4f}')
for c in TARGET_COLUMNS:
    vals = full_submission[c].values
    print(f'  {c:<20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, '
          f'>0.5={np.mean(vals > 0.5):.1%}')

print(f'\nDone!')
print(f'  Members: {len(seeds)} seeds'
      + (' + spec found (archived, not fused)' if has_spec else '')
      + f' | Fused Gold AUC: {macro_f:.4f}')
print(f'  Submission: {submission_path}')
print(f'  Ready to submit to Kaggle!')
