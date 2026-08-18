# ============================================================
# ★ 成员 checkpoint 定位 + 配置交叉核对 + 推理模型构建 (跳过训练, 纯推理)
#
# 成员:
#   best_model_s42.pt / best_model_s142.pt / best_model_s242.pt  v5 seed 成员 (288px)
#   best_model_rad.pt                                            v6a rad 成员 (224px)
#   best_model_spec.pt                                           可选 spec 专家 (仅诊断, 不融合)
# 流程:
#   1. 扫描 /kaggle/input 全部数据集找 checkpoint
#   2. 逐个与对应管线配置交叉核对 (数据侧参数不一致 → 立即报错)
#   3. 构建两个推理模型 (权重全部来自 checkpoint, 无需预训练权重数据集)
# ============================================================

print('=' * 60)
print('CHECKPOINT DISCOVERY + CROSS-CHECK')
print('=' * 60)

output_dir = Path(CFG_V5['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

def _iter_pt_files(root, max_depth=2):
    """限深度扫描 best_model_s*.pt / best_model_rad.pt — 避免 rglob 遍历竞赛数据集数万目录。"""
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for p in entries:
            try:
                if p.is_file() and p.name.startswith('best_model_') and p.name.endswith('.pt'):
                    yield p
                elif p.is_dir() and depth < max_depth:
                    stack.append((p, depth + 1))
            except OSError:
                continue


def find_checkpoints():
    """扫描 ckpt_input 数据集 (+ Kaggle 全 input 兜底), 返回 {key: path}。

    key: int seed (s42/s142/s242) / 'spec' (词表专家, 仅诊断) / 'rad' (v6a R50)。
    """
    roots = [Path(CFG_V5['ckpt_input'])]
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
            if fp.name == 'best_model_rad.pt':
                key = 'rad'
            elif fp.name == 'best_model_spec.pt':
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
                if key not in ('rad', 'spec') and cfg_seed != key:
                    print(f'  skip {fp_str}: filename seed {key} != config seed {cfg_seed}')
                    continue
                found[key] = fp_str
                print(f'  found: {key} <- {fp_str}')
            except Exception as e:
                print(f'  skip {fp_str}: {type(e).__name__}')
    return found

checkpoints = find_checkpoints()

seeds = sorted(k for k in checkpoints if isinstance(k, int))
has_spec = 'spec' in checkpoints
has_rad = 'rad' in checkpoints

if not seeds:
    raise FileNotFoundError(
        f'未找到任何 best_model_s{{seed}}.pt。请把训练产物 '
        f'(results/v5s{{1,2,3}}/checkpoints/best_model_s{{42,142,242}}.pt) '
        f'上传为 Kaggle Dataset 并挂载到本 notebook (CFG_V5["ckpt_input"] 或任意 /kaggle/input 数据集)。')
if not has_rad:
    raise FileNotFoundError(
        f'未找到 best_model_rad.pt — lateral_swap 提交必须有 rad 成员。'
        f'请把 results/v6a/checkpoints/best_model_rad.pt (98.6MB) 与 v5 seed '
        f'checkpoint 一起上传为 Kaggle Dataset (可放入同一数据集, 本 cell 自动扫描)。')

print(f'\nMembers: {len(seeds)} seeds'
      + (' + spec (diagnostic only)' if has_spec else '')
      + ' + rad')

# ============================================================
# Part 1: 配置交叉核对 (训练侧 vs 推理侧, 不一致立即停)
# ============================================================

# 数据侧参数必须与推理配置逐项一致; 训练侧独有参数 (lr/batch 等) 不检查
CHECK_KEYS_V5 = ['image_size', 'crop_mm', 'cache_slices', 'group_size',
                 'center_pct', 'dinov2_variant', 'cls_dim', 'slot_hidden',
                 'num_classes', 'unfreeze_layers', 'tta_jitter']
CHECK_KEYS_RAD = ['image_size', 'crop_mm', 'cache_slices', 'group_size',
                  'center_pct', 'feature_dim', 'slot_hidden',
                  'num_classes', 'unfreeze_layers', 'tta_jitter']

def member_label(key):
    """成员显示名/文件名后缀: int seed → s42, 'spec' → spec, 'rad' → rad。"""
    return f's{key}' if key != 'spec' and key != 'rad' else key


ckpt_meta = {}
print('\n--- Checkpoint 交叉核对 ---')
for key in sorted(checkpoints, key=lambda k: (k != 'rad', k != 'spec')):
    path = checkpoints[key]
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg_ck = ck.get('config') or {}
    check_keys = CHECK_KEYS_RAD if key == 'rad' else CHECK_KEYS_V5
    my_cfg = CFG_RAD if key == 'rad' else CFG_V5
    mismatches = [k for k in check_keys
                  if k in cfg_ck and cfg_ck[k] != my_cfg.get(k)]
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
                f'{k}={my_cfg.get(k)}' for k in mismatches if k in my_cfg) +
            f'\n  → 该 checkpoint 不属于本管线'
            f'({"224px/7片 rad" if key == "rad" else "288px/9片 v5"}), '
            '请检查上传的权重。')
    ema_ok = bool(ck.get('ema') and ck['ema'].get('shadow'))
    ckpt_meta[key] = ck
    print(f'  {member_label(key):>5s}: epoch={ck.get("epoch")}, AUC={ck.get("auc", 0):.4f}, '
          f'EMA={"OK" if ema_ok else "MISSING"}')

# ============================================================
# Part 2: 构建推理模型 (权重全部来自 checkpoint)
# ============================================================

print('\nBuilding inference models...')

# v5 成员: DINOv2-small @ 288px
infer_backbone_v5 = timm.create_model(
    CFG_V5['dinov2_variant'], pretrained=False, num_classes=0,
    img_size=CFG_V5['image_size'])
infer_model_v5 = MultiViewModel(
    dinov2_model=infer_backbone_v5, n_slots=N_SLOT, cls_dim=CFG_V5['cls_dim'],
    n_classes=CFG_V5['num_classes'], slot_hidden=CFG_V5['slot_hidden'],
    dropout=0.0, unfreeze_layers=CFG_V5['unfreeze_layers'],
).to(DEVICE)
infer_model_v5.eval()
print('  v5 MultiViewModel (DINOv2-small @288px) built')

# rad 成员: RadImageNet R50 @ 224px (checkpoint 已含全部权重, 无需 rad 预训练数据集)
infer_backbone_rad = torchvision.models.resnet50(weights=None)
infer_backbone_rad.fc = nn.Identity()
infer_model_rad = RadResNetModel(
    backbone=infer_backbone_rad, n_slots=N_SLOT, feature_dim=CFG_RAD['feature_dim'],
    n_classes=CFG_RAD['num_classes'], slot_hidden=CFG_RAD['slot_hidden'],
    dropout=0.0, unfreeze_layers=CFG_RAD['unfreeze_layers'],
).to(DEVICE)
infer_model_rad.eval()
print('  rad RadResNetModel (R50 @224px) built')


def load_member_weights(model, ck):
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
    model.load_state_dict(state_dict, strict=False)
    return ema_note
