# ============================================================
# v6a: Build model (RadImageNet R50 frozen) + optimizer + EMA
# ============================================================

if IS_MAIN: print('Loading RadImageNet ResNet50 backbone...')

# torchvision resnet50 (fc→Identity, 输出 [B, 2048])
backbone = torchvision.models.resnet50(weights=None)
backbone.fc = nn.Identity()

# ★ 竞赛禁网，从本地 Kaggle Dataset 加载官方 RadImageNet 权重
#   (本地 scripts/convert_radimagenet_r50.py 转换: h5 → torchvision 键 +
#   conv bias 吸收进 BN running_mean — eval 模式严格等价, 冻结编码器专用)
weights_path = Path(CFG.get('rad_weights', ''))
if weights_path.exists():
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)

    # 'backbone.{child_idx}.*' → torchvision 键 (0=conv1 1=bn1 4..7=layer1..4)
    sd = {}
    for k, v in state_dict.items():
        assert k.startswith('backbone.'), f'unexpected key: {k}'
        idx = int(k[len('backbone.'):].split('.')[0])
        rest = k[len(f'backbone.{idx}.'):]
        if idx == 0:
            sd['conv1.' + rest] = v
        elif idx == 1:
            sd['bn1.' + rest] = v
        else:
            sd[f'layer{idx - 3}.' + rest] = v

    backbone.load_state_dict(sd, strict=True)
    if IS_MAIN: print(f'  RadImageNet pretrained weights loaded: {weights_path}')
elif IS_MAIN:
    print(f'  WARNING: RadImageNet weights not found at {weights_path} — using random init!')

model = RadResNetModel(
    backbone=backbone,
    n_slots=N_SLOT,
    feature_dim=CFG['feature_dim'],
    n_classes=CFG['num_classes'],
    slot_hidden=CFG['slot_hidden'],
    dropout=CFG['dropout'],
    unfreeze_layers=CFG['unfreeze_layers'],
).to(DEVICE)

if N_GPUS > 1:
    model = nn.DataParallel(model)
    if IS_MAIN: print(f'[Model] DataParallel across {N_GPUS} GPUs')

# Separate LR — v6a 编码器全冻结 → 通常只有 head 参数组 (空组跳过, 防 AdamW 报错)
backbone_params, head_params = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if 'backbone' in name:
        backbone_params.append(p)
    else:
        head_params.append(p)

param_groups = [{'params': head_params, 'lr': CFG['lr']}]
if backbone_params:
    param_groups.append({'params': backbone_params, 'lr': CFG['backbone_lr']})
optimizer = torch.optim.AdamW(param_groups, weight_decay=CFG['weight_decay'])

# ★ v5 同款: 置信度加权软 BCE — 融合软标签 (text×OOF teacher) 直接作为训练目标
criterion = WeightedSoftBCELoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=CFG['lr_t0'], T_mult=CFG['lr_t_mult'],
    eta_min=CFG['lr_eta_min'])

scaler = torch.amp.GradScaler('cuda') if CFG['mixed_precision'] else None

# ★ EMA (只 shadow requires_grad 参数 → 冻结 backbone 自动排除)
ema = EMAModel(model.module if N_GPUS > 1 else model, decay=CFG['ema_decay'])

if IS_MAIN:
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f'Optimizer: backbone {n_backbone/1e6:.1f}M '
          f'@ lr={CFG["backbone_lr"] if backbone_params else "frozen"}')
    print(f'           head     {n_head/1e6:.1f}M @ lr={CFG["lr"]}')
    print(f'Loss: WeightedSoftBCE (confidence-weighted fused soft labels)')
    print(f'EMA: decay={CFG["ema_decay"]}')
    print('Model v6a ready.')
