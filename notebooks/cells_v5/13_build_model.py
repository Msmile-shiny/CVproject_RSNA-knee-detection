# ============================================================
# v5: Build model, optimizer, scheduler + EMA — 288px + WeightedSoftBCE
# ============================================================

if IS_MAIN: print('Loading DINOv2 backbone...')

dinov2_backbone = timm.create_model(
    CFG['dinov2_variant'], pretrained=False, num_classes=0,
    img_size=CFG['image_size'],
)

# ★ 竞赛禁网，从本地 Kaggle Dataset 加载预训练权重
weights_path = Path(CFG.get('dinov2_weights', ''))
if weights_path.exists():
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)

    # ★ DINOv2 原生 img_size=518 → pos_embed [1, 1370, 384] (37×37 grid)
    # v5 模型 img_size=288 → grid 288//14=20 → pos_embed [1, 401, 384]，需要插值
    if 'pos_embed' in state_dict:
        pos_ckpt = state_dict['pos_embed']          # [1, N_ckpt, dim]
        pos_model = dinov2_backbone.pos_embed.data   # [1, N_model, dim]
        if pos_ckpt.shape != pos_model.shape:
            cls_ckpt = pos_ckpt[:, :1, :]             # CLS token 保留
            patch_ckpt = pos_ckpt[:, 1:, :]           # patch tokens

            grid_ckpt = int(math.isqrt(patch_ckpt.shape[1]))
            grid_model = int(math.isqrt(pos_model.shape[1] - 1))

            patch_ckpt = patch_ckpt.reshape(1, grid_ckpt, grid_ckpt, -1).permute(0, 3, 1, 2)
            # bicubic 插值 → 目标 grid
            patch_interp = F.interpolate(
                patch_ckpt, size=(grid_model, grid_model), mode='bicubic',
                antialias=True)
            patch_interp = patch_interp.permute(0, 2, 3, 1).reshape(1, -1, pos_model.shape[-1])
            state_dict['pos_embed'] = torch.cat([cls_ckpt, patch_interp], dim=1)
            if IS_MAIN:
                print(f'  pos_embed interpolated: [{grid_ckpt}×{grid_ckpt}] → [{grid_model}×{grid_model}]')

    dinov2_backbone.load_state_dict(state_dict, strict=True)
    if IS_MAIN: print(f'  DINOv2 pretrained weights loaded: {weights_path}')
elif IS_MAIN:
    print(f'  WARNING: DINOv2 weights not found at {weights_path} — using random init!')

model = MultiViewModel(
    dinov2_model=dinov2_backbone,
    n_slots=N_SLOT,
    cls_dim=CFG['cls_dim'],
    n_classes=CFG['num_classes'],
    slot_hidden=CFG['slot_hidden'],
    dropout=CFG['dropout'],
    unfreeze_layers=CFG['unfreeze_layers'],
).to(DEVICE)

if N_GPUS > 1:
    model = nn.DataParallel(model)
    if IS_MAIN: print(f'[Model] DataParallel across {N_GPUS} GPUs')

# Separate LR
backbone_params, head_params = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if 'dinov2' in name:
        backbone_params.append(p)
    else:
        head_params.append(p)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': CFG['backbone_lr']},
    {'params': head_params, 'lr': CFG['lr']},
], weight_decay=CFG['weight_decay'])

# ★ v5: 置信度加权软 BCE — 融合软标签 (text×OOF teacher) 直接作为训练目标
criterion = WeightedSoftBCELoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=CFG['lr_t0'], T_mult=CFG['lr_t_mult'],
    eta_min=CFG['lr_eta_min'])

scaler = torch.amp.GradScaler('cuda') if CFG['mixed_precision'] else None

# ★ EMA
ema = EMAModel(model.module if N_GPUS > 1 else model, decay=CFG['ema_decay'])

if IS_MAIN:
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f'Optimizer: backbone {n_backbone/1e6:.1f}M @ lr={CFG["backbone_lr"]}')
    print(f'           head     {n_head/1e6:.1f}M @ lr={CFG["lr"]}')
    print(f'Loss: WeightedSoftBCE (confidence-weighted fused soft labels)')
    print(f'EMA: decay={CFG["ema_decay"]}')
    print('Model v5 ready.')
