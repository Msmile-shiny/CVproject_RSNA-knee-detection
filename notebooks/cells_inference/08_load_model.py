# ============================================================
# Load Model — DINOv2 权重 + best_model.pt checkpoint
# ============================================================

print('Loading DINOv2 backbone...')

dinov2_backbone = timm.create_model(
    CFG['dinov2_variant'], pretrained=False, num_classes=0,
    img_size=CFG['image_size'],
)

# ★ 加载 DINOv2 预训练权重（含 pos_embed 插值）
weights_path = Path(CFG['dinov2_weights'])
if weights_path.exists():
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)

    if 'pos_embed' in state_dict:
        pos_ckpt = state_dict['pos_embed']
        pos_model = dinov2_backbone.pos_embed.data
        if pos_ckpt.shape != pos_model.shape:
            cls_ckpt = pos_ckpt[:, :1, :]
            patch_ckpt = pos_ckpt[:, 1:, :]
            grid_ckpt = int(math.isqrt(patch_ckpt.shape[1]))
            grid_model = int(math.isqrt(pos_model.shape[1] - 1))
            patch_ckpt = patch_ckpt.reshape(1, grid_ckpt, grid_ckpt, -1).permute(0, 3, 1, 2)
            patch_interp = F.interpolate(
                patch_ckpt, size=(grid_model, grid_model), mode='bicubic',
                antialias=True)
            patch_interp = patch_interp.permute(0, 2, 3, 1).reshape(1, -1, pos_model.shape[-1])
            state_dict['pos_embed'] = torch.cat([cls_ckpt, patch_interp], dim=1)
            print(f'  pos_embed interpolated: [{grid_ckpt}x{grid_ckpt}] -> [{grid_model}x{grid_model}]')

    dinov2_backbone.load_state_dict(state_dict, strict=True)
    print(f'  DINOv2 weights loaded: {weights_path}')
else:
    raise FileNotFoundError(f'DINOv2 weights not found: {weights_path}')

# ---- Build model ----
model = MultiViewModel(
    dinov2_model=dinov2_backbone,
    n_slots=N_SLOT, cls_dim=CFG['cls_dim'],
    n_classes=CFG['num_classes'], slot_hidden=CFG['slot_hidden'],
    dropout=0.0, unfreeze_layers=CFG['unfreeze_layers'],
).to(DEVICE)
model.eval()

# ---- Load best checkpoint ----
best_ckpt_path = Path(CFG['best_model'])
if not best_ckpt_path.exists():
    raise FileNotFoundError(
        f'Best model not found: {best_ckpt_path}\n'
        'Upload best_model.pt as a Kaggle Dataset and update CFG["best_model"].')

print(f'Loading checkpoint: {best_ckpt_path}')
ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)

state_dict = ckpt['model']
first_key = next(iter(state_dict))
if first_key.startswith('module.'):
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

# ★ 优先使用 EMA 权重
if ckpt.get('ema') and ckpt['ema'].get('shadow'):
    for name in state_dict:
        ema_key = name
        if ema_key in ckpt['ema']['shadow']:
            state_dict[name] = ckpt['ema']['shadow'][ema_key]
    print('  Using EMA weights for inference')

model.load_state_dict(state_dict, strict=False)
model.eval()
print(f'  Model loaded: epoch={ckpt.get("epoch")}, AUC={ckpt.get("auc", 0):.4f}')

# Report GPU memory
if torch.cuda.is_available():
    print(f'  GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB')
