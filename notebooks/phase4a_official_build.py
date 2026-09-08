# Phase 4A: exact official DINOv3-L architecture, loaded entirely offline.
WEIGHT_SHA256 = '385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9'

def _find_one(pattern, description):
    hits = list(Path('/kaggle/input').rglob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f'Expected exactly one {description}; found {hits}')
    return hits[0]

weights_path = _find_one('OrthoFoudation-L.pth', 'OrthoFoudation-L.pth below /kaggle/input')
source_hits = list(Path('/kaggle/input').rglob('dinov3/hub/backbones.py'))
if not source_hits:
    import zipfile
    source_zip = _find_one('dinov3-source-6876159.zip', 'fixed official DINOv3 source ZIP')
    extracted_root = Path('/kaggle/working/dinov3-source-6876159')
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(extracted_root)
    source_hits = list(extracted_root.rglob('dinov3/hub/backbones.py'))
if len(source_hits) != 1:
    raise FileNotFoundError(f'Expected exactly one official DINOv3 source tree; found {source_hits}')
source_file = source_hits[0]
if weights_path.stat().st_size != 1_213_056_638:
    raise RuntimeError('Wrong OrthoFoundation file size; use the 1,213,056,638-byte Git-LFS binary')

dinov3_root = source_file.parents[2]
sys.path.insert(0, str(dinov3_root))
from dinov3.hub.backbones import dinov3_vitl16

def build_official_dinov3_backbone(load_medical_weights=False):
    global audit
    backbone = dinov3_vitl16(pretrained=False)
    if load_medical_weights:
        import hashlib
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if digest != WEIGHT_SHA256:
            raise RuntimeError(f'Unexpected OrthoFoundation SHA256: {digest}')
        raw = torch.load(weights_path, map_location='cpu', weights_only=True)
        if not isinstance(raw, dict) or not raw or not all(k.startswith('backbone.') for k in raw):
            raise RuntimeError('Unsupported OrthoFoundation checkpoint container')
        state = {k.removeprefix('backbone.'): v for k, v in raw.items()}
        backbone.load_state_dict(state, strict=True)
        audit = {
            'checkpoint': str(weights_path), 'checkpoint_bytes': weights_path.stat().st_size,
            'checkpoint_sha256': digest, 'matched_tensors': len(state),
            'target_tensors': len(backbone.state_dict()), 'parameter_coverage': 1.0,
            'architecture': 'official facebookresearch/dinov3 dinov3_vitl16',
        }
        Path(CFG['output_dir']).mkdir(parents=True, exist_ok=True)
        (Path(CFG['output_dir']) / 'phase4a_weight_audit.json').write_text(json.dumps(audit, indent=2))
        print('OrthoFoundation load audit:', audit)
        del raw, state
        gc.collect()
    return backbone

if IS_MAIN:
    print('Creating official DINOv3-L architecture for OrthoFoundation...')
backbone = build_official_dinov3_backbone(load_medical_weights=True)

model = MultiViewModel(
    dinov2_model=backbone, n_slots=N_SLOT, cls_dim=CFG['cls_dim'],
    n_classes=CFG['num_classes'], slot_hidden=CFG['slot_hidden'],
    dropout=CFG['dropout'], unfreeze_layers=0).to(DEVICE)
if N_GPUS > 1:
    model = nn.DataParallel(model)

head_params = [p for p in model.parameters() if p.requires_grad]
assert head_params and all(not p.requires_grad for p in (model.module if N_GPUS > 1 else model).dinov2.parameters())
optimizer = torch.optim.AdamW(head_params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
criterion = WeightedSoftBCELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=CFG['lr_t0'], T_mult=CFG['lr_t_mult'], eta_min=CFG['lr_eta_min'])
scaler = torch.amp.GradScaler('cuda') if CFG['mixed_precision'] else None
ema = EMAModel(model.module if N_GPUS > 1 else model, decay=CFG['ema_decay'])
print(f'Phase 4A model ready: frozen OrthoFoundation-L, trainable head={sum(p.numel() for p in head_params)/1e6:.2f}M')
