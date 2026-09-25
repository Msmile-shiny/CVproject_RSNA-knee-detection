# ============================================================
# Configuration — v3: Multi-View 6-Slot + SlotHead
# ============================================================

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]

# Soft-label column names (reused from v2)
PROB_COLS   = [f'prob_{c}' for c in TARGET_COLUMNS]
WEIGHT_COLS = [f'weight_{c}' for c in TARGET_COLUMNS]
MASK_COLS   = [f'mask_{c}' for c in TARGET_COLUMNS]

# ---- 6 Clinical Slots (matching reference code) ----
# (name, plane, fluid, fatsat)
#   fluid=True → T2/PD (Fluid_Sensitive=1 in metadata)
#   fluid=False → T1-like structural (Fluid_Sensitive=0)
#   fatsat=True → Fat Suppression enabled
SLOTS = [
    ("SAG_FLUID_FS",   "Sagittal", True,  True),
    ("COR_FLUID_FS",   "Coronal",  True,  True),
    ("AX_FLUID_FS",    "Axial",    True,  True),
    ("SAG_FLUID_NOFS", "Sagittal", True,  False),
    ("COR_T1",         "Coronal",  False, False),
    ("SAG_T1",         "Sagittal", False, False),
]
N_SLOT = len(SLOTS)

# ---- Anatomical Priors for SlotHead ----
# Clinical knowledge: which planes each pathology is best seen on
# Slot indices: 0=SAG_FLUID_FS, 1=COR_FLUID_FS, 2=AX_FLUID_FS,
#               3=SAG_FLUID_NOFS, 4=COR_T1, 5=SAG_T1
SLOT_PRIORS = {
    "ACL":               (0, 3, 5),        # Sagittal views
    "MCL":               (1, 4),            # Coronal views
    "Medial Meniscus":   (0, 1, 3, 4),      # Sag + Cor
    "Lateral Meniscus":  (0, 1, 3, 4),
    "Medial OA":         (1, 4, 5),         # Cor + Sag T1
    "Lateral OA":        (1, 4, 5),
    "PF OA":             (0, 2, 5),         # Sag + Axial
    "Effusion":          (0, 2),            # Sag FS + Ax FS
    "Synovitis":         (0, 2),
    "Baker's":           (0,),              # Sag FS
    "Contusion":         (0, 1, 2),         # All FS planes
    "Fracture":          (0, 1, 2, 4, 5),   # All except Sag noFS
}

CFG = {
    # --- Paths (Kaggle) ---
    'comp_input':   '/kaggle/input/competitions/rsna-knee-abnormality-detection',
    'pseudo_input': '/kaggle/input/datasets/easoncyy/rsna-knee-soft-labels',
    'dicom_subdir': 'train_series',
    'output_dir':   '/kaggle/working',

    # --- Data ---
    'image_size': 224,          # DINOv2 native; 16×16 patches → 256 tokens (9.4× faster than 392)
    'cache_slices': 9,          # slices cached per slot
    'group_size': 3,            # 3 adjacent slices → RGB-like channels
    'center_pct': (0.2, 0.8),   # sample from central 60% of slice stack

    # --- Training subset control ---
    # Set to None to use ALL studies. Set to N to randomly sample N studies for faster iteration.
    'train_studies_limit': 500,   # smaller = faster; increase for final run

    # --- Model ---
    'dinov2_variant': 'vit_small_patch14_dinov2.lvd142m',
    'cls_dim': 384,             # DINOv2 ViT-S hidden dim
    'feature_dim': 384 * 3,     # CLS + mean(patches) + focal_topk
    'slot_hidden': 256,         # SlotHead hidden dim
    'num_classes': 12,

    # --- Unfreeze strategy (matching reference) ---
    'unfreeze_layers': 6,       # last 6 of 12 DINOv2 blocks + final norm

    # --- Training ---
    'batch_size': 6,            # 6 studies × 6 slots = 36 DINOv2 forwards/step
    'grad_accum_steps': 2,      # effective batch = 6 × 2GPUs × 2 = 24
    'epochs': 50,
    'lr': 2e-4,
    'backbone_lr': 1e-5,
    'weight_decay': 1e-4,
    'lr_t0': 15,
    'lr_t_mult': 2,
    'lr_eta_min': 1e-6,
    'dropout': 0.2,
    'grad_clip': 1.0,
    'early_stop_patience': 10,
    'mixed_precision': True,
    'num_workers': 2,
}

# Device setup
N_GPUS = torch.cuda.device_count()
DEVICE = torch.device('cuda' if N_GPUS > 0 else 'cpu')
IS_MAIN = True

if IS_MAIN:
    print(f'GPUs: {N_GPUS} | Device: {DEVICE}')
    print(f'--- v3: Multi-View 6-Slot + SlotHead ---')
    for k, v in CFG.items():
        print(f'  {k}: {v}')
