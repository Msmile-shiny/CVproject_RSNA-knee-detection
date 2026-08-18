# ============================================================
# 推理专用配置 — 两套数据侧参数 (与各训练会话逐项一致)
#   CFG_V5:  v5 seed 成员 (DINOv2-small @288px/130mm, cache 9 片)
#   CFG_RAD: v6a rad 成员 (RadImageNet R50 @224px/130mm, cache 7 片)
# (cell 08 会拿每个 checkpoint 内的 config 与对应 CFG 交叉核对, 不一致立即报错)
# ============================================================

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]
N_CLASSES = len(TARGET_COLUMNS)

# ---- 6 Clinical Slots (v5 与 v6a 完全一致) ----
SLOTS = [
    ("SAG_FLUID_FS",   "Sagittal", True,  True),
    ("COR_FLUID_FS",   "Coronal",  True,  True),
    ("AX_FLUID_FS",    "Axial",    True,  True),
    ("SAG_FLUID_NOFS", "Sagittal", True,  False),
    ("COR_T1",         "Coronal",  False, False),
    ("SAG_T1",         "Sagittal", False, False),
]
N_SLOT = len(SLOTS)

# ---- Anatomical Priors ----
SLOT_PRIORS = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5), "Effusion": (0, 2), "Synovitis": (0, 2),
    "Baker's": (0,), "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}

# ---- Diagnostic-specific TTA pooling (两管线一致) ----
DIAG_POOL = {
    "Fracture": "max", "Contusion": "max",
    "Medial Meniscus": "max", "Lateral Meniscus": "max",
    "Baker's": "max",
    "ACL": "top2", "MCL": "top2",
    "Synovitis": "original_mean",
}

# ---- Jitter TTA 增广参数 (两管线一致) ----
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1
AUG_SEED = 42

# 共享路径
_PATHS = {
    'comp_input': '/kaggle/input/competitions/rsna-knee-abnormality-detection',
    # ★ 权重数据集挂载路径; slug 不同没关系 — cell 08 自动扫描 /kaggle/input
    #   全部数据集找 best_model_s*.pt / best_model_rad.pt
    'ckpt_input': '/kaggle/input/datasets/easoncyy/v5-seed-checkpoints',
    'dicom_subdir': 'train_series',
    'output_dir': '/kaggle/working',
    'hdr_threads': 8,
    'pix_threads': 4,
}

# ---- v5 成员配置 (★ 与 v5 训练一致, 勿改) ----
CFG_V5 = {
    **_PATHS,
    'image_size': 288,             # 288px@130mm = 0.451mm/px
    'crop_mm': 130.0,
    'cache_slices': 9,
    'group_size': 3,
    'center_pct': (0.2, 0.8),
    'dinov2_variant': 'vit_small_patch14_dinov2.lvd142m',
    'cls_dim': 384,
    'feature_dim': 1152,
    'slot_hidden': 256,
    'num_classes': 12,
    'unfreeze_layers': 6,
    'tta_jitter': True,
}

# ---- rad 成员配置 (★ 与 v6a 训练一致, 勿改) ----
CFG_RAD = {
    **_PATHS,
    'image_size': 224,             # 224px@130mm = 0.580mm/px (分辨率多样性成员)
    'crop_mm': 130.0,
    'cache_slices': 7,             # v6a 内存防御: 9→7
    'group_size': 3,
    'center_pct': (0.2, 0.8),
    'feature_dim': 2048,           # ResNet50 GAP 特征
    'slot_hidden': 256,
    'num_classes': 12,
    'unfreeze_layers': 0,          # 编码器全冻结
    'tta_jitter': True,
}

N_WINDOWS_V5 = CFG_V5['cache_slices'] - CFG_V5['group_size'] + 1    # 7
N_WINDOWS_RAD = CFG_RAD['cache_slices'] - CFG_RAD['group_size'] + 1  # 5

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

N_GPUS = torch.cuda.device_count()
DEVICE = torch.device('cuda' if N_GPUS > 0 else 'cpu')

if IS_MAIN:
    print(f'GPUs: {N_GPUS} | Device: {DEVICE}')
    print(f'--- lateral_swap 推理专用 (不训练) ---')
    print(f'  v5 成员 : 288px/130mm, {N_WINDOWS_V5} 窗口 TTA + jitter')
    print(f'  rad 成员: 224px/130mm, {N_WINDOWS_RAD} 窗口 TTA + jitter')
    print(f'  融合    : base = 3 seed rank-mean; Lateral Meniscus/Lateral OA 换 rad_rank')
