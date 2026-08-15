# ============================================================
# 推理专用配置 — 与 v5 训练配置的数据侧参数必须逐项一致
# (cell 08 会拿每个 checkpoint 内的 config 与这里交叉核对, 不一致立即报错)
# ============================================================

TARGET_COLUMNS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]
N_CLASSES = len(TARGET_COLUMNS)

# ---- 6 Clinical Slots ----
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

# ---- Diagnostic-specific TTA pooling (与训练一致) ----
DIAG_POOL = {
    "Fracture": "max", "Contusion": "max",
    "Medial Meniscus": "max", "Lateral Meniscus": "max",
    "Baker's": "max",
    "ACL": "top2", "MCL": "top2",
    "Synovitis": "original_mean",
}

# ---- Jitter TTA 增广参数 (与训练一致) ----
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1
AUG_SEED = 42

CFG = {
    # --- Paths ---
    'comp_input': '/kaggle/input/competitions/rsna-knee-abnormality-detection',
    # ★ 你上传 3 个 best_model_s*.pt 为 Kaggle Dataset 后的挂载路径
    #   若 slug 不同也没关系: cell 08 会自动扫描 /kaggle/input 全部数据集找 best_model_s*.pt
    'ckpt_input': '/kaggle/input/datasets/easoncyy/v5-seed-checkpoints',
    'dicom_subdir': 'train_series',
    'output_dir': '/kaggle/working',

    # --- Data (★ 必须与 v5 训练一致, 勿改) ---
    'image_size': 288,             # 288px@130mm = 0.451mm/px
    'crop_mm': 130.0,
    'cache_slices': 9,
    'group_size': 3,
    'center_pct': (0.2, 0.8),

    # --- Model (★ 必须与 v5 训练一致, 勿改) ---
    'dinov2_variant': 'vit_small_patch14_dinov2.lvd142m',
    'cls_dim': 384,
    'feature_dim': 1152,
    'slot_hidden': 256,
    'num_classes': 12,
    'unfreeze_layers': 6,

    # --- Inference ---
    'tta_jitter': True,            # jitter TTA (训练时开启, 推理必须一致)
    'hdr_threads': 8,
    'pix_threads': 4,
}

N_WINDOWS = CFG['cache_slices'] - CFG['group_size'] + 1  # 7

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
    print(f'--- v5 推理专用 (seed 集成 rank-mean 融合) ---')
    print(f'  288px/130mm | {N_WINDOWS} 窗口 TTA + jitter | 诊断池化')
