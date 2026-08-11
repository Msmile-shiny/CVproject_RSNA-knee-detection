# ============================================================
# Configuration — 纯推理（不训练）
# ============================================================

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
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

# ---- Diagnostic-specific TTA pooling ----
DIAG_POOL = {
    "Fracture": "max", "Contusion": "max",
    "Medial Meniscus": "max", "Lateral Meniscus": "max",
    "Baker's": "max",
    "ACL": "top2", "MCL": "top2",
}

CFG = {
    # --- Paths (修改为你自己的 Kaggle Dataset 路径) ---
    "comp_input":      "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    "dinov2_weights":  "/kaggle/input/datasets/easoncyy/rsna-dinov2-weights/dinov2_vits14.pth",
    "best_model":      "/kaggle/input/datasets/easoncyy/rsna-knee-v4-best-model/best_model.pt",
    "test_dicom_subdir": "test_series",
    "output_dir":      "/kaggle/working",

    # --- Data ---
    "image_size": 224,
    "crop_mm": 160.0,
    "cache_slices": 9,
    "group_size": 3,
    "center_pct": (0.2, 0.8),

    # --- Model (必须与训练时一致) ---
    "dinov2_variant": "vit_small_patch14_dinov2.lvd142m",
    "cls_dim": 384,
    "feature_dim": 1152,
    "slot_hidden": 256,
    "num_classes": 12,
    "unfreeze_layers": 6,
    "dropout": 0.2,

    # --- Inference ---
    "pix_threads": 4,
    "batch_size": 8,
}

# Device
N_GPUS = torch.cuda.device_count()
DEVICE = torch.device("cuda" if N_GPUS > 0 else "cpu")

if IS_MAIN:
    print(f"GPUs: {N_GPUS} | Device: {DEVICE}")
    print("--- Inference v4: Test Set Only ---")
    for k, v in CFG.items():
        print(f"  {k}: {v}")
