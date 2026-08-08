"""数据增强 pipeline — 基于 albumentations.

训练时: 空间变换 + 像素级变换 + 遮挡, 所有随机参数可配置
验证时: 仅 resize

2D/2.5D 通用:
  - 2D (in_channels=1): 对 [H, W] 单通道应用增强
  - 2.5D (in_channels=3): 对 [H, W, 3] 三通道统一应用增强
    (3 个切片作为 3 个通道, 空间变换同步施加, 保持切片间对齐)
"""

from __future__ import annotations

import numpy as np
import albumentations as A


def build_transforms(
    aug_config: dict,
    image_size: int,
    is_train: bool = True,
) -> A.Compose:
    """从配置文件构建 albumentations 增强管道.

    Args:
        aug_config: 配置的 augmentation 段
        image_size: 输出尺寸 (正方形)
        is_train: True = 训练增强, False = 仅 resize

    Returns:
        albumentations.Compose 对象, 调用方式: result = transform(image=img)["image"]
    """
    if not is_train or aug_config.get("valid", {}).get("resize_only", True):
        # 验证模式: 只 resize
        return A.Compose([
            A.Resize(image_size, image_size),
        ])

    train_cfg = aug_config.get("train", aug_config)
    pipeline = [A.Resize(image_size, image_size)]

    # --- 空间变换 ---

    if train_cfg.get("horizontal_flip_probability", 0) > 0:
        pipeline.append(
            A.HorizontalFlip(p=train_cfg["horizontal_flip_probability"])
        )

    # ShiftScaleRotate 一次完成平移 + 缩放 + 旋转 (高效, 只插值一次)
    shift = train_cfg.get("shift_limit", 0)
    scale = train_cfg.get("scale_limit", 0)
    rotate = train_cfg.get("rotate_limit_degrees", 0)

    if shift > 0 or scale > 0 or rotate > 0:
        pipeline.append(
            A.ShiftScaleRotate(
                shift_limit=shift,
                scale_limit=scale,
                rotate_limit=rotate,
                border_mode=0,            # cv2.BORDER_CONSTANT → 黑边填充
                value=0,
                p=0.5,
            )
        )

    # --- 像素级变换 (模拟 MRI 扫描参数变化) ---

    if train_cfg.get("brightness_contrast_probability", 0) > 0:
        pipeline.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=train_cfg["brightness_contrast_probability"],
            )
        )

    if train_cfg.get("gamma_probability", 0) > 0:
        pipeline.append(
            A.RandomGamma(
                gamma_limit=(80, 120),    # gamma ∈ [0.8, 1.2]
                p=train_cfg["gamma_probability"],
            )
        )

    # --- 遮挡/擦除 (模拟伪影、局部信号丢失) ---

    if train_cfg.get("coarse_dropout_probability", 0) > 0:
        pipeline.append(
            A.CoarseDropout(
                max_holes=8,
                max_height=min(32, image_size // 4),
                max_width=min(32, image_size // 4),
                min_holes=2,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=train_cfg["coarse_dropout_probability"],
            )
        )

    return A.Compose(pipeline)


def apply_transform(
    image: np.ndarray,
    transform: A.Compose,
) -> np.ndarray:
    """对 [C, H, W] 图像应用 albumentations 变换.

    albumentations 要求 channels_last ([H, W] 或 [H, W, C]),
    而项目内部用 channels_first ([C, H, W]), 此函数做双向转换.

    Args:
        image: [C, H, W] numpy 数组
        transform: albumentations.Compose 管道

    Returns:
        变换后的 [C, H, W] numpy 数组
    """
    if image.ndim == 2:
        # 单通道 [H, W] → [1, H, W]
        result = transform(image=image)["image"]
        if result.ndim == 2:
            result = result[np.newaxis, ...]
        return result

    c = image.shape[0]
    if c == 1:
        # [1, H, W] → [H, W] → transform → [1, H, W]
        result = transform(image=image[0])["image"]
        if result.ndim == 2:
            result = result[np.newaxis, ...]
        return result
    else:
        # [C, H, W] → [H, W, C] → transform → [C, H, W]
        img_hwc = image.transpose(1, 2, 0)
        result = transform(image=img_hwc)["image"]
        return result.transpose(2, 0, 1)
