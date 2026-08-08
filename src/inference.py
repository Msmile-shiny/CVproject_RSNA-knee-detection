"""2D/2.5D 推理管线.

- 加载 fold checkpoints
- 对每个 test study 的所有切片/triplet 推理
- Top-K 聚合 (2D: 切片级 top-K, 2.5D: triplet 级 top-K)
- Fold ensemble (mean logits)
- 生成 Kaggle submission.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import KneeSliceDataset
from src.models.classifier import KneeClassifier2D
from src.evaluate import aggregate_to_study


def predict(
    model: KneeClassifier2D,
    loader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """对所有切片推理, 返回 (logits, study_uids).

    Returns:
        logits: [N_slices, 12]
        study_uids: [N_slices]
    """
    model.eval()
    all_logits = []
    all_study_uids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images).cpu().numpy()
            all_logits.append(logits)
            all_study_uids.extend(batch["study_uid"])

    return np.concatenate(all_logits), np.array(all_study_uids)


def generate_submission(
    test_metadata_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    checkpoint_paths: list[Path],
    config: dict,
    output_path: str = "submission.csv",
) -> None:
    """生成 Kaggle 提交文件.

    Args:
        test_metadata_df: 测试集切片元数据
        labels_df: 测试集标签 (用于关联 StudyInstanceUID, label 列可为空)
        checkpoint_paths: 各 fold 的 .pt checkpoint 路径列表
        config: 模型/推理配置
        output_path: 输出 CSV 路径
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_study_logits = []

    for ckpt_path in checkpoint_paths:
        model = KneeClassifier2D(
            arch=config["model"]["arch"],
            in_channels=config["model"]["in_channels"],
            num_classes=config["model"]["num_classes"],
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        ds = KneeSliceDataset(
            test_metadata_df, labels_df,
            npy_root=config["paths"]["npy_root"],
            image_size=config["data"]["image_size"],
            in_channels=config["data"].get("in_channels", 3),
            is_train=False,
        )
        loader = DataLoader(ds, batch_size=config["train"]["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True)

        slice_logits, study_uids = predict(model, loader, device)
        study_logits, study_ids = aggregate_to_study(
            slice_logits, study_uids, config["inference"]["topk_fraction"]
        )
        all_study_logits.append(study_logits)

    # Fold ensemble: mean logits
    ensemble_logits = np.mean(all_study_logits, axis=0)               # [N_studies, 12]
    probs = 1.0 / (1.0 + np.exp(-ensemble_logits))                   # sigmoid

    # 写入 CSV
    sub = pd.DataFrame(probs, columns=config["data"]["target_columns"])
    sub.insert(0, "StudyInstanceUID", study_ids)
    sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}  ({len(sub)} studies)")
