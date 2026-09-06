# 阶段 3A：MRI 输入几何消融计划

## 目标

验证**物理裁剪大小**是否能为后续独立成员提供更好的 MRI 输入。首个实验只比较
`130mm` 与 `140mm`，不同时改变模型、slot、切片数、训练日程或融合方式。

这是一项控制变量实验：若同时改动多个因素，即使 gold 分数变化，也无法知道原因。

## 固定项

| 项目 | 固定设置 |
|---|---|
| 模型 | v5 DINOv2-small multi-view + SlotHead |
| slots | 6 个临床 slot |
| 切片 | 每 slot 9 张，3 张相邻切片组成一个 2.5D 输入 |
| 分辨率 | 288 px |
| 训练 | seed 42、30 epochs、EMA、WeightedSoftBCE、相同 TTA |
| 监督 | `pseudo_labels_deepseek_gpt56sol_fused.csv` |
| 验证 | 全部 58 个 gold；gold 永不参与训练 |

## 唯一变量

| 实验 ID | `crop_mm` | 目的 |
|---|---:|---|
| A0 | 130mm | 历史 v5 几何基准；仅作参照 |
| A1 | 140mm | 当前首个正式运行；增加周边解剖结构上下文 |

`crop_mm` 是按 DICOM PixelSpacing 裁出的真实世界长度，而非固定像素框。例如
140mm 在不同医院扫描的 0.3mm/px 与 0.5mm/px 图像上，仍表示同样大小的膝盖区域。

## 运行包与输入

1. 将本地融合标签上传为一个独立 Kaggle Dataset，slug 建议
   `rsna-knee-stage3a-labels`，根目录保留文件名
   `pseudo_labels_deepseek_gpt56sol_fused.csv`。
2. 运行 `python notebooks/build_v5_stage3a_140mm.py` 生成
   `notebooks/kaggle_train_v5_stage3a_140mm.ipynb`。
3. 在 Kaggle 新建/更新训练 notebook，挂载：竞赛数据、DINOv2 权重、上述标签数据集。
4. 使用 T4 x2 执行；下载 checkpoint、`gold_validation_predictions_s42.csv`、
   `gold_validation_auc_s42.csv` 和 `submission.csv` 至 `results/stage3a_140mm/`。

## 裁决

不以 gold 的微小增减单独决策。检查：

1. gold macro-AUC 与每类 AUC；
2. 特别是 ACL、半月板、Contusion、Fracture 是否出现一致改善；
3. 新模型和 0.936 的 gold/test 预测 rank 相关性；
4. 只有表现出不同排序，才做一次全局 rank-blend 扫描。

58 个 gold 病例较少；低于约 0.02 的 macro 差异只能视为探索信号，而不是确认结论。
若 A1 没有明确的类别级信号，则保持 0.936 基线并停止继续扩大 crop 的试验。

## 后续顺序

`A1` 有价值后：固定 140mm，单独改变切片覆盖/slot 选择；之后才用胜出输入训练轻量 3D 独立成员。任何新成员先做预测互补性检查，再进入 0.936 集成。
