"""双模型集成 (ResNet50 + EfficientNetV2-S).

策略:
1. 独立训练两个 backbone
2. 冻结 backbone, 训练融合层 + Head
3. 联合微调
"""