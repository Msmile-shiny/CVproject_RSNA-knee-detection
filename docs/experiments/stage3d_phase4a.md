# Stage 3D 与 Phase 4A 实验报告（2026-09-09）

## 决策摘要

- Stage 3D：Gold 从 v5s1 的 0.893262 升至 0.897097，但 Public 只有 0.880。可信排序可以作为弱辅助监督保留，不是冲榜模型，也不继续堆 seed 或扩大排序权重。
- Phase 4A：官方 OrthoFoundation 权重严格加载成功，但冻结骨干的 Gold 只有 0.790551，低于继续 dense MIL 的 0.88 门槛；其预测混入 v5s1/3D 都会降低 Gold，禁止提交融合。
- 下一步只做 Phase 4A2 输入对齐对照：把窗口中心的单张 MRI 切片复制到 RGB 三通道，以匹配官方预训练输入。若仍低于 0.88，停止 OrthoFoundation 路线。

## Stage 3D：可信病例排序

配置与历史 v5s1 对齐：130 mm、288 px、历史 `v5_labels.csv`、seed 42；新增权重 0.05 的可信正负病例排序损失。最终完整推理 Gold macro AUC 为 0.897097，训练路径第 30 轮为 0.898545。与 v5s1 的逐类别平均 Spearman 排名相关性为 0.970336，说明它仍是高度相似的同族模型。

相对 v5s1，主要上升为 Synovitis +0.0382、Lateral Meniscus +0.0335、PF OA +0.0167；主要下降为 MCL -0.0317、Baker's -0.0163、Fracture -0.0125。1,000 次病例配对 bootstrap 的差值 95% 区间约为 [-0.0044, 0.0130]，跨过 0，不能把小幅 Gold 增益视为稳定结论。

排序损失从 0.1021 降到 0.0026；只有 27.9% 的 batch 出现有效正负配对。Lateral OA 和 PF OA 在 30 轮中分别只有 136、141 对，说明随机 batch 对稀有可信病例利用率低。Public 0.880 最终否定了“仅靠这项监督能提升排行榜”的假设。

## Phase 4A：冻结 OrthoFoundation

权重审计通过：SHA-256 `385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9`，官方 DINOv3-L 架构 368/368 张量严格匹配，参数覆盖率 100%。因此低分不是错权重或部分随机初始化造成的。

20 轮 Gold 从 0.665412 单调升至 0.790551，验证损失从 0.64157 降至 0.51287，表现为明显欠拟合而非训练崩溃。相对 v5s1，只有 Lateral OA +0.0445、Synovitis +0.0060、Lateral Meniscus +0.0025；MCL -0.3197、Fracture -0.1847、ACL -0.1801、Contusion -0.1673。平均排序相关性为 0.7573，虽然足够独立，但准确率不足。给 v5s1 或 Stage 3D 混入 5% 的 Phase 4A 排名都会降低 Gold。

官方预训练代码对每张二维图像调用 `Image.open(...).convert("RGB")`，随后按 ImageNet mean/std 归一化；也就是说，灰度 MRI 是单切片复制到三通道。当前 4A 使用三张相邻切片作为 RGB，产生预训练中不存在的伪颜色分布。这使 4A 不能作为对 OrthoFoundation 本身的最终否定。官方仓库同时说明其下游结果来自完整微调，而不是我们本次的冻结探针。来源：[OrthoFoundation 数据读取](https://github.com/ytrsk/OrthoFoundation/blob/main/datasets/knee_multidata.py)、[数据增强](https://github.com/ytrsk/OrthoFoundation/blob/main/datasets/augmentation.py)、[项目说明](https://github.com/ytrsk/OrthoFoundation)。

## Phase 4A2：唯一允许的补充验证

保持 4A 的标签、病例、slot、裁剪、分辨率、优化器、冻结骨干、epoch 和 seed 不变，只把每个三切片窗口的中心切片复制三次。这样只回答“输入分布不一致是否造成 4A 失败”。

停止条件：最终 Gold < 0.88，或对 v5s1 做 5% rank blend 仍下降，则停止 OrthoFoundation，不开展 64–96 slice MIL，也不为它提交 Public。通过条件：Gold >= 0.88 且至少在三个结构类上提供互补提升；届时才考虑解冻最后两个 block 的小学习率验证。
