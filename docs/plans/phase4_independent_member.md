# Phase 4 独立模型计划：OrthoFoundation → dense MIL

日期：2026-09-08

## 当前决策

Phase 4A 先做严格的低成本筛选：保持历史 `v5_labels.csv`、130 mm、9 张缓存切片和现有 SlotHead，只把 DINOv2-Small 换成冻结的 OrthoFoundation-L/DINOv3-L 特征，输入改为 256 px 以匹配 patch16。这样回答一个问题：膝关节 MRI 专用预训练本身，是否比当前自然图像 DINO 特征提供不同信号。

Phase 4A 不使用 Stage 3D 排序损失。Stage 3D 尚未产生结果；同时改变骨干和损失会无法归因。也不直接上 64–96 张切片，因为 dense coverage 与骨干变化应分开验证。

## Phase 4A 技术契约

- Backbone：timm `vit_large_patch16_dinov3`，加载官方 `OrthoFoudation-L.pth`。
- Backbone 全冻结，只训练每病变 slot attention 和分类头。
- 旧 v5 标签固定 SHA256：`c13adffaabf4f8e518abb038282bb1aa09baac7652a9165e030710c457d0be6a`。
- 130 mm、9 slices、3-slice RGB、6 slots；256 px；seed 42。
- batch 2、gradient accumulation 6，保持每次更新约 12 个检查，但单批显存更低。
- 关闭 jitter TTA，先降低大模型推理成本并避免引入第二个变量。
- checkpoint 加载按张量数量和参数量审计，必须 100% 匹配；不允许部分随机初始化继续训练。
- Gold 58 例仍只用于开发集验证，不能视为独立验证。

Kaggle 已有可直接挂载的公开 Dataset：`leogamertetudo/knee-b01-orthofoundation-l-public`，内含 1,213,056,638 字节的 `OrthoFoudation-L.pth`。本地副本放在被 Git 忽略的 `weights_upload/orthofoundation/`，SHA-256 为 `385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9`。如果自行上传，必须使用 Git-LFS 二进制，不能使用几百字节的指针。

该权重与 timm 的近似同名模型并非同一状态结构（官方 checkpoint 368 个张量，当前本地 timm 模型 318 个），因此 Phase 4A 改用官方 `facebookresearch/dinov3` 源码精确构建。还需把官方源码作为第二个离线 Dataset 挂载；本地已准备 `weights_upload/orthofoundation/dinov3-source-6876159.zip`（SHA-256 `6a82faabf21b6da79ce5ac0c1fb705666845bf5995d04a94b0a0d5f1ea8a2a7d`），固定提交 `6876159a11b4df116f30f667f8c9888617df0751`。真实权重已在该架构上 `strict=True` 加载成功：368/368 张量、303,227,920 参数。首次 Kaggle 运行仍会生成 `phase4a_weight_audit.json` 作为留档。

竞赛禁网不影响 `hashlib`：它属于 Python 标准库。Notebook 本身不会联网，也不会调用 `torch.hub`。用 `scripts/check_phase4a_assets.py --weights <OrthoFoudation-L.pth>` 在本地验证 Git-LFS 二进制、冻结标签哈希、官方源码和 checkpoint 的严格架构匹配；任何部分加载都会在训练前停止。

## 决策门槛

Phase 4A 的目的不是单模型达到 0.936，而是快速判断该预训练是否值得做 dense MIL。

继续 Phase 4B 至少满足以下一项：

1. Gold macro AUC 明显高于同输入的 v5s1，参考门槛 `>=0.90`；或
2. 总体接近 v5s1，但 ACL、MCL、半月板、OA 中至少三个类别改善；或
3. 与 0.936 父集成的同病例排名相关性明显低于已有 DINO 成员，同时 5% rank blend 不降低 macro AUC。

若低于 0.88，停止 OrthoFoundation 路线。若 0.88–0.90，先检查逐病例互补性，不直接花费算力做 96-slice 版本。

## Phase 4B：dense pathology-specific MIL

只有 Phase 4A 通过才实现：每个主要平面选 32 个真实切片，推理时扩展至 64–96 个观察位置；冻结骨干并预计算 FP16 特征，再训练小型 MIL 头。

MIL 的意思是一套检查里有许多切片，但只有检查级标签。模型为每种病变自己选择关键切片：ACL 与半月板可关注局部强证据，OA 与积液更重视大范围平均证据。平面内先聚合，再用同平面跨序列注意力融合，避免把冠状位与矢状位当作无区别的图片。

缓存特征后，绝大多数标签、池化和融合实验无需重复运行 300M 参数骨干。这是控制 Phase 4B 成本的核心。

## 文献路线回顾与取舍

### 已进入当前实现

**病变专用注意力 / Attention MIL**：Ilse 等人的 attention-based MIL 用可学习加权汇总一袋实例；当前 SlotHead 已按病变选择 MRI slot，Phase 4B 将进一步按病变选择切片。来源：https://proceedings.mlr.press/v80/ilse18a.html

**膝关节专用自监督预训练**：OrthoFoundation 在约 89 万张膝 MRI 切片上继续训练 DINOv3-L，并公开权重；论文仓库报告 ACL、MCL、软骨等迁移结果。Phase 4A 正在验证这种领域预训练。来源：https://github.com/ytrsk/OrthoFoundation

**同平面跨序列注意力**：CoPAS 先处理不同序列中解剖方向一致的信息，再完成多序列诊断；Phase 4B 采用它的融合思想，不照搬其私有数据训练设置。来源：https://www.nature.com/articles/s41467-024-51888-4 与 https://github.com/zqiuak/CoPAS

**AUC 对齐的可信病例排序**：Deep AUC Maximization、MIDAM 表明排序/AUC 目标适合类别不平衡和 MIL。Stage 3D 是低权重工程化试验，先验证规则选出的正例是否应排在负例前。来源：https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html 与 https://proceedings.mlr.press/v202/zhu23l.html

### 下一优先级创新

**报告证据引导图像注意力**：训练时从报告提取病变、解剖位置、肯定/否定与不确定性，用这些结构化证据约束模型关注的平面和切片；测试时只输入图像。它借鉴 GLoRIA 的全局—局部图文对齐和 MedKLIP 的实体/关系知识，但需要先修复逐类 uncertain 记录，不能使用目前“整份报告含不确定词就全部排除”的粗规则。来源：https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.html 与 https://arxiv.org/abs/2301.02228

**双尺度读取**：先用低分辨率完整序列找到疑似切片，再以高分辨率重读附近 5–9 张。它兼顾 Raptor 观察到的完整覆盖收益与半月板/骨折所需局部细节，预计比所有 96 张都用 384 px 更省算力。

**病理头与报告提及头分离**：一个头预测图像病变，一个辅助头预测放射科报告是否容易提到它。这样模型不会把“报告没写滑膜炎”直接学成“图像没有滑膜炎”。测试只使用病理头。

**外部专家定位预训练**：fastMRI+ 提供膝 MRI 的放射科病灶框和检查级病变标签，可用于预训练局部注意力；MRNet 可用于 ACL/半月板头初始化。这两项要下载并核对竞赛外部数据规则，工程成本高于 Phase 4A/4B。来源：https://github.com/microsoft/fastmri-plus 与 https://aimi.stanford.edu/datasets/mrnet-knee-mris

### 暂缓

**OrthoDiffusion / 通用 3D foundation model**：研究价值高，但公开权重与比赛输入适配不够直接，3D 重采样和显存成本也较大。当前优先使用已发布的膝关节 2D foundation 权重和轻量 MIL。

**继续扩大 DINO 或堆 seed**：社区与本项目都显示收益不足以支撑 0.936→0.94。它们可作为最终小幅稳定化手段，不承担独立成员发现任务。

## 当前文件

- `notebooks/kaggle_train_phase4a_orthofoundation_probe.ipynb`：可上传 Kaggle 的 Phase 4A Notebook。
- `notebooks/build_phase4a_orthofoundation_probe.py`：可重复生成 Notebook。
- `scripts/test_phase4a_builder.py`：静态契约与 checkpoint 前缀匹配单元测试。
- `notebooks/kaggle_train_v5_stage3d_rank.ipynb`：已修复为嵌入可信候选资产，不再需要额外 Kaggle Dataset。

Phase 4A 首次 Kaggle 运行首先验证真实 checkpoint 是否与 timm DINOv3-L 同构。加载审计通过后才开始 DICOM 缓存和训练；若失败，应下载官方 DINOv3 仓库代码作为精确架构适配，而不是降低覆盖阈值继续运行。
