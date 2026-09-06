# Phase 3B 前研究复核（2026-09-06）

## 先纠正失败归因

本次仅研究和审计，没有启动训练。下述决策更新此前实验报告中的技术路线。

`scripts/build_v5_labels.py` 表明历史 v5 使用报告提取结果和图像 OOF 预测融合；`scripts/build_gpt56sol_fusion_labels.py` 默认从 `data/pseudo_labels_calibrated.csv` 构建新标签，并未以 `data/processed/v5_labels.csv` 为底稿。

对齐三个文件共同的 4,349 个 UID 后，发现：

- 相比历史 v5，新文件全部 12 类的概率都有变化；不仅是六类 GPT 替换。
- 52,188 / 52,188 个 weight 单元格不同，1,584 个 mask 单元格不同。
- 新文件每类概率只有 3–6 个不同取值，失去历史图像教师提供的逐病例连续信息。
- 概率平均绝对变化：Effusion 0.2686、Lateral OA 0.2132、Medial OA 0.1635。

因此 0.818 不能归因为 140 mm，也不能证明 GPT 标签差。更准确地说，整个监督管线和几何同时变化。低相关性可能来自错误；尚未计算与 0.936 成员的融合，因此不能宣称它绝无互补性，但当前证据不足以投入融合。

另外，两种标签生成脚本都利用 58 条金标拟合校准器。金标虽未直接作为图像训练样本，仍间接影响监督；相关 AUC 应称开发集指标，不是完全独立验证。今后冻结一部分金标为审计集，或在每个外层验证划分中重新拟合校准器、训练模型，不能只在预测文件上重新分折就声称消除泄漏。

## 社区复核

- [报告未提及问题](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)：作者报告许多标签是未提及，不能统一按阴性处理。其标签分数是作者报告，不是本项目独立复核。
- [瓶颈讨论](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735826)：报告用词、LLM 的文本判断和影像标注有差异。“读懂报告的把握”不等于“影像确有病变的概率”。
- [DINO small/base 对照](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738096)：公开摘录支持 base 早期收敛更快；不能据此宣称最终成绩一定更高。
- [H&S](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes/notebook)：本次页面摘录显示 best 0.936 V38，含私有输入。公共 notebook 不等于所有模型可复现。
- [四路集成](https://www.kaggle.com/code/nishantkharga/bend-the-knee-to-dinov3-ensembled-nk)：本次页面跳转到 Full 4-arm ensemble v55；检索快照显示 best 0.935 V13，与此前 0.936 描述不一致。锁定已实际复现的版本和资产，不从标题或不同时间快照推断最新成绩。

公开检索不能排除私有方案；没有找到某方法，不等于全社区没有实现。现有 DINO、CoAtNet、RadImageNet 和类别注意力已经覆盖常规多成员路线，单纯再训练 ResNet50 不应被称作创新。

## 学术依据与可迁移部分

| 文献 | 可借鉴内容 | 限制 |
|---|---|---|
| [Attention-based MIL, ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html) | 一例多张图，学习哪些图应贡献更多 | 注意力不是已验证病灶定位 |
| [GLoRIA, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.html) | 把报告词语与局部图像特征对齐 | 胸片证据不能直接当作膝 MRI 提分证据 |
| [MedKLIP, 2023](https://arxiv.org/abs/2301.02228) | 从报告抽取结构化实体，再对齐图像区域 | 跨语言、否定和部位抽取仍需检查 |
| [Deep AUC Maximization, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html) | 直接优化排序相关目标 | 伪标签错误会形成错误配对 |
| [MIDAM, ICML 2023](https://proceedings.mlr.press/v202/zhu23l.html) | 多实例 AUC 优化，随机抽取部分实例降低显存需求 | 简单随机采样并不等于完整复现论文估计器 |
| [3D Neuroimage MIL Benchmark, CHIL 2026](https://proceedings.mlr.press/v333/harvey26a.html) | 比较冻结二维编码器+聚合与三维网络的实验设计 | 研究解剖部位和任务不同，不能保证迁移收益 |

## 值得尝试的组合创新（尚无本比赛提分证据）

### A：报告证据指导的局部精读（中高成本，优先的研究主线）

保留报告里的部位、状态、程度和原句，例如“内侧半月板后角、撕裂、明确”，而非压缩成十二个概率。训练时让相应病变的查询向量从 MRI 切片/区域中选择证据；推理只输入 MRI，查询向量从训练中学得，不依赖测试报告。

先用冻结编码器和小聚合头验证“结构化证据辅助监督”是否有益；通过后再加入低分辨率全局扫描和少量高分辨率局部复看。训练和推理都用同样的图像选择器，报告只提供训练监督，避免训练靠报告定位、测试无报告的落差。用随机区域与均匀取片对照检查选择器价值。

创新候选是将报告部位证据、跨序列切片聚合和固定算力下局部复看组合用于本任务；各组件本身已有文献基础。尚未在本次检索到的可复现高分方案中确认完整实现。

### B：明确报告证据 + OOF 教师的可信病例排序（低中成本，优先小实验）

在保留 v5 图像教师信号的基础上，先保持 BCE 损失，再附加小权重配对损失：明确阳性病例应排在明确阴性病例前。仅用训练病例、可审计的报告证据和未训练过该病例的教师生成配对，跳过未提及和冲突病例。按类别采样，防止常见类别主导。

目标是让监督保留细粒度顺序，而非把所有阳性都压成同一个数。需要与完全相同监督的 BCE 对照；论文并不保证该噪声标签改造必然提升。

### C：影像证据与报告记录习惯分开学习（高风险探索）

设一个头学习病变，一个训练辅助头学习“报告是否提到该病变”。明确否定、明确阳性、不确定和未提及分别建模。辅助任务有机会防止模型把语言/医院的记录习惯误当病变，但也可能增加捷径；需要按协议、语言等进行分组诊断，且分组不能被当作真实医院身份。当前样本少，暂列第三优先级。

## Phase 3B 修订顺序

1. 锁定历史 `v5_labels.csv`、模型参数、划分和文件哈希。核验 OOF 训练排除关系；文件名 OOF 不足以证明无泄漏。
2. 优先运行 140 mm + 原 v5 标签，与历史 130 mm + 原 v5 标签对比，隔离几何；与已完成的 140 mm + 新标签对比，检查整体监督变化。
3. 若需独立估计标签与几何的交互，再补 130 mm + 新标签，构成完整 2×2。不要同时改模型或切片。
4. 在原 v5 标签上构建真正局部的 GPT 修正候选，保留图像教师；未授权列必须逐单元格相等。置信度政策另立实验，禁止默认沿用错误来源的可信度。
5. 恢复可靠训练后，先试 B 的小规模排序损失；再开展 A 的部位证据模型。所有新成员需取得同一开发集上的 0.936 父方案预测，预先固定融合权重并检查配对差值，避免反复逐类调权。

停止条件：没有可复核的监督血缘就不启动新训练；冻结特征的小实验没有稳定改善就不扩展到昂贵全模型。0.886 是历史三 seed 集成的 public 分数，不应被当作 seed 42 单模型的合格门槛。58 例的统计不确定性应通过配对重采样等方式报告，而不是使用固定 0.02 的通用显著性阈值。
