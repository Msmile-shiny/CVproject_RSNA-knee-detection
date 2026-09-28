# Stage 5A：密集切片与高分辨率复看

日期：2026-09-10。状态：代码与本地行为测试完成；尚未执行 Kaggle 全量训练、父模型 Gold 推理或 Public 评分。

2026-09-14 更新：用户已完成粗特征和三路 head 训练，精特征进行到至少3840/4410例时被旧代码420分钟保护主动中断。已修复缓存/模型续跑、阶段导出与正常暂停。诊断与恢复步骤见 [超时恢复报告](../experiments/stage5a_timeout_recovery.md)。目前仍没有本次 Gold AUC 或 Public 成绩。

## 决策与证据边界

用户报告 4A2 Public 0.808；暂停 OrthoFoundation 的竞赛投入。这是预算决策，不代表证明领域预训练无效，也不能单凭单模型分数断言它与所有父模型融合必然下降。此前 0.88 阈值针对 Gold，不能直接套用 Public。4A2 完整日志尚未见于本地 results。

历史 v5 三 seed 集成 Public 0.886，不是 v5s1 单模型的已知 Public。58 Gold 参与过标签校准与历史 checkpoint 选择，本项目的 Gold 数值属于开发集表现；重新分割预测或 bootstrap 均不能消除已有的信息泄漏。

父模型由用户指定：https://www.kaggle.com/code/easoncyy/rsna-knee-abnormality-detectionv1?scriptVersionId=347718768 。本次 API 下载确认 currentVersionNumber=1，API 不返回 scriptVersionId，二者映射未独立验证。保留源码 SHA 在生成的导出 Notebook metadata 中。该模型已经有 Raptor、DINO、多分支融合以及 public0033 半月板 overlay；不能把我们的 MIL 说成社区没有做过的创新。

## 本次检索

| 来源 | 可核实信息 | 对实验的影响 |
|---|---|---|
| [Raptor CoAtNet 页面](https://www.kaggle.com/code/hdhsjdjd/rsna-knee-raptor-coatnet/input) | 搜索摘录显示 V1 Public 0.933；这是该 fork，不是所有 Raptor 版本最高分 | 常规 CNN 分支已存在，不以换骨干为新贡献 |
| [四路集成](https://www.kaggle.com/code/nishantkharga/bend-the-knee-to-dinov3-ensembled-nk) | 页面重定向 Full 4-arm ensemble v55，版本 ID 346827172；当前抓取无可读分数 | 不据标题声称已找到可复制的 0.94 |
| [H&S](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes/notebook) | 动态网页未返回可读源码；用户自身 V1 已通过认证 API 下载 | 以实际父模型版本和运行资产为准 |
| [Attention MIL，ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html) | 学习一组实例的聚合权重 | 实现每病变 gated attention，并设均值对照 |
| [CoPAS，2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/) | 检索摘要：1,748 名患者、五中心、多序列与同平面注意力 | 保留 slot 身份；当前实现不是完整 CoPAS 跨序列架构复现 |
| [粗到细膝 MRI 多任务研究，2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/) | 13,419 患者、14,962 次 MRI；注意力引导的粗到细分类。主要病变 AUC 内部 0.898、两个外部集 0.852/0.812 | 支持局部复看的研究动机，也说明外部泛化可能大幅下降；不能把论文 AUC 对应 Kaggle 分数 |
| [fastMRI+](https://arxiv.org/abs/2109.03812) | 16,154 个膝病理专家框；需配套原始影像并对齐类别 | 留作将来的定位监督；本阶段不下载外部影像，不增加数据依赖 |

截至本次检索，没有核实一套可直接运行且必达 0.94 的公开方案。收益是待验证假设。

## 实现

复用 v5s1 已训练的 DINOv2-small（含 EMA），而非再次冻结未适配任务的通用骨干。它仍与 v5 同源，独立性只能来自切片覆盖、聚合和尺度，必须测量，不能预设。

- 6 个历史 slot、130 mm 物理裁剪。每个 slot 至多 32 个位置覆盖完整有效序列；每个位置取真实相邻三切片组成输入。短序列不复制填充实例，padding 有独立 mask。
- 粗看 168 px、复看 280 px，均为 patch14 的整数倍。加载权重严格匹配；位置编码做显式插值。
- 逐病例读取 DICOM 与编码，FP16 特征缓存到磁盘并按需载入 RAM。约 4,407 病例的粗特征上限约 1.82 GiB，额外精看特征约 0.23 GiB，不缓存整库高分辨率像素。
- 每病变 gated attention 汇总切片；slot、相对切片位置和分辨率标记可学习。注意力不等于经医生验证的病灶位置。
- 每类选两个高注意力位置，取并集，上限 24 个三切片窗口/病例，以 280 px 重新编码。训练与测试均由图像注意力选择，测试不使用报告或标签。
- 历史 v5 标签 SHA 固定。Gold 不进入本轮梯度训练，不用于选择 epoch。训练轮数预先固定20。
- 四路输出：mean（20轮）、coarse attention（20轮）、coarse_continue（再20轮）、fine（从 coarse 继续20轮）。fine 与 coarse_continue 对比，以排除多训练20轮的影响。

当前实现检验“密集特征与选择后高分辨率信息是否有用”。尚无均匀位置高分辨率对照，所以即使 fine 上涨，也不能单独宣称注意力选片优于均匀选片。若有初步收益，下一小实验再补该对照。

## Kaggle 运行顺序

1. 导入 `notebooks/kaggle_train_stage5a_dense_mil.ipynb`，使用 T4 x2、Internet Off。
2. 挂载比赛数据、历史 `v5_labels.csv`、**v5s1 的 `best_model_s42.pt`**。它是已训练的模型，不是 `dinov2_vits14.pth`，也不是 Stage 3D 的同名文件。当前本地 results/v5s1 未见 checkpoint，应使用原 v5s1 Kaggle 输出。Notebook 第一处配置中的 `S5['v5_checkpoint']` 建议填完整路径；`S5['labels']` 同理，留空时要求全 input 只找到一个同名文件。
3. 新实验首先 `smoke_studies=8, epochs=1` 完整跑通；该结果不可评价成绩。随后重启 session，设 `smoke_studies=0, epochs=20` 全量运行。本次全量续跑保持0/20，无需重新跑 smoke。使用300分钟粗特征预算与 session_minutes=480、reserve_minutes=20 的整体预算，达到暂停线时保存进度并正常返回，manifest 标记 completed=false，不静默减少测试病例。resume_input 指向旧输出中含 stage5a_manifest.json 的目录；校验通过才复用旧特征/已完成 head。
4. 另开 Notebook 导入 `notebooks/kaggle_stage5a_parent936_gold.ipynb`，挂载与用户0.936原Notebook完全相同的资产，导出 `parent936_gold.csv` 和分支快照。它构造仅包含 Gold 的虚拟 test 目录，原模型/融合源代码仅替换比赛根路径。所有原作者署名保留。该 Notebook 不能提交竞赛。
5. 保存输出到 `results/v5s5a` 与 `results/parent936_gold`。

Stage 5A 不产生默认 `submission.csv`，会产出 `submission_stage5a_mean/coarse/coarse_continue/fine.csv`、相应 Gold 预测、AUC表、训练历史、head权重、特征、选片位置、解码错误及 manifest。训练与推理写在一个可离线执行的 Notebook 中；目前是筛选实验，获准晋级后再制作仅推理/父模型注入版本。

## 决策

先核查解码错误、输入覆盖、标签/权重哈希及 smoke 标记。对照 fine 与 coarse_continue，以及 coarse 与 mean。父模型预测以 UID 严格一一对齐，固定仅检查全类别统一 alpha=0.02/0.05，不在58例上搜索逐类最优权重。

运行示例（项目根目录）：

```powershell
python scripts/compare_stage5a.py --parent results/parent936_gold/parent936_gold.csv --member results/v5s5a/gold_stage5a_fine.csv --truth results/v5s5a/stage5a_gold_truth.csv --out results/v5s5a/parent_fine_comparison.json
```

只有父模型融合改善且不是一两个病例驱动时，才考虑一次小权重 Public 验证。+0.003 是工程参考，不是统计定理；Gold<0.89也不能单凭此否定互补性。父模型各权重是否训练过 Gold 尚未核验，比较报告会明确记载；bootstrap 仅反映此开发集的抽样波动，不能证明无泄漏或保证 Private 提分。若四路均无融合价值，停止该分支，保留已验证0.936。
