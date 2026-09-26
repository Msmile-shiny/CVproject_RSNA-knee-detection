# 社区方案复查：2026-09-26

方式：网络检索、Kaggle API按dateRun列最新15条，再下载3份公开Notebook源码/挂载清单核查。网页正文部分不可获取，技术判断来自实际下载源码，不把搜索摘要/标题视为成绩证明。下载到工作区research-20260926，未执行不可信下载代码。

## 1. DINOsaur V5今日版本：修复、采样与分支权重

[公开Notebook](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5)，API lastRunTime 2026-09-26 08:11:20（API原始时间）。下载文件SHA256 `670bf2ca949ab804188c5f786882ed1c7bea9e477d8e165b5ec32d88a7e519f6`。

今日源码已不能只用过去“ConvNeXt独立分支”的描述概括。可见新增/现有逻辑包括：A5特定条件下物理位置排序修复、Global96容量分配、3个主CoAt成员均值加Repairv1残差、残差10%/15%/25%候选与部分逐类权重。源码明确部分输入/新分支分数未经验证。我们没有核实该下载版本的新Public成绩，不能称它保证0.945。

对项目的启示：输入排序和覆盖值得审计；权重变体只是候选，不能照搬多档搜索到我们58例Gold上。工程修复是否有价值必须先确认本项目也存在相同问题；我们已有基于IOP/IPP的排序，不需要仅因对方修复就重复改变。

## 2. Master 4-Arm：公开主干加私有/可选独立分支

[公开Notebook](https://www.kaggle.com/code/stefanoblando/rsna-knee-master-4-arm-ensemble)，API lastRunTime 2026-09-26 00:45:28。SHA256 `497f54368a0eebd7b88c8b76c2e0607d73e4aa181b438c88fb21bcfea4a95398`。

代码包含CoAt多成员、Raptor/Rad分支，并尝试加载ConvNeXt-small-384和CoAtNet-384独立checkpoint。若找不到这些权重，会打印回退到标称0.943父模型。新增分支采用保守残差/逐类处理。代码里的Gold AUC默认值不是实测证据，标题中的4-Arm也不保证运行时四臂都加载成功。

对项目的启示：每个分支必须记录实际加载与贡献，避免以为复刻了新模型而实际只跑了父模型。独立CNN+小权重融合与我们路线一致，但该Notebook不能证明我们的弱ResNet34已经有融合价值。

## 3. EXP010：监督质量对照仍有人在做

[公开Notebook](https://www.kaggle.com/code/ziyiming/rsna-exp010-llm-conf-axial-fold0)，API lastRunTime 2026-09-26 06:08:13。代码启动Axial、fold0、3轮、high-confidence LLM labels实验。引用的历史分数在该次复查中未独立验证；没有证据称它是高分方案。

意义：提供另一条训练/监督实验路径，而非只做公开融合；但“LLM置信度更高”不等于标签正确，不能直接照搬置信度阈值。

## 判断与决策

当前抽查高关注代码仍主要沿用强公开主干，加输入工程修复、密集采样、多分支和保守融合。没有找到经本次核验、可保证我们从0.942达到0.945的新公开配方；也未核验当前奖牌边界。

保留0.942比赛锚点；执行Stage6序列审计与监督归因。要超过公开父模型，仍需培养可靠的独立成员，而不是把不同文件名误认为不同模型。后续若尝试最新公开方案，应单独保留完整挂载、checkpoint哈希、有效分支、隐藏重跑和实际Public收据。
