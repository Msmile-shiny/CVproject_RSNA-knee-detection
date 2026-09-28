# 社区与论文复查：下一项可验证的提分实验（2026-09-27）

## 对本项目最有用的新证据

1. [Kaggle单模型讨论](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)中，参赛者自报224–288px的小模型也能到0.93–0.94以上；一位参赛者称单折0.938、五折0.947，另一位称288px单模型0.942、TTA 0.943。这些是帖子自述，缺同等可核验的权重/数据/隐藏提交链路，不能当我们当前ResNet34的预期分数。帖子里也有人称独立分支能带来Public +0.004，但我们的弱成员尚未验证与0.942父模型互补。
2. [DINOv2 Small→Base消融](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154)报告约3.9倍计算换来CV +0.0011，低于作者估计0.0020的噪声底；它针对作者自己的训练配方，不证明本项目加大骨干一定无效，但支持先排查输入与监督。该帖同时强调预训练权重的归一化约定；本项目使用官方ResNet34 ImageNet均值/方差，代码已核对。
3. [社区关于小CNN与标签的讨论](https://www.kaggle.com/tuckerarrants/discussion)建议先用简单池化和224/288px，验证报告标签而不是用58个Gold反复追逐0.002–0.003。他指出修正报告提取错误不一定改善Public。我们6C的结果与“更接近报告教师并不必然更像官方标签”一致，但不能由此推断所有文本监督无效。
4. [SOFT/HARD强配方对照](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734105)中，软目标在三个成对seed上均高于硬目标，但58例Gold的差异置信区间跨零，作者撤回了原先的机制解释。对本项目而言，应保持概率、权重、mask分开改；6C已经完成概率单变量对照，不宜接着搜索很多比例。
5. [CoPAS原论文](https://www.nature.com/articles/s41467-024-51888-4)用同一平面多序列的注意力处理1748名患者的12类异常，报告研究自身数据总体AUC 0.812。这提示先保持序列身份和缺失mask，再尝试跨序列对应；其AUC与本竞赛Public不可比较。当前6D仅对切片做类别注意力，非CoPAS复现。
6. [半月板定位与分类研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC11811310/)先分割半月板前角、体部、后角，再裁局部区域分类；弱监督定位在部分分区也有可用表现。[另一项以手术为参照的研究](https://pubmed.ncbi.nlm.nih.gov/32170334/)选择矢状和冠状脂肪抑制序列，并围绕半月板裁剪。它们支持未来建立局部半月板成员，但需要可靠定位器/标注，不能把当前attention权重误称病灶框。

## 本项目输入几何审计

复用此前已下载的Stage6序列头审计，对20,477条实际被槽位选择的series连接`sample_shape × sample_spacing_mm`，脚本`scripts/audit_stage6_geometry.py`和汇总`experiments/stage6/audit_v1/geometry_summary.json`可复算。五个有数据槽位的最大视野中位数都约160mm，10–90分位大致150–180mm。当前完整视野缩到224px，典型采样约0.714mm/px；单纯升到288px可到约0.556mm/px，输入面积和缓存上限约增至1.65倍。固定224px只把物理中心视野收至150mm，约0.670mm/px，且会裁掉周边组织。社区有人使用392px/150mm（约0.383mm/px），但其结果是自报，不等于我们应直接付出约3倍像素代价。

这只是头信息推算：尚未用像素图确认真实病灶清晰度、裁剪中心或异常DICOM。尤其MCL、骨折等类别可能依赖周边组织；直接全类收窄视野有风险。

## 决策与已启动动作

- 保留0.942父模型不变，6C不作为全类新成员。6C的报告标签在复用Gold上对外侧半月板AUC约0.846，6C图像模型该类上涨，但ACL/MCL/骨折下降；这是局部研究线索，不据此配置逐类权重。
- [Stage6E覆盖冒烟](https://www.kaggle.com/code/easoncyy/rsna-stage6e-coverage-smoke) v1：224px不变，仅每槽4→8个位置；`SMOKE_COMPLETE`，8训练/4留出/4 Gold、CUDA梯度非零、耗时478.8秒。16例压缩像素缓存共65.2 MB，只用于粗估全量空间；4例Gold AUC不具评价意义。
- [Stage6F分辨率冒烟](https://www.kaggle.com/code/easoncyy/rsna-stage6f-resolution-smoke) v1：每槽4位置不变，仅224→288px；`SMOKE_COMPLETE`，相同16例、CUDA梯度非零、耗时1305.2秒。6E/6F的压缩缓存分别65.2/51.6 MB；由16例粗外推全量约18.0/14.2 GB。小样本墙钟时间受DICOM与挂载I/O波动影响，不能直接断定6F全量必为6E的2.7倍；训练单轮8例耗时仅3.24/3.60秒，差异主要在准备阶段。两项冒烟的4例Gold AUC均无评价意义。
- 选择先跑6E固定折12轮：[Stage6E完整训练](https://www.kaggle.com/code/easoncyy/rsna-stage6e-coverage-pilot) v1已推送。理由是8位置仍使用已验证224px解码和空间尺度，冒烟成本较低；也能检验之前从未完成的深度覆盖假设。Notebook启动时把16例实际压缩缓存外推至4407例，要求剩余空间高于预计缓存加1 GB；若空间不足会在解码前失败，不会消耗数小时。真实全量占用可能偏离该小样本估计，需要以云端收据为准。6F保留为下一项单变量候选，不同时全量启动。
- 2026-09-27后续查询：6E完整训练在Kaggle仍为`RUNNING`，尚无完整预测/Gold AUC。本地增加`--resume-after-paused <run_receipt.json> --resume-source <owner/kernel>`生成入口，仅在收据明确为`PAUSED_PREPROCESSING`或`PAUSED_TRAINING`且哈希/配置匹配时启用；对训练暂停还要求`last.pt`。当前运行未暂停，不启动续跑。连续多次暂停若缓存分布于多份输出，需先合并/挂载全部缓存来源，不能只指向最近一份输出。
- 完整训练须在同一852留出病例、旧v5目标及Gold逐例排序上对照6B，再按预先写定的诊断标准决定是否续至24轮。此次不叠加150mm裁剪、注意力、报告标签或多折。
- 如果4→8/224→288都无可靠收益，下一项是从完整视野224px改为150mm物理中心裁剪的单变量实验，并先导出多病例、多平面的可视化确认解剖区域没有被裁掉。若看到外侧半月板的重复收益，再核查真正的半月板定位资产与外部证据，建设专科成员。
- 有足够强的单折候选且与父模型同病例排序互补后，再考虑五折/小权重；五折不用于拯救一个明显弱且错误相关的单折模型。

结论：近期社区的0.94+自报配方没有一个可直接保证本仓库升至0.945；目前最便宜、最可归因的机会在输入清晰度与覆盖，其次才是解剖定位。任何Gold58收益仍属于开发诊断。
