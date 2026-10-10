# RSNA 2026 Knee Abnormality Detection

膝关节MRI的12类异常识别项目，Kaggle指标为宏平均ROC-AUC。**当前确认的最佳Public候选为0.950**（用户0.95+版提交57012367）；来源为社区模型及多项推理/融合改动，不能归因单项。保留0.949（56994783）和0.943回退，最终提交选择未改。

## 当前入口

2026-10-10：[独立224固定15%融合](docs/experiments/swa224_20261010.md)可见运行94.574秒、两级融合核验PASS，已提交 **57034764**。保留0.950父本，等待隐藏评分，不重复提交，不改变最终选择。

后续[双候选终局](docs/experiments/apex_next_20261009.md)：用户版57012367 Public **0.950**；单因素probit提交57012629已COMPLETE Public **0.949**，与其父本持平，停止该方向。不能将用户版提升归因probit。本次跟进结束，最终选择未变。

2026-10-09终局：[Apex审计与优化](docs/experiments/apex_20261009.md)。提交 **56994783**（356635120）隐藏COMPLETE，Public **0.949**，较0.943父本提高0.006；列为候选，不改最终选择，不保证Private或奖牌。跟进已结束，禁止重复提交。

| 用途 | 路径 / 结果 |
|---|---|
| 主方案 | [四成员构建器](experiments/sprint0930/build_public943.py)，[Kaggle Notebook](https://www.kaggle.com/code/easoncyy/rsna-sprint-public-fourway)；提交 **56700487：0.943** |
| 独立成员对照 | [ConvNeXt 10%构建器](experiments/sprint0930/build_cnx10.py)；提交 **56706632：0.943**，显示精度下未见额外增益 |
| 回退基线 | [Anchor942](experiments/anchor942/README.md)；提交 **56454576：0.942** |
| 标签及来源 | [标签说明](data/README.md)：v5和DeepSeek/GPT融合标签已纳入版本控制 |
| 实验记录 | [冲榜报告](docs/experiments/leaderboard_sprint_20260930.md) |
| 最新对照 | [Outer70执行记录](docs/experiments/outer70_execution_20261004.md)：提交56817443已COMPLETE，Public **0.943**，显示精度下持平；停止该权重方向、保留父本 |
| 双候选终局 | [执行记录](docs/experiments/sprint1005_execution.md)：半月板56866776 Public0.943持平，停止该增量；ConvNeXt v2提交56867864隐藏重跑再次报错，无分数，不自动重跑。保留0.943父本，最终选择未变 |
| 当前排查 | [失败复盘与下一步](docs/experiments/cnx_failure_review_20261006.md)：CPU扫描21,347条选中序列全部通过，解码/过短/空病例异常为0。下一排查重点为完整规模执行预算，隐藏根因未确认；本次未新增GPU任务 |
| 推理诊断结果 | [1,024病例诊断](docs/experiments/cnx_scale_probe_20261007.md)：全部通过，65分23秒，读图约57分钟，GPU传输/推理约6.8分钟，峰值GPU分配4.17GB。未复现异常或两小时超时；下一重点为整栈整合/累计预算 |
| 独立评分终局 | [ConvNeXt独立版](docs/experiments/cnx_solo_execution_20261007.md)：56905796隐藏重跑同样失败，无有效分数；停止原样重试。排查回到reader约束/隐藏输入差异，父栈不是解释独立失败的必要条件；预取尚未GPU实测 |
| 最新执行对照 | [作者v5独立模式](docs/experiments/cnx_author_control_20261007.md)：提交56910183隐藏评分成功，Public **0.929**；同组三折权重可用，具体失败根因仍未确认。下一步准备执行差异审计与父栈整合，不因独立分低于父本而淘汰互补性；最佳与最终选择不变 |
| 固定融合终局 | [10月8日固定融合](docs/experiments/cnx_author_fusion_20261008.md)：提交 **56947372**（356315744）隐藏COMPLETE，Public **0.943**，与父本显示精度下持平；停止此方向，不扫描比例，保留父本与最终选择 |
| 仓库整理 | [main合并与整理记录](docs/experiments/repository_cleanup_20261001.md) |
| 历史材料 | [history](history/README.md)、[文档索引](docs/README.md) |

Stage6F24已完成，Gold58 AUC为0.84724，6E24为0.84690；尚无改善0.943父模型的证据，继续扩展训练暂缓。Gold58已被多次用于开发，不能当作独立测试集。Outer70隐藏评分已完成，Public0.943未见显示精度下增益；不再扫描该方向权重，当前没有进行中的训练，最终提交选择未改动。

## 重建与检查

从仓库根目录运行：

```powershell
python experiments/sprint0930/build_public943.py
python experiments/sprint0930/build_cnx10.py
python experiments/sprint0930/test_rank_blend.py
```

构建仅生成本地Notebook，不启动Kaggle GPU。挂载以对应目录的`kernel-metadata.json`为准。代码竞赛须提交已完成Notebook的具体版本，不能用三个可见测试病例的本地CSV替代隐藏运行。

## 目录约定

- `experiments/sprint0930/`：当前0.943方案、候选及回执。
- `experiments/anchor942/`：可靠回退；`experiments/stage6/`：自研实验记录。
- `notebooks/`、`scripts/`：构建器与共享实现；仍被引用的旧阶段源码保留原路径。
- `data/`：元数据、标签和来源；DICOM、缓存和权重不入Git。
- `history/`：旧0.941方案、初期探索、旧计划、截图、失败日志和打包快照。
- `docs/`：实验报告、研究和决策历史；文档中的“当前”仅指各自日期。

后续默认在`main`工作。本次保留旧分支和Git历史；分支仍存在不代表尚未合并。提交时排除凭据、API配置和大型缓存。
