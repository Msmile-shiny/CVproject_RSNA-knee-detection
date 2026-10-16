# RSNA 2026 Knee Abnormality Detection

膝关节MRI的12类异常识别项目，Kaggle指标为宏平均ROC-AUC。**当前确认的最佳Public为0.956**：R7原版57144024及关闭积液修正版57147149均已COMPLETE，比R6的0.954提高0.002。保护R7原版，停止未显示增益的积液开关方向；保留0.954（57131878）及更早回退。最终提交选择未改，不保证Private或奖牌。

## 当前入口

**2026-10-16**：S6 固定融合 v1 已 COMPLETE，881.61 秒、可见三例完整覆盖、源码/权重/哈希/公式独立核验通过；已提交 **57257123** 等待隐藏评分。新增 `experiments/s61015/scoring.py` 只读跟进，不重复提交；保护 0.956，不更改最终选择。详见 [S6 执行记录](docs/experiments/s6_20261015.md)。下方为历史状态。

**2026-10-15 新候选已启动**：[R7 + S6 all-data](docs/experiments/s6_20261015.md)，`easoncyy/rsna-r7-s6-all-data` v1（138878127）。固定 50% Raptor + 40% R7 + 10% 新 S6，避免重复 Raptor 推理；本地源码/公式门禁通过，GPU 结果及 Public 尚未确认。保护 57144024 的 0.956，按用户要求暂不处理最终选择。以下“无待运行候选”是此前快照。

2026-10-15：[执行新资产筛选](docs/experiments/asset_screen_20261015.md)：已归档nartaa新S6单模型及dkim54 v11c的公开清单/源码。前者训练配方不同，优先审计；后者20模型暂缓。尚未启动GPU或证明融合收益，最终选择仍待截图核验。

**截止时间纠正**：官方API最终提交deadline为UTC 2026-10-22 23:59，即**北京时间10月23日07:59**；10月16日07:59是报名/组队合并截止。见[剩余窗口计划](docs/experiments/deadline_plan_20261015.md)。Kaggle最终选择尚未核实（浏览器读取超时，提交API无选择字段），不得将保护版本等同已选版本。

**2026-10-15终局**：[候选对账](docs/experiments/finalists_20261015.md)确认三模型57197910也为Public0.956。原版、无积液校正、镜像、三模型均已评分；停止这些增量方向及比例扫描，保护原版57144024。新增只读对账入口`python experiments/r71014/refresh_finalists.py`。当前无本轮待评分候选或新GPU任务；最终选择未读取/修改。

**2026-10-15最新**：三模型v1已COMPLETE、642.51秒、完整成员及公式核验PASS，提交 **57197910** 等待评分。镜像 **57185504** 已Public **0.956**持平，停止此方向。当前保护原版57144024 Public0.956，最终选择未改；下方为历史进展。

2026-10-14：[完整三模型候选](docs/experiments/threeway_20261014.md)已启动easoncyy/rsna-r7-full-three-model v1，kernelId138687818。固定公开配方384 Raptor 32.5% + 独立224 17.5% + R7 50%，拒绝缺失成员回退，尚未评分。不追加Reader/S6或比例扫描；镜像57185504仍待评分，保护0.956。

2026-10-14最新：[资源收敛与社区审计](docs/experiments/sprint_review_20261014.md)。镜像候选已COMPLETE并通过核验，提交 **57185504** 等待评分。标称0.957的Helios代码并非新模型；新增dm成员方案依赖数据集403，禁止静默退回基线冒充升级。停止扩展积液/镜像小实验，保护0.956。

2026-10-14：R7两候选均Public0.956；新建单项训练配套镜像推理候选easoncyy/rsna-r7-training-mirror v1（kernelId138613104），已启动，尚未评分。保持模型、校正器、50/50融合不变，仅启用原作者镜像及内外侧标签交换，MCL使用原始预测。待完整日志/结果及预算检查后才考虑提交，不扫描比例。

2026-10-13：[R7公开0.956标称方案复现](docs/experiments/r7_20261013.md)基线 **57144024**（357976215）等待隐藏评分；关闭积液修正对照已完成（16分27秒），源码/哈希/覆盖/融合及可见病例11个非目标列一致性核验PASS，已提交 **57147149**。本方尚无两者得分，不覆盖0.954，不重复提交或新增扫描。

2026-10-13：[社区Compact完整融合](docs/experiments/community_compact_20261012.md)v3隐藏评分已COMPLETE，提交 **57131878** Public **0.954**，未返回错误。可见运行645.38秒不能当作隐藏耗时。只读跟进入口：`python experiments/community1012/check_scoring.py`；不重复提交成功版本。

2026-10-12：[关闭疾病关联修正](docs/experiments/nopeer_20261012.md)提交 **57097443**已COMPLETE Public **0.950**，显示精度下持平，停止此方向。

2026-10-12：[8视角Reader对照](docs/experiments/reader8_20261010.md)提交 **57087361**（357096578）已COMPLETE Public **0.949**，低于0.950父本，停止此方向。显存探针v2在离线依赖安装失败，尚无GPU收益证据，不继续重跑已被淘汰路线。

2026-10-10：[独立224固定15%融合](docs/experiments/swa224_20261010.md)提交 **57034764** 已COMPLETE，Public **0.950**，显示精度下持平；停止这一固定224方向，保留0.950父本，最终选择未变。

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
