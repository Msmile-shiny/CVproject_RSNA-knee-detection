# RSNA 2026 Knee Abnormality Detection

膝关节MRI的12类异常识别项目，Kaggle指标为宏平均ROC-AUC。**当前确认的最佳Public为0.943**；来源为社区公开模型的复现和融合。

## 当前入口

| 用途 | 路径 / 结果 |
|---|---|
| 主方案 | [四成员构建器](experiments/sprint0930/build_public943.py)，[Kaggle Notebook](https://www.kaggle.com/code/easoncyy/rsna-sprint-public-fourway)；提交 **56700487：0.943** |
| 独立成员对照 | [ConvNeXt 10%构建器](experiments/sprint0930/build_cnx10.py)；提交 **56706632：0.943**，显示精度下未见额外增益 |
| 回退基线 | [Anchor942](experiments/anchor942/README.md)；提交 **56454576：0.942** |
| 标签及来源 | [标签说明](data/README.md)：v5和DeepSeek/GPT融合标签已纳入版本控制 |
| 实验记录 | [冲榜报告](docs/experiments/leaderboard_sprint_20260930.md) |
| 最新对照 | [Outer70执行记录](docs/experiments/outer70_execution_20261004.md)：提交56817443已COMPLETE，Public **0.943**，显示精度下持平；停止该权重方向、保留父本 |
| 正在推进 | [半月板增量计划](docs/plans/meniscus_increment_20261005.md)：已推送独立CPU工程检查v2（修复v1元数据错误），不训练、不评分、不使用GPU；通过后才进入专项融合测试 |
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
