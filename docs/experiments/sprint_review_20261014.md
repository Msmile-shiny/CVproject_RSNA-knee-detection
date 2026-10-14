# 2026-10-14：停止扩展微调，完成已有实验评分

已确认R7原版57144024和无积液校正57147149均Public0.956，停止积液开关方向。镜像候选COMPLETE，外层839.712秒；两worker明确FLIP_TTA_READY及WORKER_DONE，0失败、无超时。MCL原始概率和Raptor可见排名与原版一致。全部源码/哈希/覆盖/融合门禁通过，已单次提交57185504，尚无评分。可见3例不证明隐藏预算或提分。

## 最新社区筛选

官方API按scoreDescending获取前12项，不把该排序或标题当精确Public分数。下载并检查两个公开源：

- heliosli/0-957-latest-dataset-score-in-one-click：与已复现R7的代码差异仅find_mount和挂载候选路径，bundle同为8c7b666e27dc0e8f。不是新权重，不重复运行。
- bliverrigpdoge/r7-raptor-knee-centred-convnext：新增膝中心独立成员dm，最终原基线80%+dm20%；挂载元数据包含空来源。源码依赖bliverrigpdoge/rsna-knee-kcd25leg中的leg_main.py及members.json，官方API访问403，未证实公开可用。该脚本捕获成员失败后保留基线，故直接复刻可能没有实际新增成员。不启动GPU，不绕过权限，不宣称完整复现。

源码已归档experiments/r71014。上述成员若后续合法公开可访问，应先审计权重、训练/标签来源及全量预算，再考虑一次完整新增成员实验。当前不以空资产或同权重的新标题消耗算力；不做融合比例/seed扫描。保护0.956原版及最终选择。镜像仅等待这一次真实评分，持平或下降停止。
