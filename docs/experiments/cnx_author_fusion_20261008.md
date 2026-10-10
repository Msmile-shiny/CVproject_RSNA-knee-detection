# 2026-10-08 固定30%融合执行

## 终局：10月9日07:10北京时间核验

提交56947372（scriptVersionId356315744）已COMPLETE，Public **0.943**，API未返回errorDescription字段。与父本56700487在显示精度下持平，没有已证实的Public提分。停止此固定融合方向，不扫描更多比例，不重复提交；保留父本及最终选择，停用本次自动跟进。

原作者reader接入后，完整融合能够通过隐藏运行；这证明本次执行路径可用，但不能确定旧版异常的单一根因。独立0.929与融合0.943不证明reader完全无用，也不能证明Private提升。后续更强SWA主模型属于另一个候选，本次没有启动它或任何新GPU任务。

以下为历史执行记录，状态以本节和author_fusion_scoring.json为准。

实时确认作者独立版56910183为COMPLETE、Public0.929。没有发现本地Python训练进程；父本56700487 Public0.943和最终选择不变。

## 已执行

新Notebook实际地址 `easoncyy/rsna-author-reader-fixed-30` v1，kernelId137622801，已启动，最近查询RUNNING。Kaggle按title生成了不同于请求slug的地址，后续必须使用launch回执中的实际ref，不能按旧请求slug查询或重推。

按顺序完整保留已评分的作者独立Notebook代码及父本代码，中间保存reader预测，最后固定融合：rank(0.7×rank(parent)+0.3×rank(reader))。不扫描比例，不改权重、输入预处理或父本模型。reader先运行，避免父本残留显存干扰；这不是隐藏根因已经证实的结论。

构建检查两段原始代码序列完全一致，所有代码单元通过语法和Notebook格式检查。三个合成测试通过：乱序病例对齐/融合公式、缺失ID拒绝、父本门禁失败拒绝。测试曾遇Windows GBK编码及临时目录权限错误，已通过显式UTF-8和仓库内临时目录修复；这不是模型错误。启动发生在这两项本地环境修复之前，测试最终通过，尚未提交竞赛评分。

## 后续门禁

### 17:40北京时间进展

v1可见运行COMPLETE，累计274.275秒；精确版本/源码、三份权重、reader源码、ID、概率及固定融合公式核验PASS。已提交 **56947372**（scriptVersionId **356315744**），回执已保存，禁止重复提交。首次查询返回提交记录但尚无status/publicScore/errorDescription字段，因此仅记录等待评分，不宣称隐藏成功或失败。3个可见病例不是隐藏覆盖证据，decode_coverage_verified仍为False。

后续仅查询56947372的评分，不再运行提交动作。终局比较0.943父本并更新报告。

运行 `python experiments/convnext1005/collect_author_fusion.py --submit`。核对实际远端版本与源码、reader三权重与运行时源码哈希、两路输出ID及概率、融合公式、累计8.5小时预算。门禁通过只提交一次，已有attempt无receipt先对账。运行或隐藏失败不自动重跑，保留日志。

reader沿用已评分原作者缺失序列策略，decode_coverage_verified=False；不可宣称全DICOM覆盖。运行成功只证明执行完成，是否提分必须看隐藏Public结果。持平或下降保留父本；提升仅列候选，不保证Private或奖牌。

本次未训练模型，未修改最终选择，未commit/push。构建、启动与后续验证回执位于 `experiments/convnext1005/author_fusion_*.json`。
