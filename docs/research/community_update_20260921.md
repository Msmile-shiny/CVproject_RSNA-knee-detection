# 2026-09-21 社区更新与下一步

## 当前变化

Kaggle Code 的 score-descending 列表已将 `maverickss26/rsna-knee-0942-restructured` 放在首位。该 Notebook 自述 Public 0.942，并提供完整公开依赖。相比我们已复现的 0.941，它不是单一新模型：

- A5 融合权重 0.45 → 0.52；
- RadImageNet 主分支 0.50 → 0.55，第二遍 0.15 → 0.20；
- 四个 Raptor 视图均使用 94 个推理窗口；
- 原 residual-gated CoAtNet 与新增 D4 depth-zone CoAtNet 先等权排名融合，再以 0.40 进入 Raptor 分支；
- 外层逐病变权重保持 0.941 的 probe22 映射。

这套配置包含多个同时变化的参数，并在公开榜上发展而来。它适合用作更强公开锚点，不适合作为某个机制有效的因果证据。

## 对其他新 Notebook 的判断

- `haideptry/...-0943` 页面写的是 **0.943+ Target**，并加入概率/排名共识、积液到滑膜炎先验及挫伤权重；标题是目标，不是已核实的实际 0.943。暂不采用。
- `sushanthtiruvaipati/...-0942-rad-fracture-oof` 与 0.942 源码仅有说明单元和 RadImageNet 的 Fracture 排除设置差异。标题不足以证明独立增益，等待实分或做严格对照。
- 社区单模型讨论报告 0.942 raw、0.943 TTA，以及五折模型 0.943/0.947；这些没有同等完整的公开权重与配方，暂时不能直接复现。
- Native64 已由我们实际验证为 0.940，停止该路线。

## 已推进

已从公开 0.942 源码生成 `easoncyy/rsna-anchor942-audited`。v1 完成全部模型推理，但新增的最终审计单元引用了错误的根目录变量，因此没有提交。v2 只修正该审计变量，未改模型和融合配方；禁网双 T4 运行完整，随后提交为 Kaggle submission `56446116`。

v2 的执行回执显示：3 个可见测试病例均完成，CoAt family 同时包含 `resgated_top3` 和 `d4_swa3`，residual fallback 为 0；最终 UID 顺序、有限概率和范围检查均通过。这个结果证明执行完整性，榜分仍以 Kaggle 最终评分为准。

## 后续判定

- 实际 0.942 或更高：将它设为新锚点，下一次只测试 Fracture 是否重新进入 RadImageNet 分支。
- 实际 0.941：保留原锚点，说明当前公开源码/资产未在本账户复现显示增益。
- 低于 0.941：检查版本、D4 输出与融合回执，不追随标题继续叠加启发式规则。

submission `56446116` 在公开三病例运行中完整通过，但 Kaggle 的更大隐藏数据重跑抛出异常，因此没有榜分。Kaggle 按规则只返回异常类别，不提供隐藏 traceback。最可能的原因是我们的审计门禁过严：公开原版会记录单病例回退或子模型失败并继续，而审计版把它们升级成 fatal exception。

hidden-robust 版本已从未修改的公开源码重新构建，32 个推理代码单元逐字节一致；只在最终输出之后检查 CSV 并记录是否降级。公开运行完整且未降级，已作为 submission `56454576` 提交隐藏评分。Fracture 单变量消融继续冻结；只有该提交真正获得至少 0.942 后才放行。

**2026-09-24 后续：** `56454576` 已得到 Public **0.942**，Fracture 对照放行。当前决策和后续评分见 [项目复盘与路线](../experiments/project_review_20260924.md)；上文保留了当时的实验前判断。

来源：

- https://www.kaggle.com/code/maverickss26/rsna-knee-0942-restructured
- https://www.kaggle.com/code/haideptry/rsna-knee-speedy-raptors-coatnet-d4-0943
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304
