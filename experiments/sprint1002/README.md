# Outer70：额度恢复后的一次推理对照

终局状态：v1可见推理及输出检查通过；提交56817443 **COMPLETE，Public0.943**。与父本显示精度下持平，停止该权重方向、保留父本56700487，不更改最终提交选择。评分记录见`scoring_receipt.json`。禁止重复push或提交。

父本是提交56700487、Public 0.943的四成员方案。只将最终融合的默认CoAt/Raptor分支权重由0.60提高至0.70；另一分支相应从0.40降至0.30。影响MCL、Medial OA、PF OA、Effusion、Synovitis、Baker's、Contusion。ACL、两类半月板、Lateral OA、Fracture的专用权重保持原值。

这是已有社区思路的一次严格对照，不是自研模型，也没有已知提分保证。整个组合仍只推理一次；结束时用相同分支输出重算旧配方，便于验证五类未改动。

## 本地准备（不启动GPU）

```powershell
python experiments/sprint1002/build_outer70.py
python experiments/sprint1002/test_outer70.py
python experiments/sprint1002/preflight.py
```

构建器锁定父本SHA，预检查对照全部代码及挂载清单。沿用父本14个Dataset、2个Notebook输出、1个DINO模型和比赛数据，无新增挂载。全部具体名称在`outer70/kernel-metadata.json`。10月2日在线清单17/17访问通过；下载的四个CoAt manifest指纹与父本匹配，相关文件名齐全。证据在`remote_audit.json`及`asset_check_receipt.json`；权重字节仍由运行时加载器检查。

## 额度恢复后执行

以下命令会开始消耗Kaggle GPU，只在账户确认额度恢复后执行：

```powershell
kaggle kernels push -p experiments/sprint1002/outer70 --accelerator NvidiaTeslaT4
kaggle kernels status easoncyy/rsna-sprint-outer70
```

完成后下载并检查`outer70_receipt.json`：`ready_for_scoring=true`、四个CoAt成员完整、父本审计无异常，再提交该完成版本的`submission.csv`。记录实际Notebook版本号及提交号，不复用父本或CNX10的旧版本号。测试阶段不自动上传或提交。

下载输出后先执行`python experiments/sprint1002/check_result.py --output results/outer70`，自动核对源码身份、文件SHA、病例/列、数值范围、两种融合公式、异常事件及内部计时。还需在Kaggle确认对应版本COMPLETE及整个运行时长；通过可见检查不等于隐藏运行已通过。

输出包含`baseline60_replay.csv`、`transformer_branch_rank.csv`、`coat_raptor_branch.csv`及候选`submission.csv`。可见测试只有3例，这些文件用于检查程序，不能据此选权重或估计AUC。隐藏评分中的全量预测不等于可下载的可见输出。

## 决策与预算

- 新候选显示分数高于0.943：暂列候选，保留旧0.943，进一步检查完整运行和版本来源；不视为Private改善证明。
- 显示同为0.943：记作显示精度下无增益，停止这个权重方向。
- 低于0.943：继续使用父本，停止这个方向。
- 程序或依赖失败：先修工程问题；失败运行不用于模型优劣判断。

额度恢复后的第一项只安排这一次可见推理及一次评分。公开3例父本曾用约252秒，不能用它推断隐藏测试耗时；组合模型未增加，但仍需遵守比赛9小时上限。首轮最多预留1小时账户GPU用于启动与可见运行，超过时先检查日志，不自动重跑。保留其余额度，直到另一个候选具备明确资产、代码和验证目标。
