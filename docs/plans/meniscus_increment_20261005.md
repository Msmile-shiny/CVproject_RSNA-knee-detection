# 0.943父本之后：半月板增量，2026-10-05

后续更新：CPU v2已COMPLETE/PASS，已进入完整双T4候选v1。最新状态与并行ConvNeXt候选见[执行记录](../experiments/sprint1005_execution.md)，以下保留当时计划。

## 已执行

- 已把Outer70 0.943持平结论与公开资产审计提交、推送到main：faaee8cdbf6272e97f8abf92828d70872a715faf，远端SHA已核对。
- Outer70停止，不再扫描这条融合权重方向；父本56700487保留。
- 下载并审计两份公开高分标题Notebook。`sushanthtiruvaipati/rsna-knee-d4-mcl-latmen-blend-v1`明确更正自身为0.943；`ryokucha/rsna-knee-d4-blend-0946-ours05`含缺失资产与私有ConvNeXt路径。不能将标题当可复现成绩。
- 通过Kaggle搜索找到原作者真正公开入口：[半月板残差Notebook](https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid)。此前403针对旧slug，不再作为源码不可得的结论。
- 对比AST：pick_slots、lat_of、normalise_laterality、adopt_config_globals一致；order_slices和read_slot不同。不能仅看缓存shape相同就宣称等价。
- 已构建并推送独立CPU工程检查，kernelId137193788：[CPU gate](https://www.kaggle.com/code/easoncyy/rsna-meniscus-cpu-gate)。v1因构建器遗漏kernelspec，在执行代码前报错；已修复并加入nbformat格式验证，推送v2。Internet/GPU均关闭，不提交比赛、不训练、不运行20成员父本。

## 当前实验合同

CPU gate沿用原作者预处理代码和已核验DINO manifest的336px/130mm/12切片配置，去掉原模型集成的启动调用。它检查可见病例的UID、缓存shape/dtype、非空序列，核验包内文件哈希，严格加载专项权重并执行作者的合成单病例CPU前向测试。

合成前向不等于真实病例准确率；真实缓存检查不等于与当前父本逐像素一致。成功输出`CPU_GATE_PASS`才完成这道工程门槛。若只有`PREPROCESS_AND_ASSETS_PASS`，模型加载/前向仍未完成。没有`submission.csv`，不可将该Notebook提交比赛。

## 后续步骤与预算

1. 收取CPU日志/回执并验证实际版本。失败时仅修具体依赖/输入问题，不转为完整GPU盲跑。
2. 若通过，优先为专项成员保留独立的原作者预处理；不修改父本现有预处理。避免为了复用缓存引入不可解释变化。
3. 下一候选固定为父本90% + 专项10%的两类半月板排名融合，其他10列数值/序列化token保留。注意当前公开作者Notebook后来改为只调内侧半月板并混合三个分支；不声称我们的两类10%方案是其0.937精确复刻。
4. 先用一次小规模双T4工程测量确定增量耗时/显存；专项10窗口FP32开销必须计入总预算。第一轮可见GPU检查最多预留1小时；异常不自动重跑。当前尚未启动GPU或自动评分。
5. 父本仍为0.943。通过完整运行和输出检查后只提交一个固定候选；若显示持平/下降，停止该候选，不扫比例。若提升，只列候选，不能担保Private或奖牌。

## 为什么选择这一条

它不需要重新训练，公开权重和原始输入流程现在均有入口，且能把影响限制在两类。独立成员是指能补充正确排序信息，不是“换个模型名”；DINOv2同架构也可能互补，但必须实测。当前CNX10已经持平，继续用同一公开fold扫比例优先级更低。

这个方案仍是有明确停止线的候选，不是保证铜牌的捷径。CPU检查期间不启动重型3D预训练，不把已失败的旧标签路线重新包装。

## 重建

先用Kaggle API将原作者Notebook拉取到`results/research1005/meniscus`（只读下载），再运行`python experiments/research1004/build_cpu_probe.py`。构建产物已保存到`experiments/research1004/cpu-probe`。已有v1启动和v2修复记录，禁止重复运行launch脚本；先查询状态。
