# 双候选冲榜执行记录（2026-10-05）

父本提交56700487 Public0.943保留。Outer70与CNX M448单fold10%均0.943持平，停止对应扫描。本轮直接推进两个具备完整公开资产的独立候选，不开新训练。

## 官方时间和当前排名

10月5日实时`KaggleApi.competitions_list`返回：最终截止2026-10-22T23:59Z，即北京时间10月23日07:59；报名/合并队伍截止10月15日23:59Z，即北京时间10月16日07:59。API当时返回userRank=500、teamCount=5225、每日最多5次提交。排名会变，Public排名不保证Private奖牌。

## 已完成的检查

半月板CPU v2 COMPLETE，`CPU_GATE_PASS`，总352.1073秒，真实3病例预处理约4秒；权重233张量strict load通过，SHA e1443bb0f518418a31428cd7eb153f15af4c48c7175baf1f9d21c84f1c2a93e7。其余时间主要是包/权重哈希与加载。合成前向只证明能运行，不代表AUC。

两份候选均保持父本所有代码单元逐字符一致；新成员在独立子进程先推理并退出，父本随后运行，避免占用父本GPU张量或污染输入设置。每份候选都有8.5小时总计时截止、专项子进程2小时上限及同一次运行内满批显存检查，失败即中止。可见病例不用于选择比例。

## 本轮固定配方和线上版本

| 项目 | 半月板 | 三折ConvNeXt |
|---|---|---|
| 实际Notebook | easoncyy/rsna-sprint-meniscus10 | easoncyy/rsna-sprint-three-fold-convnext-30 |
| 版本/kernelId | v2修复版 / 137197263（v1已ERROR、未提交） | v1 / 137197650 |
| 启动UTC | 2026-10-05 15:40:43 | 2026-10-05 15:44:17 |
| 配方 | 两类：90%父本排名+10%专项排名，其他10列文本token保留 | 十二类：70%父本排名+30%三折概率均值排名，再统一rerank |
| 新资产 | renta0426/rsna-knee-public0033-meniscus-bag-v1 | goodpjw2008/rsna-knee-2-5d-convnext-reader |
| 模型/输入 | DINOv2 Base，原作者6序列12切片336px/130mm；10窗口FP32 | 三fold ConvNeXt-Tiny，6序列×12三切片窗口，256px；FP16 |
| 满批检查 | 单卡32病例形状前向 | 单卡4病例、全部6槽位前向，3模型依次检查 |
| 当前状态 | v1统计初始化错误，v2已启动RUNNING；未获评分 | COMPLETE，门禁通过，提交56856316隐藏评分中 |

ConvNeXt实际slug由Kaggle根据title生成，与构建metadata的cnx3fold30不同，启动回执保存了真实ref。后续一律使用回执response.ref，不重复push。两者完整候选独立评分，不把两个增量一起叠加导致无法归因。

新ConvNeXt作者报告独立模型0.929、加入0.943栈30%后0.944，是作者自述，尚非我方成绩。权重已公开三fold，Apache2.0；相比之前M448单fold，它的架构头、输入与训练均不同。构建固定Notebook版本的三个源码，未混用Dataset和Notebook不同的KneeNet版本。工程改动仅为strict load、安全权重读取、解码异常报错和运行回执，模型/像素数学流程保持。

后续核对确认：去掉作者开头的绘图单元后，其全部23个父栈代码单元与我们保护的父本完全一致，进一步减少父栈不同造成的复现偏差。三折候选实际可见总耗时622.2221秒，reader134.4014秒，三模型满批测试8.66秒，峰值4,167,718,400 bytes，三fold均strict load、零fallback、完整3病例；这些是运行证据，不是隐藏耗时保证。门禁通过后仅提交v1一次，实际提交号56856316。

半月板v1错误为`torch.cuda.reset_peak_memory_stats`在CUDA allocator初始化前调用导致Invalid device argument；真实预处理12/12槽位完成，模型尚未执行，不能将其视为模型准确率失败。已在每卡统计重置前分配并释放单个tensor，配方、像素处理和权重不变，重新推送v2。v1源与启动回执保存在failed-v1，v2以当前launch_receipt为准。

## 收取、验证和提交

从仓库根目录：

```powershell
python experiments/finish_sprint1005.py meniscus --submit
python experiments/finish_sprint1005.py cnx3 --submit
```

RUNNING时命令只查询，不提交。COMPLETE后核对启动version与源码、父本完整成员及异常事件、输出SHA/UID/列/范围、固定融合重算、专项覆盖/显存/精度/模型数；全部通过才提交该Notebook具体版本一次。已有提交尝试时先对账，不重发。CSV只有可见3例，不能用来替代代码竞赛隐藏提交。

隐藏评分后保存实际提交号和Public成绩：高于0.943列候选；持平/下降停止对应候选，保留父本；失败报告工程原因，不能弱化门禁通关。最终提交选择仍由用户决定，Public增益不保证Private增益。

已设置本线程每小时跟进`automation-2`，仅处理这两份已启动版本的完成验证、一次评分和结果记录，两个终局后停用。无变化时安静。依据OpenAI Docs的[定时任务说明](https://learn.chatgpt.com/docs/automations?surface=app)，使用同线程跟进；不扩大为新训练或参数扫描。

当前本地测试：3项半月板overlay测试、3项评分门禁测试通过；两份Notebook格式/语法及父本代码保留检查通过。测试不会证明模型准确率。

## 下一决策

先获得两份实分。若三折模型提高，可用其公开训练源码作为后续持续优化基线，先检查每类错误和训练/推理一致性；当前不追加多折训练。若两条都持平/下降，就停止这两个增量并重新选择有独立证据的路径，不以换权重名延长试验。
