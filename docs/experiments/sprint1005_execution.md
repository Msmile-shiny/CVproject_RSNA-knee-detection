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

### 2026-10-06 20:11 北京时间：两个候选终局

- 半月板v2提交56866776，scriptVersionId355502633：API COMPLETE，Public **0.943**，无errorDescription。与父本56700487显示精度下持平，停止该固定10%增量，保留父本；不能将持平当作Private改善。
- ConvNeXt v2提交56867864，scriptVersionId355605911：API COMPLETE但errorDescription为隐藏重跑unhandled error，Public缺失。仍是失败，不是0分。取消多进程预取及CUDA初始化修复没有解决隐藏失败；此前不能确认根因的判断保留，不再自动追加重跑。
- 两个终局回执分别保存scoring_v2_receipt.json和scoring_receipt.json。没有新最佳成绩，最终选择未改变，不启动新GPU、训练或权重扫描。本次跟进automation-2停用。

### 2026-10-06 10:27 北京时间跟进

ConvNeXt v2可见COMPLETE，版本与远端源码核对、完整覆盖、三fold、满批显存、融合重算等门禁全部PASS。总耗时276.353483504秒，reader51.101626715秒，推理12.520523549秒，满批7.962192423秒、峰值4,167,718,400 bytes。原始reader预测SHA与v1完全相同：1606432b738e10e2ed05223ca67ecb3dd6e92195469f9fbb5cd6aa80ca17f55f，说明可见输出未因执行方式改变。速度差异不能全归因于修复或外推隐藏耗时。

已仅提交v2一次：56867864，确切kernel version2；隐藏结果尚未返回，不能称隐藏异常已解决。半月板56866776亦无分数/错误结果。下一步仅跟进两提交，不重复推送/提交，不新增GPU任务，保留父本及最终选择。

### 用户要求再审计后的执行方式修复v2

用户明确授权再次审核、修复后重新执行。ConvNeXt v1只有3例/一个batch，满批GPU压力测试没有覆盖多进程连续预取的CPU/shared-memory路径；原worker在CUDA模型加载后启动4个loader worker，每个完整uint8 batch为4×6×12×3×384×384=127,401,984 bytes，默认8个预取batch约1.02GB张量，另有解码、进程及锁页内存。此为可验证的资源风险，**不是已确认隐藏失败根因**。

v2仅改为num_workers=0、显式CUDA allocator初始化及批次进度打印；保留batch4、FP16、12windows、三fold、原始像素、全部异常检查、固定30%融合和父本源码。取消预取可能降低吞吐，原2小时reader及8.5小时总时限仍保留，实际性能须复测。

本地d2l环境的多批次顺序测试和配置测试2项PASS，原门禁测试3项PASS，Notebook格式/语法构建通过，三个模型/像素源码SHA均未改变。v1源及回执归档failed-v1；v2实际kernelId137197650、versionNumber2、源SHA e069edfc4b96954f3ca9b729afd5f0da7d47a27ff25e8c48d86015dc5729239f，以launch_receipt记录为准。已启动，未经运行核验不得评分；不再自动追加修复重跑。

依据OpenAI Docs的[定时任务说明](https://learn.chatgpt.com/docs/automations?surface=app)，已更新同线程跟进到ConvNeXt v2和半月板提交56866776；v1失败不重复通知。

### 2026-10-06 09:27 北京时间跟进

- 半月板v2可见运行COMPLETE，完整门禁通过，1266.873852秒。已仅提交一次：56866776，scriptVersionId355502633；尚无评分结果。
- 三折ConvNeXt提交56856316的API状态为COMPLETE，但errorDescription明确为隐藏重跑unhandled error，Public为空。这是失败终局，不是有效评分完成，更不能记为0或准确率下降。
- API仅提供通用错误，没有隐藏堆栈；当前不能确认OOM、解码或其他根因。可见3例通过不证明隐藏规模通过。不自动重推、重交或弱化门禁。
- 保留父本56700487 Public0.943和最终选择不变；自动化继续仅跟进半月板评分，两个终局后停用。

先获得两份实分。若三折模型提高，可用其公开训练源码作为后续持续优化基线，先检查每类错误和训练/推理一致性；当前不追加多折训练。若两条都持平/下降，就停止这两个增量并重新选择有独立证据的路径，不以换权重名延长试验。
