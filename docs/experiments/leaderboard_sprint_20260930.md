# 9月30日冲榜纠偏：从实验数量转向强基线净增益

## 问题到底在哪里

Kaggle 提交 API 本轮核实：当前最佳 Public 仍为 0.942；9月24日骨折分支对照也是0.942。截至本轮启动前，此后六天没有新评分提交。Stage 6 的多次训练有开发集结果，却尚未带来一个晋升到0.942正式提交中的自研成员。不能把“实验完成”当成“冲榜推进”。

主要问题是优先级和证据链，而非缺少可尝试的模型名：

1. **优化对象偏离了现有最强系统。** ResNet34从Gold约0.82提升到0.85，并不能推出加入0.942父模型有益。需要评估“父模型+成员”而不只是成员本身。
2. **验证资源被反复使用。** Gold58很小且参与过多次选择；852例留出测的是伪标签拟合，不是另一组人工真值。微小局部收益无法可靠筛选榜分收益。
3. **高成本重训练占据了短期冲榜预算。** 数据处理、单变量控制有工程价值，但最近缺少从结果到实际提交的闭环。
4. **社区方案没有及时转为可执行资产。** 旧锚点只有两组CoAt读片器；新公开代码有四组。另一方面，部分高分依赖未公开权重，不能根据标题承诺复现。

保留现有0.942锚点及所有实验资产；正在执行的6F24不取消，但不再自动延长训练或启动Top-half/新分辨率网格。此决策取代Stage6F报告中“接着做Top-half”的短期优先级，并非删除研究主线。

## 本轮社区代码和资产核查

| 候选 | 实际检查结果 | 决策 |
|---|---|---|
| [D4-lite](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-lite) | 附加分支需要私有eff6/ens14资产，元数据有空白依赖 | 不把其榜分当成公开部分可复现分数 |
| [S75-w50](https://www.kaggle.com/code/aastikrajan15/knee-s75-w50) | 存在私有成员和空白数据依赖 | 不能完整复现；不盲用其融合系数 |
| [public0946](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946) | 作者正文明确更正为0.943；0.946来自父本私有ConvNeXt集成 | 精确复现公开四成员部分，自己提交核实 |
| [DINOsaur V5](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5) | 主要是同一四成员框架将外层权重0.60改为0.70；挂载DINO-base不等于实际新增模型 | 不当作全新独立成员；本轮先不扫四种权重 |
| [ConvNeXt M448 fold0](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-cnx-m448-f0-public) | 公开完整推理源码、checkpoint与SHA清单；ConvNeXt-small DINOv3、448有序切片拼图、层级槽位池化 | 固定10%全局权重的独立增量候选 |
| [EfficientNet B3五折](https://www.kaggle.com/datasets/prvsiyan/rsna-knee-b3-v47-public-deployment) | 资产公开；其自身审计Gold约0.779且标签对照有差异，旧父本也弱于当前锚点 | 不优先，不能因架构不同就纳入 |

这些是下载的源码/元数据和数据集文件清单核查，不是对所有作者榜分的独立验证。ConvNeXt公开包只是一个fold，不等同于作者未公开五折集成。

## 已执行的两个候选

**A：四成员公开升级。** 保留上游23个推理代码单元不变，增加来源哈希与运行收据。CoAt由resgated+D4升级为resgated+D4+Global96+Repair-v1，并按上游配方先平均概率再排序；其他上游差异也原样保留，所以这不是只改变一个参数的消融。所有依赖公开，Repair-v1已确认位于D4数据集的子目录。

- 构建：`python experiments/sprint0930/build_public943.py`
- Notebook：[rsna-sprint-public-fourway](https://www.kaggle.com/code/easoncyy/rsna-sprint-public-fourway)，v1。
- 可见运行已完成：251.93秒、3例；20个DINO成员、5个A5折、4个Raptor视图、4/4 CoAt；未记录失败/缺失/回退事件。
- 3例只证明程序可运行，不证明泛化成绩。已提交隐藏评分，提交号 **56700487**；此时尚无新榜分。回执：`experiments/sprint0930/fourway_submission_v1.json`。

**B：A + 10%公开ConvNeXt。** 保留A完整配方，独立成员先在子进程推理并释放GPU，然后运行A。根据ID严格对齐，`0.9*rank(A)+0.1*rank(ConvNeXt)`，所有类别相同权重。checkpoint和全部运行源码按公开清单验SHA；推理禁网；保留A原预测；新成员失败不冒充成功融合。

- 构建：`python experiments/sprint0930/build_cnx10.py`
- Notebook：[rsna-sprint-cnx10](https://www.kaggle.com/code/easoncyy/rsna-sprint-cnx10)，v1可见运行COMPLETE。原始输出`results/sprint_cnx10_20260930/sprint_cnx10_receipt.json`核实四成员齐全、无父本回退、ConvNeXt公开checkpoint SHA正确，生成的最终提交哈希为`b171ff0f0d7e9f3f56b82b628b2187361438204f1c86946634f042b5f38a7b19`。已提交隐藏评分 **56706632**，现为PENDING。
- 10%是评分前固定的保守探针，不是Gold上调优出的“最优权重”。不直接沿用资产中的50%，避免过度依赖一个未验证单折模型。
- 单元测试：`python experiments/sprint0930/test_rank_blend.py`；覆盖打乱ID、重复/缺失ID、非有限值、越界值、错列、并列排序与权重端点。

## 接下来如何决策

先拿到A实分，再以A作为B的唯一对照。A低于0.942则不晋升；B高于A才有该成员净增益的实际证据；B持平或下降则不继续盲目加权。如果B优于A但仍不如旧锚点，也不替换旧锚点。保留0.942安全版本，不用排行榜变化推断某一病例的隐藏标签，不从三例公开样本推断融合优劣。

若出现可重复的成员净增益，下一轮预算用于同一成员的可靠训练/更多折或预先定义的增强；若没有，回到强CoAt/DINO系的表示与训练资产上进行针对性开发，而非再训练一个低起点模型。每个新训练任务必须写清：替换/补充父模型哪部分、何时能提交、什么结果就停止。0.945是目标，不是已证明一定可达的结果。

## 9月30日晚：GPU额度用尽后的检查

- 用户报告本周Kaggle GPU时长耗尽。本次仅检查现成结果、提交已完成Notebook，没有推送新Notebook训练/推理版本。
- 两个新增提交56700487、56706632均在Kaggle评分队列中，状态PENDING，没有Public分数；不能依据上游0.943标题代替自己的结果。
- Stage6F24已完成训练：Gold58宏AUC **0.847240**；6E24为**0.846901**，差仅+0.000340。6F24伪标签留出MSE **0.040061**，6E24为**0.038062**。这是开发集指标，当前输出只有训练回执/历史和模型权重，未产生`submission.csv`，不能拿训练Notebook本身提交评分，也不值得在额度耗尽时另跑推理。
- Kaggle结果CDN在受限环境报TLS错误；使用可访问的官方结果下载通道只获取了所需的ConvNeXt运行回执，避免递归下载Notebook安装的数千个依赖文件。回执通过后才进行了代码提交。
