# 2026-10-12：社区资产升级，直接执行完整Compact融合

## 结论与执行

近期224固定15%持平0.950、8视角下降0.949；这些结果不支持继续扩大同源推理微调。停止新增比例/视角小实验。关闭关联图的57097443已提交，仍待评分，只收取结果。

本轮Kaggle官方API按scoreDescending检索35个公开Notebook，前列出现多个新compact融合。选择排名首项heliosli/blend-mine-in-private-lb-luck-to-all-upvoters v2，完整复现其公开图像处理及固定50/50融合。已启动easoncyy/rsna-compact-public-ensemble v1，kernelId138276850。不是另起训练，不改变最终选择。

API列表及GetKernel元数据未返回精确Public分数字段。goodpjw2008 v9源码说明记录其CoAtNet+S6的70/30组合及加入reader后均为0.954；这是作者报告，不是本方复现。最新Helios v2采用50/50，不能把旧70/30的0.954直接当作本次分数。当前本方已确认最佳仍为57012367 Public0.950。

## 三项来源比较

| 来源 | 实际读取内容 | 判断 |
|---|---|---|
| [Helios最新融合](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters) v2 | 四折compact公开包、方向分类器、解剖定位、Raptor固定50/50融合 | 本轮执行；新增已训练成员，不重复旧reader视角实验 |
| [ConvNeXt原作者方案](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954) v9 | 作者报告CoAtNet+S6 70/30为0.954，再加原三折reader显示仍0.954 | 佐证主要新资产是S6；暂不额外运行reader堆叠 |
| [velci kcd25 20](https://www.kaggle.com/code/bliverrigpdoge/velci-kcd25-20) v1 | 基础融合加蒸馏成员；元数据含空dataset源 | 资产依赖不完整，不作为限时首选 |

搜索引擎对最新Notebook索引不足，甚至标题0.95的旧DINO页实际显示0.809；因此不按标题选择。上述比较以下载源码、官方API元数据和数据集文件列表为依据，来源归档于experiments/community1012/upstream。

## 为什么可能有效

新compact是ConvNeXt-small四折、训练16epochs、每次176个窗口，推理384输入/352裁剪、3.5mm物理切片间隔，使用关节定位视图v4、左右方向分类器及状态描述。直白地说：先判断膝关节在哪里和朝哪边，再看对应的小区域。它与Raptor的输入组织、模型及报告监督配方不同，具备互补可能。

这与重复seed或只给旧模型增加翻转不同。仍不能从架构不同推断一定提分；四折及标签训练说明为作者manifest记录，并非本方重训验证。作者gold_in_training=0也不保证Gold58从未参与选择。

## 复现与检查

数据集heliosli/rsna-knee-compact-r6cnxfsss目前公开7文件，约4×200MB四折权重、85MB方向分类器、state_spec和manifest。bundle_id为209a32c8b50a6b1c；本地manifest SHA1c1211e64b73f1f6433ade804816572db1a2aa5422ed9c57365184bc6a9517e6。没有下载大权重到Git，直接挂载公开资产。源代码包、SWA数据集均沿用作者挂载，保留许可证及OAI致谢。

构建器逐字保持上游图像推理和融合代码，所有Notebook代码及7个内嵌Python文件语法通过；新增前后检查：manifest固定、每条预测覆盖记录、输出哈希、两成员均失败为0、各成员病例失败不超过3%、8.8小时总预算。作者原compact容错允许最多50%未评分，本方收紧至3%，未用静默退回Raptor冒充完整成功。仍decode_coverage_verified=False，病例有预测不代表每张DICOM解码成功。

线上完成后核对准确版本/代码、manifest、所有输出哈希、ID、概率和对成功病例各自排名后的50/50公式；仅通过一次性提交。attempt无receipt先查竞赛记录。用固定公开配方直接检验新模型组合，不扫描融合比例。分数高于0.950列候选；相同或下降则记录真实结论，不能承诺Private或奖牌。

入口：python experiments/community1012/run.py collect --submit。研究不依赖新伪标签、Gold58拟合或重新训练。待实际本方评分后才决定是否用新方案作基线。旧0.950及最终选择保持。
