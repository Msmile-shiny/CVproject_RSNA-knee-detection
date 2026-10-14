# 2026-10-12：社区资产升级，直接执行完整Compact融合

## 跟进与版本修复

### 2026-10-13 19:19北京时间：隐藏评分终局及优化边界

官方API两次确认57131878为COMPLETE，Public0.954，未返回errorDescription，准确scriptVersionId357879252。相较旧最佳57012367的0.950提高0.004；新候选成为当前最佳Public记录，最终选择不变。不能据Public推断Private或奖牌，也未查询当前奖牌线。资产来源修复已获得实际提交及隐藏评分成功的证据。

已实现跟进流程优化：新增check_scoring.py，仅按实际回执读取指定提交，验证kernel和script版本后保存原始状态、时间、分数及基线差值。COMPLETE带错误不能判成功，缺状态/缺分数保持等待，NaN等非法分数拒绝；五种情况测试通过。此脚本不启动GPU、不提交Notebook、不修改最终选择。源码、清单和现有成功回执继续保留。

下一步优化优先级：先保护这套已验证四折成员与固定50/50配方，禁止继续在成功Notebook上直接覆盖试验。之前0.950父栈的8视角、224和关联修正没有提分证据，不能直接搬进新栈并宣称有效。若继续模型试验，先审计新增成员与Raptor/Compact的权重、输入及依赖重叠，排除同模型重复加权；只有具有明确不同信息来源且满足离线资产和时间预算的方案，才另建候选。不得用三个可见病例的相关系数拟合融合权重，不做大规模Public比例扫描。本轮未新增GPU、训练或未验证的融合提交，已确认提升来自当前完整社区组合，而非本地流程修复本身。

### v3完成并成功提交评分

v3可见COMPLETE，外层总耗时645.3819秒（约10分45秒），3个可见病例两成员失败均为0。准确线上版本和源码、manifest、输出哈希、ID/概率/覆盖及固定50/50排名融合核验PASS。输出submission.csv SHA仍为5c4828369e39d80a6e67ece4e784206e99323120aff975216005755f7f7053dd；仅三个可见病例，不能据此判断隐藏分数或完整隐藏运行时间。

2026-10-13T05:32:09.090Z，Kaggle已接受提交57131878，scriptVersionId357879252；submission.json保存真实回执。这次没有出现此前non-versioned datasets拒绝。首次只读对账找到该提交，但API暂未返回status、publicScore、errorDescription，记录为SUBMITTED_AWAITING_STATUS，不把缺失字段当COMPLETE或失败。后续只查询此ID，禁止重复提交。最佳57012367 Public0.950及最终选择不变。

### 2026-10-13：v2拒绝与v3资产来源修复

v2可见COMPLETE，外层耗时443.494秒，3例两成员均失败0；版本、源码、输出哈希、ID/概率及固定融合核验PASS。最终CSV SHA为5c4828369e39d80a6e67ece4e784206e99323120aff975216005755f7f7053dd，与v1相同。三个可见病例不构成提分证据。

提交再次HTTP400，正文仍为non-versioned datasets，说明仅在挂载后加/2未解决限制。核对比赛记录后诊断重试也同样拒绝，没有生成Compact提交ID或Public分数。停止相同请求重试，原v2源码、元数据、运行核验、attempt和错误归档于experiments/community1012/v2。

作者原公开数据集heliosli/rsna-knee-compact-v1/13含同名四折权重、方向分类器、state_spec及manifest，文件尺寸一致，并额外有README。实际下载该版manifest，其字节及SHA与原资产清单完全一致：1c1211e64b73f1f6433ade804816572db1a2aa5422ed9c57365184bc6a9517e6。模型文件本体仍由线上原有SHA门禁验证；不能仅凭清单宣称已在本地验证全部权重。

已启动同kernelId138276850的v3，来源改为上述/13。源码SHA1352ca5a192c2615053e8a35e1e06c940a65d7d3bcd7a72c15214c144f5a1845。构建与启动检查证明代码相对v2仅替换数据集slug，保留所有推理、50/50排名融合及严格覆盖检查；7个内嵌Python文件和Notebook语法检查通过。评分资格仍需提交时实际确认，不能以push成功替代。

改进提交器：异常时立即保存按版本区分的HTTP状态与正文，保留attempt并停止，不再需要为了查看正文重发。当前只推进v3，不增加训练、比例扫描或视角实验；保留57012367 Public0.950和最终选择。

v1可见COMPLETE，public_receipt的完整计时599.1495秒，3例Raptor/Compact均失败0，版本、源码、manifest、输出与50/50公式核验PASS。上游blend_receipt记166.7秒、Raptor45.8秒，计时范围不一致；报告使用外层599.15秒，不能把内部较短计时当整体耗时，不能据3例外推隐藏预算。

评分请求HTTP400：Submission not allowed: Your Notebook uses non-versioned datasets [/datasets/heliosli/rsna-knee-compact-r6cnxfsss] (see Dataset Settings)。经查比赛记录无对应提交后单次诊断重试确认相同错误，非模型推理失败，也没有评分回执。v1日志和attempt、verified、拒绝正文保存在experiments/community1012/v1。

修复：本地SDK支持owner/slug/version-number，使用heliosli/rsna-knee-compact-r6cnxfsss/2。通过下载第2版manifest确认SHA与v1使用清单完全相同。只改挂载元数据，Notebook源码SHA仍e91ace0c7f492f7a06cc6cdbacc538e8bf84c93c96ad00075c4bfbfda714bcae。已启动同kernelId138276850的v2；GetKernel确认currentVersionNumber=2，但返回datasetDataSources省略版本，所以最终评分资格仍需v2提交时确认，不能仅凭push成功宣称问题彻底消除。

进一步改进收取流程：提交attempt按Notebook版本分开记录，保留v1拒绝历史；v2核验通过后单次提交，出现attempt无receipt仍需对账。不会绕过Kaggle版本约束或删掉质量检查。旧NoPeer57097443已Public0.950持平，停止小实验，当前重点仅完成这项完整模型升级。

## 结论与执行

近期224固定15%持平0.950、8视角下降0.949；这些结果不支持继续扩大同源推理微调。停止新增比例/视角小实验。关闭关联图的57097443已完成Public0.950持平，停止该方向。

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
