# 公开 0.941 配方复现：执行记录

2026-09-21 更新：基线实际 Public **0.941**；Native64 提交 56333692 已完成，Public **0.940**。保留基线，不采用加密配置。下文运行过程中的 PENDING 为历史状态，以结构化提交回执及本更新为准。

Kaggle Notebook：https://www.kaggle.com/code/easoncyy/rsna-anchor941-audited

## 已完成

- 下载并锁定 `jiweiliu/rsna-knee-fast-2xt4-inference` v5 源码。
- 源码 SHA-256：`b02169d71b873985d5f4272beb191dbce73377150605ed111f87830c4d571c86`。
- 核查 12 个数据集与 2 个 Notebook 输出的文件清单，均可访问；数据集当前版本记录于 `dependency_audit.json`。
- 保留 upstream `probe22` 配方：模型、图像处理和融合运算不变。
- 对 DINO 缺失成员、前沿预测切换失败、A5 病例预处理失败、Raptor 病例推理失败和残差 CoAtNet 失败增加中止检查。
- 保存 A5、RadImageNet 中间预测与最终运行回执。
- 全部代码单元语法检查通过，5 个最终输出检查测试通过；服务端回读源码与本地所有单元一致。
- 已通过 API 上传 v1，禁网双 T4 运行完成，依赖自动挂载。
- 运行前 39 个关键资产检查通过；20 个 DINO 成员完整，全部必需分支完成。
- 最终回执 `complete=true`，3 个可见病例，核心流程耗时 204.67 秒，残差 CoAtNet `fallback_studies=0`。
- 已提交比赛评分，提交编号 `56290048`，提交时间 `2026-09-17 00:33:31 UTC`（北京时间 08:33:31）。2026-09-18 查询确认状态为 `COMPLETE`，实际 public 为 **0.941**。

本账号已实际复现 public 0.941。两个引用的历史版本无法通过 API 取回，因此采用可核实的 Fast v5；本次实际评分验证了当前复现产物的公榜表现。

## 下一实验：Native64

Notebook：https://www.kaggle.com/code/easoncyy/rsna-anchor941-native64

代码核查发现原有 44 张缓存切片已经遍历了 42 个中心窗口；单独提高 `k_eval` 只会重复窗口。新实验只改变 `native384-v8` 分支的采样密度：44 张到 64 张，随后遍历所有 62 个中心窗口。采样范围仍为 6%–94%，其余三个分支、模型权重、分辨率、裁剪、标签与融合比例不变。

构建检查已证明其余分支配置相同，新配置在完整有效缓存下得到 62 个不同窗口。增加采样点不保证得到同样数量的不同原始 DICOM：短序列仍可能重复采样，因此本次仅检验该密度设置的实际净收益。

已生成并上传 v1，禁网运行验收通过，核心流程 171.04 秒。已提交公榜对照，提交编号 `56333692`，当前 `PENDING`。如果下降，保留 0.941 基线；如果显示分数相同，记为“未观察到显示精度以上的收益”，不凭 3 个可见病例选择权重。

## 文件

- `notebook/anchor941.ipynb`：可导入 Kaggle 的完整 Notebook。
- `notebook/kernel-metadata.json`：上传、加速器、禁网和挂载配置。
- `fast-locked/source.ipynb`、`source_record.json`：未修改来源及版本信息。
- `dependency_audit.json`：依赖访问和文件清单。
- `build_receipt.json`：改动单元及源码校验值。
- `test_anchor.py`：病例错位、非法值和不完整分支的检查。

## 运行验收

必须产生 `anchor941_run_receipt.json`，其中 `complete=true`，残差 CoAtNet 三模型全部成功且 `fallback_studies=0`。仅有 `submission.csv` 不算通过。

可见 test 只有少量病例。它的正常输出用于检验通路，不用于计算 AUC，也不能证明完整隐藏集的运行时长。成功后已提交同一版本进行实际比赛评分，保留原 0.936 提交。

第二阶段的窗口覆盖实验以这次实际评分为起点。若分数未达到预期，先查版本、分支和预处理，不同时更改训练标签与模型。

## 版本限制

Notebook 源码已按哈希锁定。数据集版本号已记录，运行前检查关键文件名及字节数，并保留上游已有的哈希校验。Kaggle 上传清单仍按公开资产名称挂载；这不是所有外部资产永久不可变的保证。Notebook 输出依赖的精确历史版本未由文件清单 API 提供。
