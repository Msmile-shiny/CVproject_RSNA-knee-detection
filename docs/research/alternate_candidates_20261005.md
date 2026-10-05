# 2026-10-05 第二条公开增量路径审计

## 决策

优先候选是 [goodpjw2008 的独立 2.5D ConvNeXt reader](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader)，不是再次扫描先前 M448 fold0 的融合权重。完整公开 [推理 Notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944) 已成功下载。只读审计，无 GPU 启动、无提交、无父本修改。

作者正文自述：独立三折模型 Public 0.929；以固定 30% 排名融合加入 0.943 社区栈后，2026-10-04 私有运行达到 0.944。**这是作者自述，不是我方复现结果，也不是标题分数的独立核验。** 不保证当前父本复现同一收益。其架构、训练、预处理均明确公开，相比无完整私有权重的高分壳，具备实际接入条件。

## API 与代码核验

- Dataset v1，更新时间 2026-10-05T00:09:57.943Z；API licenseName 为 Apache 2.0，竞赛数据约束仍然适用。
- `cnxt_v0_fold0.pt`、`fold1.pt`、`fold2.pt`，每个 123,035,271 bytes。仅核实远端清单，未下载/加载权重，故未验证 checkpoint 内容或哈希。
- `infer.py`、`knee.py`、`preprocess.py`、`train.py`、`make_labels.py`、`oof.py` 均已下载并 AST 语法检查通过。
- 附带 pylibjpeg 及 libjpeg/openjpeg 的 Python 3.11/3.12 Linux wheel。实际 timm wheel 来自父栈资产；需要离线依赖门禁。
- 源码保存在 `results/research1005/alternate/goodpjw2008/`，公开 Notebook 与 metadata 在其 `notebook/` 子目录。
- Notebook Part B 为 0-based cell 33 建目录，34/35/36 写三份推理源码，37 运行及融合。Notebook 内 knee.py 比 dataset 版本多 img_size 兼容分支；不要混用两套源码，优先固定完整 Notebook Part B。

## 输入、模型与成本

- 六槽位：Sagittal/Coronal/Axial × 脂肪抑制/非抑制，每槽取切片最多的 series。
- DICOM 按物理位置排序、方向规范化，0.4mm/像素，384px 缓存对应153.6mm；最多64层；百分位1/99.5归一化。
- 每序列12个覆盖4%至96%的窗口，各窗口3张相邻切片；实际视野再乘0.92，模型输入256px（Notebook自述与训练默认；checkpoint args 尚未加载核实）。
- ConvNeXt-Tiny，384维 token，序列槽位和切片位置编码，2层 study Transformer，12类各自 attention pooling。最多72个窗口/病例/模型，三折最多216次2D编码；不是完整3D训练。
- 训练源码为14 epochs、BCE、EMA，4份公开报告标签平均，按报告文本分组5折；只发布其中3折。Gold58排除训练，但仍是开发诊断，不能作私有榜保证。
- 推理为单GPU FP16，原 batch_size=4。作者自述五checkpoint单独提交到评分少于90分钟，完整栈6.5至8小时；均不等同净推理计时，需本次实测，完整提交限制9小时。

## 必须补的安全门禁

1. 保留当前父本56700487，不复制其余整套社区栈覆盖父本；仅接入独立 reader，固定30%全球排名融合做一次预注册实验。
2. 固定3个明确checkpoint，strict state load，记录哈希/args；避免 glob 不完整时默默只跑一折。优先使用 `weights_only=True` 的安全加载；若不兼容，先核实具体类型，不直接执行任意pickle。
3. 源码存在 `except Exception: continue` 跳过 DICOM/series，原融合存在静默父本回退，pip安装 `check=False`。须记录解码失败、每病例有效槽位、模型完成数和回退；未实际加入reader时，不得当新候选提交。
4. 每病例至少有有效输入，输出行ID与父本严格一致、12列有限值；保存 reader 原始预测、父本回放、最终融合及receipt。模型/像素流程不作额外改变。
5. GPU可见测试证明可加载和增量耗时，不证明AUC；门禁通过才提交隐藏评分。不能以3病例相关性判断互补或调融合比例。
6. 不直接把 meniscus 和 ConvNeXt 同时叠在同一候选；分别评价，避免结果无法归因和总推理时间失控。

## 淘汰/暂缓：two-target student v2

[prvsiyan/rsna-knee-two-target-mri-student-v2](https://www.kaggle.com/datasets/prvsiyan/rsna-knee-two-target-mri-student-v2) 权重公开2,751,375 bytes，训练源码和receipt公开；README标明competition数据派生、license other。

仅684,538参数，预测 Lateral Meniscus 与 Lateral OA；三平面各中心16层，256缓存内部128输入。教师为0.5 old ridge rank + 0.5 routed ridge rank。3874训练/472验证主要拟合教师MSE，并非有独立真实标签的AUC验证。

receipt报告旧v52基线Gold宏AUC .85581→.86330，但bootstrap增益区间[-.00124,.01704]跨0。外侧半月板单模型Gold .7764、外侧OA .7505。这不能证明改善当前.943父本；任务与现有半月板候选重叠且证据更弱，因此不优先占用本轮GPU。

## 本次边界

仅源码/元数据下载、只读审计和报告。没有训练，没有评测公开checkpoint，没有声称已提分，没有修改最终提交。下一步应由主线程选择并行GPU预算，不无限扩张候选。
