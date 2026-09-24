# 项目复盘与下一阶段（2026-09-24）

## 核实结果

Kaggle 提交 API 已返回 submission **56454576 / Public 0.942**。这是真正经过隐藏数据重跑的成绩，来源是 `maverickss26` 公开 Speedy/D4 配方的复现；32 个上游推理代码单元在我们的 hidden-robust Notebook 中逐字节一致。我们的贡献是依赖整理、可重建 Notebook、运行回执和失败诊断，不能把这些预训练模型或 0.942 分数称为自研。

| 路线 | Public / 本地结果 | 结论 |
|---|---:|---|
| Anchor941 社区复现 | 0.941 | 历史可靠锚点 |
| Anchor942 Speedy/D4 社区复现 | **0.942** | 当前可靠锚点 |
| Anchor942 严格审计版 | 隐藏重跑异常 | 三例可见测试通过不能保证隐藏运行；不再作为提交父版本 |
| Native64 单分支采样加密 | 0.940 | 未优于 0.941，停止 |
| Stage 5A 自训成员 + 父模型 | Gold58 父模型 0.954846，加入 2% 后 0.954967 | 同一开发集上的差异极小且不确定；没有榜分证实 |
| Stage 3A / 3C / 3D | 0.818 / 0.807 / 0.880 | 标签或可信排序的本地收益没有转成榜分 |
| OrthoFoundation 冻结探针 / Phase 4A2 | Gold58 0.7906 / Public 0.808 | 现有实现不具备融合资格 |

**验证边界：**只有 58 个 Gold 病例，且已参与多轮方案选择；现有父模型的部分 checkpoint 也用它们选择。Gold58 是诊断工具，不能作为当前融合的独立泛化证明。Public 0.942 与论文报告的 AUC 属于不同数据集、标签定义和评测协议，不能横向比较。

## 社区现况与证据强弱

1. 可执行公开配方：我们已实际复现 Speedy/D4 的 0.942。它同时改动 A5 权重、RadImageNet 权重、Raptor 窗口数并加入 D4 CoAtNet，所以 0.001 的提升不能归因于任何单项。
2. 比赛讨论中有人报告单模型 0.942、TTA 0.943、五折 0.947，以及更高的多模型成绩。这些是参赛者自述，提示强单模型可行，但未提供同等级可复核的权重、训练和隐藏评分链路。[社区讨论](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
3. 同一讨论中有 392px、150mm 裁剪、随机 32 张切片的经验，也有人认为标签质量比单纯增大模型更关键。参数应当作为候选实验条件，不应照抄为结论。[社区讨论](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
4. 公开 `rad-fracture-oof` 候选与当前配方在 RadImageNet 的 Fracture 排除设置上有可隔离的差异。标题分数不足以证明增益；我们可以固定其余所有步骤，只试这一项。
5. 某些 0.943 标题写的是目标分数或加入多项启发式规则。先不依据标题改变病变关系或逐类权重。

## 论文带来的可检验方向

- [CoPAS，Nature Communications 2024](https://www.nature.com/articles/s41467-024-51888-4)：跨序列同平面注意力，用不同平面共同判断病变；在其自有五中心数据上还观察到外部中心分数下降。对本项目的启发是保留序列槽位身份、显式处理缺失序列，并评估跨平面的病例级特征融合；它没有证明在本比赛上会增加 0.01。
- [多任务半月板损伤研究，2025](https://pubmed.ncbi.nlm.nih.gov/39620311/)：分割提供解剖位置先验，结合矢状/冠状序列改善半月板判断。可测试一个聚焦半月板区域的独立成员，而不是再复制全图 DINO。
- [OrthoDiffusion，2026](https://arxiv.org/abs/2602.20752)：在 15,948 份未标注膝 MRI 上训练三个方向的 3D 表征。这支持厚切片/体积与方向特异表示的研究，但训练和资产成本高，且其八病变任务不等同本比赛十二目标。
- [OrthoFoundation，2026](https://arxiv.org/abs/2601.18250)：膝部图像自监督预训练具有迁移潜力；我们已经做过冻结探针并得到负结果，所以继续使用它必须先解决输入表示、微调策略和训练标签的差距。

## 执行顺序与进入门槛

**现在：Fracture 单变量对照。** 以已得 0.942 的 hidden-robust Notebook 为父本，只把 `_RAD_EXCLUDE = ("Baker's", 'Fracture')` 改为 `("Baker's",)`。这表示允许 RadImageNet 分支影响“骨折”列，其他 11 类和其他融合参数不动。已验证构建差异，Kaggle 可见运行完成且无降级；隐藏评分提交为 **56514045**，目前待评分。结果高于 0.942 才晋升；持平只算候选，不宣称有效；下降则停止。这是快速且可解释的试验，但它不一定带来可见的三位小数增益。

**随后：独立的局部解剖成员。** 先核对伪标签的病例 ID、类别和来源，再按病例固定五折，使同一病例的所有序列只出现在一个折中。第一版聚焦冠状/矢状半月板局部区域，保留序列存在掩码，输出每例两个半月板病变的 OOF 预测；不在 Gold58 上选择模型或逐类权重。先检查独立成员自己的 OOF 可靠性和错误病例，再与锚点的病例级排序做相关性诊断。若无法获得无泄漏的锚点病例级预测，就不能把 Gold58 的微小融合收益当作入选证据；需要预先固定一个小的全局融合权重并用真正的外部提交检验。若输入区域、标签和 checkpoint 选择无法审计，也不加入融合。

**较长期：**考虑跨序列注意力或方向专用 3D 特征，与局部模型做对照。先检查可用权重、协议和算力，再决定是否训练；不盲目追加 DINO seed、单纯加密切片或在 58 例上搜索十二个权重。

## 本轮执行与复现

- 构建器：`experiments/anchor942/build_fracture_ablation.py`
- 父本：`experiments/anchor942/hidden-robust-notebook/anchor942-hidden-robust.ipynb`
- 构建回执：`experiments/anchor942/fracture_ablation_build_receipt.json`
- 构建命令：`python experiments/anchor942/build_fracture_ablation.py`
- Kaggle Notebook：`easoncyy/rsna-anchor942-fracture-ablation` version 1；可见运行完成，提交 `56514045` 待评分。
- 代码竞赛提交必须使用 Kaggle Notebook 的版本号，不能直接上传本地 `submission.csv`。

此报告记录的是研究优先级和实验设计。后续榜分与运行状态须另补回执。
