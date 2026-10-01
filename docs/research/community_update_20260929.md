# 6E后的社区与论文检索（2026-09-29）

本次检查时6E的24轮续训仍RUNNING。12轮实验Gold宏AUC改善，但内侧半月板下降，支持继续研究采样及聚合；并不能证明某种病理机制或保证Public提高。

## 新增证据

- [社区受控负结果](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737597)：作者自报中心相邻切片优于等距覆盖，EMA、Mixup和延长训练收益很小，ASL明显下降。其模型、标签、折和数据处理不同，数值不能直接移植。本项目已经使用相邻三片构成输入，但三片组中心仍等距覆盖；“增加组数”同时改变组中心位置，不能解释为纯粹更多信息。
- [单模型瓶颈讨论的新回复](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148)：有参赛者自报224px ResNet、24片、增强和五折到0.94；另有人从384降到288影响很小。均为社区自报，无完整复现凭据。我们的训练增强只有亮度缩放，适度几何增强仍是尚未测试的变量；需要同一组三片共享空间变换，不能逐片随机扭曲邻接关系。
- [主办方影像定义](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)说明半月板判断涉及多张图像上的信号或形态异常。因此三片上下文、局部信息保留有依据，但不能把“两个高分窗口”当成符合诊断标准，因为窗口可能重叠，且没有定位标注。
- [MRNet原论文](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699)是膝MRI切片编码和检查级预测的基线参考；不是本竞赛分数证据。
- [MIMS CNN，MICCAI 2019](https://arxiv.org/abs/1907.02413)针对小ROI、缺少局部标注，使用top-k聚合多尺度特征。借鉴的是局部强证据聚合思想，下面的窗口logit聚合并非论文完整复现，也没有已验证的RSNA收益。

## 已实现候选与限制

`notebooks/stage6_resnet_core.py`新增`pooling='top_half'`：每个病例、每个类别选最高的一半有效窗口分数再平均，比例固定为1/2，缺失窗口不参与。保留原网络参数结构；dropout对同病例同类别的所有窗口共享特征掩码。默认mean和既有attention路径不变。当前已推送的6E Notebook嵌入的是原代码，不受本地改动影响。

直观上，若病变只出现在少量切片，所有切片平均可能冲淡证据；top-half可减轻这一点。但它也可能放大伪影或槽位偏置、忽略弥漫性病变。均值训练出来的模型直接改推理聚合还会产生分布偏移，因此不直接改0.942提交、不用Gold扫描比例。只作为后续从相同ImageNet初始化训练的单变量候选。

`scripts/test_stage6_top_half.py`检查有效数量不足、奇数取整、缺失窗口屏蔽、空病例拒绝、梯度、单窗口等价和检查点兼容。完整训练Notebook尚未生成或启动；后续builder须嵌入新core并记录新实现SHA，不能把旧SHA续跑验证直接用于这个新模型。

## 实验顺序

先完成6E24与6B24的同轮数比较。已有288px单变量对照仍是候选；top-half作为解释局部类别退步的低额外参数方案，不能仅凭Gold的一类变化插队为确定最优路线。若启动，固定同一输入、标签、折、12轮预算，分别训练mean/top-half，比较852例伪标签误差、全部12类Gold以及与父模型的互补性。随后才考虑独立的几何增强对照。当前未启动第二个GPU训练。
