# 固定三折ConvNeXt增量

公开模型与代码作者：[goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944)，[权重Dataset](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader)，Dataset API许可Apache2.0。保留原始Notebook快照，构建器提取其中的三份推理源码。作者报告0.944尚不等于我方复现成绩。

构建器保持0.943父本所有代码不变。reader先在独立进程运行，固定全12类30%排名融合，三个公开fold都必须完成；原文的silent fallback改为停止并记录错误。checkpoint使用weights_only=True和strict load，DICOM解码失败报错，满批4病例/6槽位显存检查。

v1实际slug为`easoncyy/rsna-sprint-three-fold-convnext-30`，不同于metadata中构建时的cnx3fold30。实际启动ref、源SHA、kernelId均保存在launch_receipt。已COMPLETE，输出门禁通过，提交56856316隐藏评分中；禁止重复push或提交。可见内部总耗时622.222秒，reader134.401秒、满批峰值4.17GB。

重建：`python experiments/convnext1005/build.py`。收取已启动版本：`python experiments/finish_sprint1005.py cnx3 --submit`，已有提交回执会阻止重发。新评分结论见docs/experiments/sprint1005_execution.md。
