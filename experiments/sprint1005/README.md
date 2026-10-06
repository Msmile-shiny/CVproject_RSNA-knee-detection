# 固定半月板增量

2026-10-06 20:11终局：提交56866776 COMPLETE，Public0.943与父本持平。停止此固定10%增量，保留父本及最终选择；scoring_receipt.json为结果依据。

2026-10-06最新状态：v2可见COMPLETE，全部门禁通过，总耗时1266.874秒。已提交56866776（scriptVersionId355502633）等待隐藏评分，下文启动说明为历史记录。

模型和原始预处理作者：[renta0426](https://www.kaggle.com/code/renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid)，[公开权重与runtime](https://www.kaggle.com/datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1)。author_preprocess_source.py是公开Notebook的原始DINO代码单元；构建仅移除末尾启动20成员预测的调用，在独立子进程使用原始预处理和hash绑定runtime。

两类半月板固定90%父本排名+10%专项排名，其他十列保留CSV token。CPU v2预处理、权重严格加载与合成前向已PASS。完整GPU v1在CUDA统计初始化前失败，未预测、未提交；失败版本保存在failed-v1。修复allocator初始化后v2已启动，等待满批显存与完整运行，不弱化门禁、不扫比例。

重建：`python experiments/sprint1005/build.py`。收取已启动版本：`python experiments/finish_sprint1005.py meniscus --submit`。当前确切版本以launch_receipt.json为准，不能重复启动。
