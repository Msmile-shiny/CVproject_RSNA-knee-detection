# Stage 6E全量预处理暂停与续跑（2026-09-28）

## 已核实状态

[6E首轮完整任务](https://www.kaggle.com/code/easoncyy/rsna-stage6e-coverage-pilot) v1在Kaggle显示COMPLETE，但`run_receipt.json`状态为`PAUSED_PREPROCESSING`，含义是代码按预设480分钟预算正常退出，**不是完成12轮训练**。输出没有`history.json`、`last.pt`、Gold或留出预测，因此没有本轮AUC或榜分可分析。开头的debugger提示和结束的文档转换警告不是暂停原因。

启动磁盘预检记录可用20,940,476,416字节，基于16例估计全量缓存17,960,120,884字节；预检通过。通过Kaggle输出目录API逐页只列文件名、不下载DICOM/NPZ，数得3,793份`pixel_cache/*.npz`，目标为4,407例，还剩614例。缓存文件名由Study UID哈希生成，计数说明进度，不意味着已训练任何模型。按本轮实际进度，继续原实验比重新生成全部图像更合算，但剩余耗时不能由线性外推保证。

## 已执行续跑

`notebooks/build_stage6_coverage.py --resume-after-paused`以本次收据SHA锁定来源；生成的[续跑任务](https://www.kaggle.com/code/easoncyy/rsna-stage6e-coverage-resume) v1挂载首轮输出、官方ImageNet权重和原v5标签，配置仍为224px、每槽8位置、mean聚合、fold0、seed42、12轮，训练实现SHA未变。运行时逐项复核旧收据的标签、权重、元数据和模型配置，再复用已有缓存；没有先前训练检查点，因为首轮尚未进入训练。最后核查Kaggle状态RUNNING。

本续跑代码针对**一次**已暂停输出。若再次在预处理阶段暂停，新旧两份输出会各含一部分像素缓存，不能只挂最近一次输出就宣称完整复用；需先建立两份缓存的联合挂载/索引并核对UID数量。若训练阶段暂停，则优先从最后完整epoch检查点恢复，不使用不完整的epoch。

## 结果关口

仅当收据为`PILOT_COMPLETE`、`completed_epochs=12`、训练/留出/Gold数量为3497/852/58且输出逐例预测完整时，运行`scripts/compare_stage6_input.py`与原6A 12轮比较。同一旧标签上的852例MSE和58 Gold逐类AUC用于开发诊断；Gold已被历史选择使用，不等于独立验证，也不据此搜索逐类融合权重。若覆盖收益不足，回到已通过冒烟的6F 288px输入实验，先解决其明显较高的预处理成本。
