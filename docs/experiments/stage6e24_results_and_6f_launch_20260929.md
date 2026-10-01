# Stage 6E 24轮结果与6F启动（2026-09-29）

## 结论

6E续训完成，8位置覆盖的开发集收益在24轮仍保持。相同24轮下Gold宏AUC提高0.018934，852例伪标签留出MSE降低约10.0%。这是继续研究输入的依据，但尚无新的Public成绩；0.942仍为已复现的社区父模型成绩。

## 完整性与复算

Kaggle任务`easoncyy/rsna-stage6e-coverage-continuation`为COMPLETE，收据PILOT_COMPLETE、completed_epochs=24。日志确认缓存并集为614+3793=4407例，从epoch12恢复；训练3497、留出852、Gold58，CUDA梯度正常。前12轮history保留，新增13至24轮。运行收据报告续训主体8743.3秒，Notebook日志约9351秒结束；两个计时起点不同，后者包括检查点预检等启动步骤。

`scripts/compare_stage6_input.py --epochs 24`已成功校验相同标签/权重/元数据/实现SHA、病例集合和列顺序，并从预测重算Gold逐类AUC。数值结果保存在`experiments/stage6/coverage/comparison24.json`；收据和history保存在`experiments/stage6/coverage/epoch24/`。

## 结果

| 模型 | Gold宏AUC | 852例伪标签MSE |
|---|---:|---:|
| 6A：224px，4位置，12轮 | 0.815587 | 0.040327 |
| 6E：224px，8位置，12轮 | 0.834748 | 0.040046 |
| 6B：224px，4位置，24轮 | 0.827966 | 0.042292 |
| 6E：224px，8位置，24轮 | 0.846901 | 0.038062 |

24轮配对Gold差值的描述性95% bootstrap区间[-0.017909, 0.057444]仍跨零。两个训练时点并非独立重复，且Gold58已参与长期开发，不能把持续同向解释为统计确定。伪标签留出误差反映与教师目标的一致性，也不是独立临床真值或Public AUC。

与4位置/24轮相比：10类上升。外侧半月板+0.070807，ACL+0.052696，Baker's+0.050725，MCL+0.029478。内侧半月板仍下降0.055288（0.864183→0.808894），骨折微降0.001389。内侧半月板相对6E12轮自身提高了0.040865，说明之前退步有部分可通过训练缓解，但尚未消失。不能直接用Gold选择每类最好的模型。

6E留出MSE在epoch20为0.037172，epoch24为0.038062；存在波动，没有证据保证继续延长会提高表现。当前产物为最终24轮快照，不宣称拥有epoch20的完整模型。停止无限续训，转下一项输入对照。

## 已执行下一步

已推送 [6F resolution pilot v1](https://www.kaggle.com/code/easoncyy/rsna-stage6f-resolution-pilot)，首次查询QUEUED。使用此前通过的16例288px冒烟收据，保持4位置、mean池化、原v5标签、fold0、seed42、优化设置和12轮预算，只改变224→288px。

选择4位置是为了与6A12轮形成分辨率单变量对照，而不是直接把288px、8位置和Top-half一起加入。生成Notebook的训练实现SHA与6A完全相同；本地后来新增的top_half模块没有进入本次任务。

通过冒烟缓存大小估算磁盘需求并设置容量预检。288px需要重新构建独立缓存，480分钟预算可能先耗在预处理；如果收据是PAUSED_PREPROCESSING，它只是可恢复暂停，不等于训练完成。沿用`build_stage6_coverage.py --variant resolution --resume-after-paused ... --resume-source ...`恢复，并保留每次输出来源；多次暂停须联合全部缓存。

完成12轮后执行：

```powershell
python scripts/compare_stage6_input.py ../stage6-pilot-output <6F输出目录> data/metadata/train.csv --dimension size --epochs 12 --output experiments/stage6/resolution/comparison.json
```

若分辨率单独有收益，再讨论8位置+288px组合；若没有，使用现成224px缓存研究Top-half或几何增强，每次仅改一个因素。进入五折与父模型融合前，仍需独立成员的实际推理能力和同病例融合收益验证。
