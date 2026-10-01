# Stage 6F 12轮结果与固定24轮续训（2026-09-30）

## 结论与任务状态

288px/4位置的6F pilot已完成全部预处理和12轮训练。Kaggle为COMPLETE，内部收据PILOT_COMPLETE，CUDA且骨干梯度非零；训练/留出/Gold数量3497/852/58。并非预处理暂停。总耗时21699.8秒（约6.03小时），单epoch约506秒。0.942仍是现有Public基准，本实验未提交评分。

同折、同标签、同初始化及优化设置、同12轮对照：

| 指标 | 6A：224px/4位置 | 6F：288px/4位置 |
|---|---:|---:|
| Gold58宏AUC | 0.815587 | 0.820621 |
| 852例伪标签留出MSE | 0.040327 | 0.039200 |
| 第12轮训练损失 | 0.389263 | 0.392329 |

Gold增量+0.005034；留出MSE相对降低约2.8%。配对病例bootstrap描述性95%区间[-0.019325, 0.029771]跨零，收益不确定。相比之下，6E的224px/8位置12轮Gold为0.834748，但它的MSE为0.040046，因此覆盖与分辨率也不是所有指标都同向排序。Gold小样本已用于开发，留出目标是伪标签，不能用两项数值保证Public表现。

## 类别与训练轨迹

8类Gold AUC上升、4类下降。较明显上升为MCL+0.043084、Effusion+0.022360、外侧半月板+0.021118、ACL+0.019608、骨折+0.019444。内侧半月板-0.021635、外侧OA-0.032882；没有证据支持“提高分辨率自然解决细小结构错误”。

训练损失从0.5349持续降至0.3923；留出MSE第8轮0.037627、第12轮0.039200。两组轨迹表明继续训练存在泛化回退风险，也可能随后改善；需要固定终点而非Gold选轮次。当前只保留最终检查点，不能把第8轮误差当成已保存的可用模型结果。

比较脚本已检查相同训练代码SHA、标签/权重/元数据哈希、病例集合、列顺序和逐类AUC。复算结果在`experiments/stage6/resolution/comparison.json`；运行收据和history归档在`experiments/stage6/resolution/epoch12/`。原始输出位于`results/stage6f_pilot_20260930/`。

## 已执行下一步

已推送 [6F resolution continuation v1](https://www.kaggle.com/code/easoncyy/rsna-stage6f-resolution-continuation)，本次末次查询RUNNING。由`notebooks/build_stage6f_continuation.py`生成：SHA锁定12轮收据、要求完整4407例缓存、校验checkpoint epoch12与配置，再恢复模型/优化器/随机状态继续到固定epoch24，预算240分钟。训练实现SHA仍与6A/6B/6E一致，本地top_half候选不参与。

决定追加这次训练，是因为汇总指标同向小幅改善且完整缓存已存在，可以用约两小时量级的额外训练完成与6B24的公平比较；不再重复约六小时的完整首轮流程。实际时长受Kaggle I/O影响，RUNNING并不保证已通过所有启动检查。

```powershell
python scripts/compare_stage6_input.py ../stage6-continuation-output <6F24输出目录> data/metadata/train.csv --dimension size --epochs 24 --output experiments/stage6/resolution/comparison24.json
```

评估后不机械延长到36/48轮。若288px收益依然有限，优先保留成本更低且已有覆盖收益的224px/8位置主线，进入Top-half聚合或几何增强的单变量对照；若更清晰输入出现持续优势，再测试288px+8位置组合。当前不直接上五折、不按Gold逐类挑融合权重。
