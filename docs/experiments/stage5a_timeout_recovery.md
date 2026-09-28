# Stage 5A 时间预算中断与恢复（2026-09-14）

依据：用户粘贴的 Kaggle 运行日志。未取得该运行的实际输出文件，文件是否保留需查看原运行 Output。

## 根因

在25506.6秒时抛出 `RuntimeError: Total preparation budget exceeded; preserve diagnostics for a new run`。`begun` 在约302秒启动，因此经过约25204秒，超过代码中的 `(300 + 120) * 60 = 25200` 秒。是本项目主动中断，不是日志显示的 OOM、NaN 或 Kaggle 强制超时。开头 debugger 提示和结尾转换文档提示均不是根因。

原实现有两个工程缺陷：已完成的控制组直到精看全部完成才导出指标；逐病例缓存虽已落盘，却没有读取旧缓存继续执行的入口。此前“preserve diagnostics”错误提示并没有实现完整恢复流程。

## 已完成工作

- 4410例粗特征已提取完毕，约235分钟；4410=4349训练+58 Gold+3可见test。
- mean完成20轮，训练 loss 0.50284→0.41665。
- coarse完成20轮，训练 loss 0.49440→0.39716。
- coarse_continue完成额外20轮，训练 loss 0.39523→0.37190。
- 精特征日志最后一次报告3840/4410；实际已写出数量可能多于3840。剩余约570例，按本次速度估算还需约27分钟编码/读取，另加校验、元数据扫描和训练导出时间。
- fine head尚未训练，Gold评估尚未执行。训练loss下降不是泛化提升证据，不能从本日志预测Public成绩。
- SAG_FLUID_NOFS匹配0例是输入覆盖观察项；它不是这次异常的触发原因，恢复阶段保持原实验定义。

## 修复内容

1. 增加显式 `resume_input`，从该路径与当前 working目录恢复。要求 manifest 中 checkpoint SHA、标签 SHA、特征和训练参数相符，防止拿错实验缓存。
2. 对每例缓存检查形状、数据类型、有限值、mask、slot、scale和精看坐标。坏缓存重新计算；精看坐标必须与当前 coarse head选择一致。
3. 恢复已经完整训练的三个head，严格加载参数；新增head记录训练UID摘要。旧版head通过原manifest与smoke设置检查，仍假设挂载的是同一竞赛版本。
4. coarse、mean与coarse_continue各自完成后立即导出Gold、test预测和AUC，精看是否完成不再阻止控制组评估。
5. 选片位置、进度和错误记录每100例保存。单病例NPZ先写临时文件再替换，减少中断文件被误用的风险。
6. 总session默认480分钟，预留20分钟；到暂停线正常返回并标记 `status=paused, completed=false`。这表示需要续跑，不表示完成实验或可提交成绩。

## 现在怎么跑

使用修复后的 `notebooks/kaggle_train_stage5a_dense_mil.ipynb`。

若原失败运行的 Output 仍可访问，保留完整输出，尤其：

```text
stage5a_manifest.json
stage5a_features/coarse/*.npz
stage5a_features/fine/*.npz
stage5a_mean.pt
stage5a_coarse.pt
stage5a_coarse_continue.pt
stage5a_*_history.csv
stage5a_decode_failures.json
```

在新会话挂载该输出（或打包成Dataset），将 `S5['resume_input']` 填为含 manifest 的目录。例如 `/kaggle/input/datasets/easoncyy/<实际数据集名>`。不要求旧输出有 selected_slices.json；精看位置可以由已保存coarse head与粗特征重建。

同时继续挂载竞赛、v5-bestmodel和rsna-knee-v5-labels。Notebook已填入用户日志中验证通过的两个资产路径。保持 smoke_studies=0、epochs=20、其他模型参数不变。日志应显示 `Restored completed head` 和 reused 计数；精看仅补缺失/损坏/坐标不匹配的病例。

若旧输出未保留，不能承诺救回此前7小时；可使用修复版从头运行，仍会提前输出控制组并支持以后的暂停续跑。不要将部分test预测拼成最终提交。

## 验证

8项本地测试通过，包括真实DINOv2两种分辨率前向、特征训练、UID顺序、掩码、缓存恢复与损坏拒绝、来源不一致拒绝，以及模拟精看阶段暂停后成功续跑并保留控制组导出。修复Notebook已重新生成并编译。尚未在Kaggle真实GPU/旧缓存上验证恢复耗时或分数。
