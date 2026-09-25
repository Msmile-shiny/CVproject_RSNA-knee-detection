# Stage 6A 进展：冒烟通过，正式训练已启动

检查日期：2026-09-25（Asia/Shanghai）。

## 核实结果

- `easoncyy/rsna-stage6a-resnet34-reference` v2：Kaggle COMPLETE；产物状态SMOKE_COMPLETE，CUDA。
- 8训练、4伪标签验证、4Gold开发病例，1轮，224px、每槽4个真实三切片窗口；骨干梯度范数0.213895887，训练loss 0.650643；总耗时422.016秒，其中训练循环3.011秒。
- 这16例只验证工程路径；4例Gold出现0/1或不可计算AUC很正常，不能据此评价泛化。总耗时包含准备/元数据扫描/读取，不能按16例简单线性外推全量训练时间。
- 标签与官方权重哈希均符合预注册值；源码实现哈希 `fef584b3353c5b757caf1f0b2659d65edec8d76995d69da825368aa942bcd8e0`。
- 实际分组单位是StudyInstanceUID，不是PatientID。Gold仍是重复使用的开发集，历史教师OOF来源未独立核实。

完整收据保存在 `experiments/stage6/smoke_v2_receipt.json`。

## 已执行下一步

使用生成器的 `--pilot-after-smoke` 关卡，核对真实GPU收据及同一源码哈希后，发布 **v3**；最后查询为RUNNING。保持模型、标签、学习率、224px、4位置、batch1、fold0、seed42和mean聚合不变，仅由smoke数据改为全量固定划分，预定训练12轮，单次预算480分钟。

本次不加载smoke模型，不择优选择Gold checkpoint，不生成竞赛submission.csv，也未新增评分提交。8小时是暂停预算，不保证该次会话能完成预处理及全部12轮；到时按产物中的阶段判断续跑。

下一次检查顺序：

1. 先查看真实运行状态和run_receipt：PILOT_COMPLETE、PAUSED_PREPROCESSING、PAUSED_TRAINING或异常。
2. 如暂停，挂载本次输出并恢复，不把暂停当模型失败；如异常，修工程原因并审计受影响病例。
3. 如完成，检查12轮曲线、有效窗口覆盖、伪验证趋势、Gold逐例预测。必要时先完成约定的延长训练/单一学习率诊断，再考虑监督对照。
4. 只有成员自身训练可信，才做与0.942父模型的严格UID对齐与小权重融合分析。当前没有新模型Public成绩。

## 其他已完成实验

Fracture-only提交56514045已COMPLETE，Public **0.942**；原父模型56454576也是0.942。只能说展示精度下持平，不能声称底层预测完全相同或获得了提升；保留原父模型。

## 测试与环境边界

2026-09-24：4项CPU单元测试、合成DICOM完整训练/预测/恢复测试通过。2026-09-25：本地d2l环境导入torch._C发生DLL错误，加入其Library/bin与torch/lib搜索目录后仍失败，因此今天的本地重跑未完成。云端真实MRI冒烟成功，正式版本的训练源码未改变；生成器的哈希关卡通过。未修改或重装用户本地PyTorch环境。
