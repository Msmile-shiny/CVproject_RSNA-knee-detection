# RSNA Knee Super Ensemble Replica (LB 0.92)

复刻 `amanatar/rsna-knee-super-ensemble` v1 (源码逐字一致, 本地 SHA 已核对),
即 Tony Li 的 DINO-RadImageNet Rank Ensemble + master rank blend。

## 配方 (默认 env, 无覆盖)

- parent = 0.55·(20×DINOv2 rank 均值) + 0.45·(5×DINOv3 rank 均值)
  → E10 rad 分支 (R50 编码器 + 5 参考头 + 5 E13 头, 0.5/0.5, 10 类; Baker's/Fracture 保留 parent)
  → E11 二遍 E13 (0.85/0.15, 全 12 类)
- legacy = 20 DINOv2 五折 frontier+soft 混合 (0.06)
- **最终 submission.csv = 0.94·parent_rank + 0.06·legacy_rank** (B3 未挂载时)
- 若挂载 B3 包 (prvsiyan/rsna-knee-b3-v47-public-deployment): +0.05·B3_rank (audit 门控)

## Kaggle 侧操作 (两条路)

### 路线 A (推荐, 版本精确): 网页 Copy & Edit

1. 打开 https://www.kaggle.com/code/amanatar/rsna-knee-super-ensemble
2. **Copy & Edit** → 数据集/模型/加速器 (T4) 全部继承原作者的精确版本 pin
3. Settings: Accelerator 建议 **GPU T4 x2** (评分不耗配额, 更快更稳; 原版是单 T4)
4. **Save Version** → 交互 Run All 冒烟一次 (3 行 test, 分钟级): 看日志无 Traceback, 输出 submission.csv
5. **Submit** → 预期 LB ≈ 0.92

### 路线 B (备选): CLI push 本包

```bash
export PYTHONUTF8=1
export KAGGLE_USERNAME=$(python -c "import json;print(json.load(open('C:/Users/eason/.kaggle/kaggle.json'))['username'])")
export KAGGLE_KEY=$(python -c "import json;print(json.load(open('C:/Users/eason/.kaggle/kaggle.json'))['key'])")
kaggle kernels push -p notebooks/kernel_push_super
```

然后网页 Settings 设 **GPU T4 x2** → Save → 交互冒烟 → Submit。
注意: push 按 slug 挂**当前版本**数据集 (原作者 08-15 跑时的版本已漂移, 主要是
tonylica repro-assets 08-17 有过更新); 路线 A 继承原版 pin, 复刻保真度更高。

## 可选实验: B3 分支 (或 >0.92)

原作者评分配置 (12 数据集) **没有** B3 包, 0.92 = parent+legacy。
notebook 设计上支持 B3 (master 加 0.05, audit 门控): 在网页 notebook 的
Data 里**加挂 `prvsiyan/rsna-knee-b3-v47-public-deployment`** (自定义挂载路径
`/kaggle/input/rsna-knee-b3-v47-folds-0-3`, 代码自动发现), 再交一次。
建议先拿到 0.92 基线后再做这个实验。

## 检查清单 (提交前)

- [ ] Accelerator = GPU T4 x2 且已 Save Version
- [ ] Data 面板: 12 个数据集 + 2 个 dinov2 模型 + 竞赛数据
- [ ] 交互冒烟无 Traceback, submission.csv 存在 (3 行 test)
- [ ] 评分运行时长记录到 PLAN_STAGE2.md 结果表

## 风险

- tonylica repro-assets 08-17 更新过 (评分用当时版本, 见上); rad 头有 sha256 校验
  兜底 (校验失败会跳过该臂), DINOv2/DINOv3 折权重无 sha 校验
- 私有 BYOD docker (gcr.io/kaggle-private-byod) 未 pin — v47 经验: 默认镜像分数与
  作者完全一致 (0.914=0.914), 预期无影响
