# PLAN STAGE 2 — 0.914 之后的一阶段计划 (2026-08-17 起)

> 状态快照 (2026-08-19 更新): v47 诊断**已结案** — 作者本人成绩就是 0.914, 我们的复刻是忠实的, 不存在部署缺陷。
> 阶段 1 目标已升级为 **super-ensemble 复刻 (LB 0.92)**, 复刻包已备好并 push 到 git, 等你自己提交 Kaggle。
> 每步标注 **[SOLO]** (可独立完成) / **[CLAUDE]** (等我回归)。
> 竞赛截止日: ____ (自行查 Kaggle 填); 每日提交上限: ____ (查 Submissions 页, 通常 5/天)。
> 铁律不变: 本地永远 CPU (BSOD 禁令) / GPU 只用于 Kaggle 云端 / 评分提交不耗周配额 / 结论必须落本文件或 memory。

---

## 已知事实 (我刚查证的, 无需重查)

| 事实 | 来源 |
|---|---|
| v47 原版 docker = `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` (私有 BYOD 镜像) | v47_pull.json |
| v47 原版 machineShape = `NvidiaTeslaT4` (单 T4) | v47_pull.json |
| 我们的 fork 是 CLI push → metadata **未带** docker_image / machine_shape → 评分跑默认镜像 + 默认加速器 | kernel_push/kernel-metadata.json |
| 默认加速器曾出过 P100 probe FAIL (sm_60 kernel 缺失 → CPU 降级), run-2 记录 | 之前运行日志 |
| LB 0.93 公开 notebook = `ranjithragavan07/rsna-knee-dinov2-0-93` (v16, 53 votes, 最后运行 2026-08-14) | Kaggle 搜索 |
| 0-93 notebook 数据集 **9 个全部公开** (knee-mri-fold-weights / pilkwang-public-figures / resnet-50-radimagenet / e11-diverse-heads-v20 / llm-labels ×2 / radimagenet-foldsv1-heads / v52-radimagenet-heads-20260812 / pilkwang-weights), 无 kernel 源, **同样引用上述 gcr.io 私有镜像**, machineShape 也是 T4 | API pull |
| 网页 **Copy & Edit 会继承 docker 镜像设置**; CLI push 不会 | Kaggle 行为 |
| M2 权重 (m2_f0..4.pt) 无任何公开来源 → 私有, v47 复刻上限天然低一小截 (+0.0014~0.0039, 作者自报) | 已结案 (memory) |
| 58-gold α 扫描裁决: v47 核心在 gold 饱和 0.9956, 任何 α>0 掉分 → 第 21 成员混入无本地证据支持 | scripts/alpha_scan_v47_v5.py |
| **0.914 结案**: v47 作者 (sofiaanjenje) 自己的得分就是 0.914 → 我们的 fork 是忠实复刻, docker/machineShape 假设作废 (至少解释不了任何差距 — 本就没有差距) | 用户消息 2026-08-19 |
| 更高公开基线 = `amanatar/rsna-knee-super-ensemble` (Tony Li DINO-RadImageNet Rank Ensemble), **LB 0.92**; 35 成员 + R50 rad 臂 + legacy fold blend | reference_code/ 已下载 |
| 其评分配置 = 12 数据集 (tonylica repro-assets 集成包) + 2 dinov2 模型, **无 B3**; 0.92 = 0.94·parent_rank + 0.06·legacy_rank | 复刻时核出 |
| 复刻包已备好: `notebooks/kernel_push_super/` (notebook 与原文逐字 sha256 一致 + metadata + README), 校验 `scripts/verify_super_ensemble_package.py` 全过 | 本次会话 |

---

## 阶段 0 — 诊断 0.914 (已结案 2026-08-19)

作者 (sofiaanjenje) 本人成绩就是 **0.914** → 我们的 fork 是忠实复刻, 不存在部署缺陷。
docker/machineShape 假设作废 (v47 经验: 默认镜像分数与作者完全一致)。
v47 线 (含第 21 成员注入) 归档弃用 — 其注入模板 (member cell exec 隔离 + emission) 保留在
`notebooks/kaggle_fork_v47_ours.ipynb` + `build_fork_v47.py`, 阶段 2 复用, 不删。

---

## 阶段 1 — super-ensemble 复刻提交 (0 配额, 全部 [SOLO], 目标 LB ≈ 0.92)

> 复刻包已备好并 push 到 git: `notebooks/kernel_push_super/`
> (notebook 与 `amanatar/rsna-knee-super-ensemble` 原文逐字一致, sha256 校验过; README 含两条路线)。

1. **[SOLO] 路线 A (推荐)**: 打开 https://www.kaggle.com/code/amanatar/rsna-knee-super-ensemble → **Copy & Edit**
   (数据集/模型/加速器版本 pin 全部继承) → Settings: Accelerator **GPU T4 x2** → **Save Version** →
   交互 Run All 冒烟 (3 行 test, 分钟级; 扫日志: 无 Traceback / submission.csv 存在) → **Submit** → 记 LB。
2. **[SOLO] 路线 B (备选)**: 本机 CLI push 包 (`kaggle kernels push -p notebooks/kernel_push_super`,
   README 有 env 设置), 然后网页设 T4 x2 → Save → 冒烟 → Submit。注意 push 挂的是**当前版本**数据集,
   不如路线 A 精确 (tonylica repro-assets 08-17 有更新)。
3. **[SOLO] 可选 B3 实验**: 拿到 0.92 基线后, 网页 Data 面板**加挂 `prvsiyan/rsna-knee-b3-v47-public-deployment`**
   (自定义路径 `/kaggle/input/rsna-knee-b3-v47-folds-0-3`, 代码自动发现 + audit 门控) → 再交一次, 记 LB。
   - 预期 +0.001~0.003; 无增益也不亏 (0 配额)。
4. **决策门**: LB **≥0.92** → 基线升到 0.92, 转阶段 2; **<0.92** → 检查日志关键词 (rad 头 sha 校验跳过?
   数据集版本漂移?) → 转路线 A 重试或等 [CLAUDE]。

---

## 阶段 2 — 我们的成员注入 super-ensemble (等 [CLAUDE] 回归)

目标: 在 0.92 基线上测第 21 成员 (v5 3-seed, gold 0.8959 / LB 0.886)。注意: 该 notebook 只有 **6 个 cell**,
master rank blend 在 cell 5 (0.94/0.06/0.05 权重, env 可调), 注入面比 v47 (40 cells) 简单得多 —
加一个 member cell + 在 master blend 处加 (1-α)/α 项即可, 不必碰内部管线。

1. 移植 member cell: 复用 `build_fork_v47.py` 的 exec 隔离模式; 核对该 notebook 的 globals (ASSET/ROOT/DINO/CKPT + rank-blend 输出变量名)。
2. 移植 gold emission → 交互跑 (3 行占位 test) → 下载 gold_members/ → α 扫描 → **只用全局 macro 裁决 α** (≥+0.003 标准, lateral-swap 教训: 单类决策勿做)。
3. α>0 → ACTIVE=True 提交; α=0 → 保持纯复刻, 把精力转阶段 3。

---

## 阶段 3 — 自研管线 (长线, 每周 GPU 15h, 训练步骤 [SOLO] 可跑, 裁决建议 [CLAUDE])

目标: 自己的新成员 (Bend the Knee 设计笔记: DINOv3 slot-token / E11 cross-attn)。

1. 每次训练 ≤2h (15h 配额内), RAM 防泄漏护栏照 v6a 教训 (DP replicate 泄漏结案), 每跑完: CPU 冒烟 + 下载产物到 results/。
2. 新成员一律先过 `scripts/fusion_scan_v6.py` 全局裁决 (≥+0.003) 才允许入队。
3. 公共资产继续用: champ/llm199 五折权重、two-target lateral 学生 (CPU 可跑)、公开 OOF —— fusion_scan 扫描入场。
4. 队内 ≥3 个裁决通过的成员 → 3-seed 风格 rank 融合 → 提交。

---

## 阶段 4 — 日常纪律 (全程)

- 评分提交 **0 配额** 但计入每日次数: 错峰提交, 别一天打光。
- 交互运行烧配额: 只有冒烟必需时才 Save&Run, 控制在分钟级 (3 行 test)。
- 本地一切脚本 CPU-only (已全部如此); 新写脚本同理。
- 关键结论随手写进本文件「结果记录」; 我回归后按此续接。

---

## 交接包 (我回归时, 把这几样给我)

1. 「结果记录」表: 每行 = 日期 / notebook / 设置 (加速器+镜像) / 运行时长 / LB。
2. 各 notebook 是否用 Copy & Edit 继承镜像 (是/否)。
3. 阶段 3 若有训练: 产物路径 + CPU 冒烟输出。
4. 每日提交余量 + 竞赛截止日。

---

## 结果记录

| 日期 | notebook | 设置 | 运行时长 | LB | 备注 |
|---|---|---|---|---|---|
| 2026-08-17 | easoncyy/rsna-knee-fork-v47-ours (CLI push v4) | 默认镜像 + 默认加速器 | __ | **0.914** | 结案: 作者本尊就是 0.914, 忠实复刻 ✓ |
| 2026-08-19 | super-ensemble 复刻包 (`kernel_push_super/`) | — (未提交) | — | — | 包已备好 + verify 全过 + push git; 等用户按阶段 1 提交 |
| | | | | | |
