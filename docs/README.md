# 项目文档索引

根目录只保留项目入口。计划、实验报告和调研记录按用途归档在这里；文件名中的阶段表示当时的实验顺序，并不代表仍然有效。

## 当前入口

- [双候选冲榜执行（2026-10-05）](experiments/sprint1005_execution.md)：CPU权重检查通过，两个独立完整候选已启动，完成后自动核验并各提交一次。
- [公开三折ConvNeXt与替代路径审计](research/alternate_candidates_20261005.md)：公开0.944作者报告的依赖和输入，two-target student暂缓。
- [半月板增量执行计划（2026-10-05）](plans/meniscus_increment_20261005.md)：找回原作者公开入口，已启动CPU输入/权重门槛检查，保护0.943父本。
- [社区与论文复核（2026-10-04）](research/community_and_literature_20261004.md)：半月板bag资产审计、缓存兼容性阻碍及下一轮启动门槛，尚未启动新GPU任务。
- [Outer70执行与评分（2026-10-04）](experiments/outer70_execution_20261004.md)：提交56817443已COMPLETE、Public0.943持平；停止该权重方向，保留父本。
- [0.943完整审计及提交策略（2026-10-02）](experiments/strong_pipeline_audit_20261002.md)：在线资产及manifest核验、各分支输入、Outer70结果检查器、保护版本和最终选择标准。
- [无GPU准备与Outer70（2026-10-02）](experiments/no_gpu_preparation_20261002.md)：父本融合审计、单项对照Notebook、CPU验证和额度恢复后的运行顺序。
- [main整理与合并（2026-10-01）](experiments/repository_cleanup_20261001.md)：0.943状态、标签资产、历史目录和分支合并。
- [9月30日冲榜纠偏与实际候选](experiments/leaderboard_sprint_20260930.md)：四成员公开升级、固定10% ConvNeXt增量；暂停自动扩展Stage6消融。
- [Stage 6F结果与24轮续训（2026-09-30）](experiments/stage6f_results_20260930.md)：288px小幅收益、逐类变化和完整缓存复用续训。
- [Stage 6E24结果与6F启动（2026-09-29）](experiments/stage6e24_results_and_6f_launch_20260929.md)：覆盖收益保持，Gold 0.8469；启动288px单变量试验。
- [Stage 6E结果与24轮续训（2026-09-29）](experiments/stage6e_results_20260929.md)：Gold 0.8347、配对不确定性、内侧半月板退步与双缓存续训。
- [Stage 6E预处理暂停与续跑（2026-09-28）](experiments/stage6e_preprocessing_pause_20260928.md)：3,793/4,407例已缓存，首轮未训练；已启动锁定来源的续跑。
- [社区与论文复查（2026-09-27）](research/community_and_literature_20260927.md)：单模型、标签、输入几何与半月板定位的证据，以及6E/6F单变量冒烟。
- [Stage 6C结果与6E覆盖冒烟（2026-09-27）](experiments/stage6c_results_and_6e_launch_20260927.md)：报告混合监督的同目标对照与8位置输入冒烟。
- [Stage 6D细析与6C监督对照启动](experiments/stage6d_analysis_and_6c_launch_20260926.md)：注意力的类别收益/损失、逐例排序与经提及语义审核的新标签试验。
- [序列审计完成与注意力对照](experiments/stage6_audit_and_attention_20260926.md)：头信息抽查无异常，6C监督暂缓，6D单变量训练已推送。
- [Stage 6B结果与序列审计（2026-09-26）](experiments/stage6b_results_20260926.md)：24轮完成，Gold 0.8280；不再机械续训，进入输入和监督归因。
- [社区最新代码复查（2026-09-26）](research/community_update_20260926.md)：DINOsaur今日版本、Master 4-Arm和监督对照的证据边界。
- [Stage 6A结果与6B续训（2026-09-26）](experiments/stage6a_results_20260926.md)：12轮完成、Gold开发AUC 0.8156、固定续训至24轮；不是Public成绩。
- [main 合并审计（2026-09-25）](experiments/main_merge_audit_20260925.md)：两边文件清单核对、冲突处理与历史入口保留。
- [Stage 6A 正式训练启动（2026-09-25）](experiments/stage6a_launch_20260925.md)：真实MRI冒烟通过、正式v3运行中、Fracture对照0.942。
- [Stage 6 持续训练主线](plans/stage6_sustained_training.md)：失败归因、端到端ResNet34参考、固定验证和续跑；当前执行入口。
- [项目复盘与路线（2026-09-24）](experiments/project_review_20260924.md)：0.942 实分、当前社区和论文证据、Fracture 对照及独立模型的进入门槛。
- [项目状态（2026-09-21）](experiments/project_status_20260921.md)：历史阶段记录；复现 0.941 与 Native64 0.940 回退。
- [社区更新（2026-09-21）](research/community_update_20260921.md)：公开 0.942 Speedy/D4 路线、标称 0.943 方案的证据分级和 Anchor942 决策。
- [Stage 5A 计划与运行说明](plans/stage5a.md)：历史路线，已完成筛选，暂停扩大训练。
- [Stage 5A 超时恢复](experiments/stage5a_timeout_recovery.md)：420分钟主动中断的根因、旧特征续跑与阶段性结果导出。
- [Phase 4 独立成员计划](plans/phase4_independent_member.md)：已暂停路线、Phase 4A/4B 门槛和 Kaggle 离线资产。
- [Stage 3C 实验报告](experiments/stage3c.md)：新融合伪标签退化的证据，以及 Stage 3D 的由来。
- [Stage 3D 与 Phase 4A 实验报告](experiments/stage3d_phase4a.md)：可信排序的 Public 结果、OrthoFoundation 冻结探针和 Phase 4A2 决策。
- [Stage 3B 文献与社区调研](research/phase3b_2026-09-06.md)：社区方案、论文与候选创新方向。

## 历史计划

- [Stage 2 集成计划](plans/stage2.md)
- [Stage 2 成员注入说明](plans/stage2_member_injection.md)
- [Stage 3A 计划](plans/stage3a.md)

## 实验报告

- [Stage 3B](experiments/stage3b.md)
- [Stage 3C](experiments/stage3c.md)

当前推理在 `experiments/sprint0930/`，回退在 `experiments/anchor942/`；0.941旧方案迁入 `history/anchor941/`。

## 归档

- [最初图像训练计划（2026-08-06）](archive/initial_training_plan_2026-08-06.md)
- [早期 NLP 伪标签说明](archive/nlp_pseudo_label.md)

归档文档用于解释历史决策，不应直接作为当前运行指令。当前状态以根目录 README 和最新项目状态报告为准。
