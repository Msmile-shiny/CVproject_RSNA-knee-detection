# 2026-10-16 社区复查

只读查询最近运行的 30 个公开 Notebook，下载审计四份源码及元数据；没有新增 GPU 或比赛提交。原始清单和源码位于 experiments/community1016。

## 优先候选：准确 R6 + 双 Raptor + 镜像 + 逐类别融合

来源 https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters

代码 SHA256 ad0b97aca3361f16e32b0836655dd68bc2a37ff0cc5a44f6d6f4de07880d986d。
并未从网页取得该准确版本的真实 Public 分数，不把下游注释中的 0.958 当已验证成绩。

- R6 bundle fe055406fad6a12f，四折 ConvNeXt-small，384 输入/352 裁剪；不是已用的 R7 bundle 8c7b666e。
- 开启配套镜像 TTA（内外侧标签交换回来，MCL 保持原始通道）。
- Raptor 部分为 65% 384 模型 + 35% 224 模型的排序融合。
- 最外层 Raptor 比重：Synovitis .80，Lateral OA .20，PF OA .60，Baker's .35，其余 .50。
- 源码写着 private，但本次 API 能读取 heliosli/rsna-knee-r6-frozen-fe055406-w1。rostigor/rsna-knee-r6-fe055406 的 manifest 完全一致：6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648。只检查清单，未下载或验证所有大权重字节；运行时必须逐项 SHA 验证。

值得优先准备一次严格复现，而不是直接把这些比例套到 R7 上。不同疾病用不同权重，意思是把某类判断交给更擅长它的模型；不代表从少量 Gold58 数据就能可靠估计这些权重。

## 其他三份

1. goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-957：实际增加 R6/R7 混合及旧三折 reader 小权重，存在 R7_STAGE_SKIPPED 与 reader 失败保留父本逻辑。不是一个全新独立模型；不可把回退输出当完整组合。暂排第二，先不要同时增加五个分支。
2. prvsiyan/the-bee-s-knees-final-rsna-push：说明中重点是有界预处理、镜像、固定 55/45 家族权重和执行改造；未确认新模型或真实成绩，不优先消耗 GPU。
3. ltcsmokeboi/rsna-knee-gp958e：实际 A224=0、外层 .5/.5，且源码明确指出名字叫 R6 的最新包已装 R7；名称 gp958e 不能证明新模型或 0.958。必须按 bundle 判身份，暂不复现。

网络搜索另发现标题 Fast Parent 0.957 的页面实际展示 Public 0.939，更说明不能按标题排序候选。

## 决策

S6 57257123 继续等待实际评分。下一优先是准确 R6 逐类别配方的单次复现，先补齐完整成员拒绝回退、总时长和哈希门禁，再考虑启动。当前不自动新增训练或比例扫描，保护 57144024 Public .956，不改最终选择。若复现提升可作为候选；不能保证 Private 或奖牌。
