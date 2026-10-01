# 2026-10-01：main合并与仓库整理

## 分支核查

整理前GitHub main为`9055476`（9月25日），当前工作分支为`integrate-stage6-20260925`，已有`852b1df`、`2d8fcad`未进入main，另有9月29日至30日未提交成果。

`dinov2-pseudo-training`的`c1ab030`已是main祖先，因此其历史已经合并。`archive/legacy-working-tree-20260928`的`9f72a83`尚未并入提交图；其Stage4/5代码已存在于当前树，本次合并它的历史，README冲突采用最新项目状态，未回退到Stage5。

先以`29614fe`保存未提交成果，再将main快进到该提交，并合并上述legacy分支。继续在本工作目录的main工作。旧分支保留供追溯，不做强推和历史重写。

## 文件与标签

- 完整旧0.941方案和初期材料迁到`history/`，映射见其README。通过Git重命名及内容哈希检查确保旧内容保留。
- 已完成的Stage6实验及0.943方案保留在experiments，仍有共享引用的源码保留原路径。
- 原本忽略的v5、DeepSeek/GPT融合标签及两份报告纳入`data/processed/`白名单。v5标签SHA与训练回执一致；没有重算或更改数值。
- 9月25日合并时选择main版本的三份旧CSV不覆盖；功能分支版本额外保存到`history/labels/dinov2-pseudo-training/data/`。
- 权重、DICOM、图像缓存、认证信息仍不入Git；`.gitattributes`保留标签及Notebook的字节级哈希。

## 当前项目事实

Kaggle API于10月1日核实：四成员提交56700487为**0.943**，ConvNeXt 10%提交56706632同为**0.943**，两者COMPLETE。主入口选择四成员版，旧0.942作为回退。6F24训练完成，Gold58为0.84724，不是Public成绩。根README更新这些结果；旧实验报告按历史日期保留。

## 检查与同步

验证范围：Git合并祖先关系与未解决冲突、迁移前后的blob一致性、标签SHA、当前Notebook构建及融合测试、旧Anchor941测试、Stage6 CPU测试。最后推送main并用远端SHA核对。实际提交SHA以Git历史为准。
