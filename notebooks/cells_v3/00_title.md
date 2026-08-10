# DINOv2 Multi-View v3 — 6-Slot Clinical MRI + SlotHead

### 相比 v2 的核心变更

| 维度 | v2 (昨天) | v3 (本次) |
|------|----------|----------|
| **输入** | 单视角 Sagittal T2 FS × 5 slices | 6 slot × 3 平面 (SAG/COR/AX, Fluid+T1) |
| **特征** | CLS token only (384-dim) | CLS + mean(patches) + focal_topk (1152-dim) |
| **Head** | SPA + CrossModalFusion + SliceTransformer + MLP | SlotHead: per-diagnosis attention over slots + anatomical priors |
| **图像** | 392×392 | 224×224 (DINOv2 原生, attention 快 9.4×) |
| **验证** | 56 gold studies (仅 Sagittal T2 FS) | ~200 gold studies (全部有任意 slot 的) |
| **速度** | ~1.5h/epoch | 预估 ~5-10min/epoch (224² + 简化架构) |

### 为什么多视角能解决过拟合

1. **信息量提升 6×**: ACL 需要 Sagittal, Baker's 需要 Axial, OA 需要 Coronal——单视角看不到的东西永远学不会
2. **SlotHead anatomical priors**: 模型内置先验——ACL 优先关注 Sagittal 序列, MCL 优先 Coronal, 不是从零学
3. **特征表达提升 3×**: CLS + mean + focal top-k 捕捉全局+平均+最强局部信号

### 与参考代码 (knee-rsna.ipynb) 的对齐

- ✅ 相同的 6 个 clinical slot 定义
- ✅ 相同的 SlotHead 架构 + anatomical priors
- ✅ 相同的特征提取 (CLS + mean + focal_topk)
- ✅ 相同的 224×224 输入 + 3-slice RGB窗口
- ✅ 相同的 DICOM 空间排序 + 侧位归一化
- ⚠️ 暂不包含物理 crop (PixelSpacing 依赖, 简化版)
- ⚠️ 暂不包含 TTA 重叠窗口 (仅训练, 非推理)
