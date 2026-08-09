# NLP 伪标签系统 — RSNA 2026 Knee Abnormality Detection

## 概述

本项目使用 LLM（DeepSeek-Chat）从放射报告中自动提取 12 类膝关节 MRI 异常标签，为多序列 2.5D 模型训练提供伪标签。

**核心问题：** RSNA 2026 膝关节异常检测竞赛仅有 58 个有标注样本，但有 4,349 份无标签放射报告。本系统将这些报告转化为可用的训练标签。

## 流水线架构

```
Phase 1: 验证                       Phase 2: 批量标注
─────────────────────              ─────────────────────
58 个有标签报告                      4,349 个无标签报告
      │                                    │
      ▼                                    ▼
┌──────────────┐                   ┌──────────────┐
│  LLM 推理     │                   │  LLM 推理     │
│  (deepseek-  │                   │  (deepseek-  │
│   chat)      │                   │   chat)      │
└──────┬───────┘                   └──────┬───────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐                   ┌──────────────┐
│  规则修正层   │                   │  规则修正层   │
│  (关键词纠正  │                   │  (关键词纠正  │
│   明显错误)   │                   │   明显错误)   │
└──────┬───────┘                   └──────┬───────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐                   ┌──────────────┐
│  置信度打分   │                   │  置信度打分   │
│  HIGH/MEDIUM │                   │  HIGH/MEDIUM │
│  /LOW/REVIEW │                   │  /LOW/REVIEW │
└──────┬───────┘                   └──────┬───────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐                   ┌──────────────┐
│  对比真值     │                   │  输出伪标签   │
│  计算指标     │                   │  CSV          │
└──────────────┘                   └──────────────┘
```

## 技术方案

### LLM 选择

| 模型 | 价格 (每百万 token) | 适用性 |
|------|---------------------|--------|
| **deepseek-chat** (选用) | $0.27 输入 / $1.10 输出 | 性价比极高，4,349 次调用 ~$2 |
| deepseek-v4-pro | — | 推理模型，输出格式不稳定，已弃用 |
| GPT-4 | $5 输入 / $15 输出 | 效果更好但贵 15 倍 |

### Prompt 设计

核心策略：
- **Few-shot 示例 (5 个)**：覆盖多发性损伤、骨水肿≠挫伤、滑膜炎+hoffitis、西班牙语、正常报告
- **多语言词汇表**：支持英语、西班牙语、荷兰语、希腊语、土耳其语、保加利亚语
- **关键区分规则**：`bone marrow edema ≠ contusion`、`soft tissue edema ≠ effusion`、`hoffitis = synovitis`

详细 Prompt 见 `scripts/llm_validate.py` 中的 `SYSTEM_PROMPT`。

### 置信度过滤

两层过滤机制：

**1. 规则修正层（免费，0 API 成本）**
- 报告有关键词但 LLM 判 0 → 纠正为 1（LLM 漏检）
- 报告有强否定但 LLM 判 1 → 纠正为 0（LLM 幻觉）

**2. 置信度打分**
- `HIGH`：LLM 预测有原文关键词证据支撑 → 直接用作 gold label
- `MEDIUM`：LLM 预测但找不到原文证据 → 可降权使用
- `LOW`：LLM 大概率过度解读 → 建议丢弃
- `REVIEW`：LLM 判正常但报告有关键词 → 可能是假阴性，人工核查

## 验证结果（58 样本）

```
Overall Accuracy: 0.845
Macro F1:         0.782

Per-class F1:
  ACL 0.920 | MCL 0.857 | Med Meniscus 0.868 | Lat Meniscus 0.826
  Med OA 0.848 | Lat OA 0.667 | PF OA 0.773
  Effusion 0.814 | Synovitis 0.638 | Baker's 0.786
  Contusion 0.684 | Fracture 0.743
```

### 主要局限

| 问题 | 影响 | 原因 |
|------|------|------|
| Synovitis 漏检 41% | F1=0.638 | LLM 不认识 "synovial hypertrophy"、"pannus" 等间接表达 |
| Lateral OA 精确率 56% | F1=0.667 | LLM 难以区分 OA 的具体间室 |
| 非英语报告 | 部分漏检 | 希腊语/土耳其语/保加利亚语术语覆盖不全 |

## 输出文件

### 批量标注结果

| 文件 | 大小 | 说明 |
|------|------|------|
| `data/pseudo_labels.csv` | 647 KB | 4,349 × (StudyInstanceUID + 12 pred + 12 conf) |
| `data/pseudo_labels_stats.csv` | — | 置信度分布统计 |

### 验证结果

| 文件 | 说明 |
|------|------|
| `data/pseudo_labels_valid.csv` | 58 样本逐样本对比（预测 vs 真值） |
| `data/nlp_validation_report.csv` | Per-class Precision/Recall/F1/Accuracy |
| `data/nlp_validation_filtered.csv` | HIGH 置信度过滤后的指标 |

## 伪标签统计

```
样本数: 4,349 (0 FAIL)

置信度分布:
  HIGH     88.9%  46,380 预测 — 直接用作训练标签
  MEDIUM    9.9%   5,146 预测 — 可降权使用
  LOW       1.3%     655 预测 — 建议丢弃

全 HIGH 样本: 2,086 (48%) — 可直接当 gold
HIGH+MEDIUM:  3,846 (88%) — 全量可用
```

### 各类阳性率

| 类 | 阳性数 | 占比 | 可靠性评估 |
|---|--------|------|-----------|
| Effusion | 2,435 | 56.0% | 高（样本充足） |
| PF OA | 1,828 | 42.0% | 中（精确率 0.77） |
| Medial Meniscus | 1,779 | 40.9% | 高（F1=0.87） |
| Medial OA | 1,584 | 36.4% | 中（精确率 0.78） |
| Synovitis | 1,093 | 25.1% | **低（召回率 0.56）** ⚠️ |
| Baker's | 1,072 | 24.6% | 高（F1=0.79） |
| Lateral OA | 892 | 20.5% | **低（精确率 0.56）** ⚠️ |
| ACL | 696 | 16.0% | 高（F1=0.92） |
| Lateral Meniscus | 677 | 15.6% | 中（F1=0.83） |
| Contusion | 467 | 10.7% | 中（F1=0.68） |
| Fracture | 323 | 7.4% | 中（F1=0.74） |
| MCL | 321 | 7.4% | 高（F1=0.86） |

> ⚠️ Synovitis 和 Lateral OA 的伪标签噪声较大，建议训练时降权或使用 HIGH-only 子集。

## 训练使用建议

### 方案 A：保守（推荐先用这个验证基线）

```python
# 只使用全 HIGH 置信度样本
high_mask = df[[f"conf_{c}" for c in LABEL_COLS]].apply(
    lambda row: (row == "HIGH").all(), axis=1
)
train_df = df[high_mask]  # ~2,086 样本
```

### 方案 B：全量（更多数据但更多噪声）

```python
# HIGH+MEDIUM 全部使用
usable_mask = df[[f"conf_{c}" for c in LABEL_COLS]].apply(
    lambda row: ((row == "HIGH") | (row == "MEDIUM")).all(), axis=1
)
train_df = df[usable_mask]  # ~3,846 样本

# 关键: Synovitis 和 Lateral OA 仅保留 HIGH 置信度
train_df.loc[train_df["conf_Synovitis"] != "HIGH", "pred_Synovitis"] = 0
train_df.loc[train_df["conf_Lateral OA"] != "HIGH", "pred_Lateral OA"] = 0
```

### 方案 C：加权训练

```python
# 根据置信度给样本加权
weight_map = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0, "REVIEW": 0.5}
sample_weight = df[[f"conf_{c}" for c in LABEL_COLS]].map(weight_map).mean(axis=1)
```

## 脚本清单

| 脚本 | 用途 |
|------|------|
| `scripts/llm_validate.py` | Phase 1: 在 58 个有标签报告上验证 LLM 准确率 |
| `scripts/llm_batch_label.py` | Phase 2: 批量标注 4,349 个无标签报告 |
| `scripts/api_config.py` | API 密钥配置（gitignored） |
| `scripts/api_config.example.py` | API 配置模板 |

## 运行命令

```bash
# Phase 1: 验证（58 样本）
python scripts/llm_validate.py --limit 0

# Phase 2: 批量标注（4349 样本，约 1.5 小时）
python scripts/llm_batch_label.py
```

## 成本

```
58 验证样本:     ~$0.03
4,349 批量标注:   ~$2.00
─────────────────────────
总计:            ~$2.03
```

## CSV 字段结构

### `data/pseudo_labels.csv` — 批量标注输出（4,349 行）

```
列名                         类型    说明
──────────────────────────────────────────────────────
StudyInstanceUID             str     研究实例 UID（主键）
pred_ACL                     int     预测标签: 0=正常, 1=异常, -1=失败
pred_MCL                     int
pred_Medial Meniscus         int
pred_Lateral Meniscus        int
pred_Medial OA               int
pred_Lateral OA              int
pred_PF OA                   int
pred_Effusion                int
pred_Synovitis               int
pred_Baker's                 int
pred_Contusion               int
pred_Fracture                int
conf_ACL                     str     置信度: HIGH / MEDIUM / LOW / REVIEW / FAIL
conf_MCL                     str
conf_Medial Meniscus         str
conf_Lateral Meniscus        str
conf_Medial OA               str
conf_Lateral OA              str
conf_PF OA                   str
conf_Effusion                str
conf_Synovitis               str
conf_Baker's                 str
conf_Contusion               str
conf_Fracture                str
```

### `data/pseudo_labels_valid.csv` — 验证输出（58 行）

```
除上述 pred_* 和 conf_* 列外, 额外包含:
  Report_snippet             str     报告前 200 字符
  true_ACL                   int     竞赛官方标签 (0/1)
  true_MCL                   int
  ... (12 个 true_* 列)
  n_corrected                int     该样本被规则修正层翻转的预测数
```

### `data/nlp_validation_report.csv` — Per-class 指标（12 行）

```
  class                      str     类名
  precision                  float   精确率
  recall                     float   召回率
  f1                         float   F1 分数
  accuracy                   float   准确率
  n_positive                 int     真值中阳性样本数
```

### `data/pseudo_labels_stats.csv` — 置信度分布（5 行）

```
  confidence                 str     HIGH / MEDIUM / LOW / REVIEW / FAIL
  count                      int     预测数量
  pct                        float   百分比
```

## 分支协作

```
main                              ← 干净基线代码（35 文件）
nlp-pseudo-label（当前分支）        ← NLP 伪标签 + 数据预处理
archive/contaminated-baseline     ← 归档: 旧实验历史（NaN→0 bug 等）
origin/Ensemble                   ← 远程: 集成/实验分支
```

