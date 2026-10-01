# 标签入口与来源

本次合并把已经使用的标签资产及历史分支纳入main，没有重新计算标签。

| 文件 | 用途 |
|---|---|
| `processed/v5_labels.csv` | Stage6使用的概率/权重标签；SHA256 `c13adffaabf4f8e518abb038282bb1aa09baac7652a9165e030710c457d0be6a`，与训练回执一致 |
| `processed/v5_fusion_report.csv` | v5融合诊断 |
| `processed/pseudo_labels_deepseek_gpt56sol_fused.csv` | 历史DeepSeek/GPT融合结果；SHA256 `efd3d39e7a0b7e8c78ead857c3ef43953eac3eb7a7e1a8a040bb1dac19b49d59` |
| `processed/pseudo_labels_deepseek_gpt56sol_fused_report.csv` | 类别替换报告 |
| `pseudo_labels*.csv`、`external_labels/gpt56sol/` | 原始/校准标签及GPT标签来源，保留既有引用路径 |

四个processed文件从原项目本地目录复制并核对SHA后纳入Git；缓存、打包图像及checkpoint继续忽略。当前0.943是推理方案，所需挂载以kernel metadata为准。

9月25日合并中三份旧标签/诊断文件采用了main版本；功能分支版本另存于`history/labels/dinov2-pseudo-training/`。Gold58曾用于标签校准和多轮选择，不能视为独立泛化证明；历史脚本关于teacher OOF的描述须结合来源审计解读。
