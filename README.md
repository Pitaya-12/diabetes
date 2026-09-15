# Risk screening assessment of diabetes prevalence based on XGBoost and Voting
This repository contains all analysis scripts for the manuscript:
*Risk screening assessment of diabetes prevalence based on XGBoost and Voting: a methodological evaluation with SHAP interpretability across two public cohorts.*

## File Description
| Filename | Description |
| ---- | ---- |
| `pima‑deal.py` | Data preprocessing, model training and evaluation for Pima‑Indians Diabetes cohort |
| `chinese_deal.py` | Data preprocessing, model training and evaluation for Chinese clinical diabetes cohort |
| `supplemental_analysis.py` | Supplementary analysis: index‑based internal stability test, subgroup analysis, ablation study, DCA and SHAP interpretability analysis |
| `README.md` | This document |

> ⚠️ Important Note: **Raw patient datasets are NOT included in this repository**. Redistribution of original data is prohibited by copyright and dataset license terms.

### How to obtain datasets
1. **Pima‑Indians Diabetes Dataset**
The original UCI archive has been removed. De‑identified backup data can be obtained upon reasonable request with reference to the original publication (Smith JW, et al., 1988).

2. **Chinese clinical diabetes dataset (Kaggle)**
Download via kagglehub:
```python
kagglehub.dataset_download("pkdarabi/diabetes‑dataset‑with‑18‑features")

# 基于XGBoost与Voting的糖尿病患病风险筛查评估
Risk screening assessment of diabetes prevalence based on XGBoost and Voting

本仓库存放论文《基于XGBoost与Voting的糖尿病患病风险筛查评估：双公开队列下的方法学评估与SHAP可解释性分析》的全部分析脚本。

## 文件说明
| 文件名 | 功能描述 |
| ---- | ---- |
| `pima‑deal.py` | 皮马印第安糖尿病队列数据预处理、模型训练与评估脚本 |
| `chinese_deal.py` | 中国临床糖尿病队列数据预处理、模型训练与评估脚本 |
| `supplemental_analysis.py` | 补充分析脚本：基于样本索引的内部稳定性检验、亚组分析、消融实验、DCA、SHAP可解释性分析 |
| `README.md` | 本说明文档 |

> ⚠️重要说明：本仓库**不包含原始患者数据集**，受版权及数据集许可限制，无法分发原始数据。

### 数据集获取方式
1. **皮马印第安糖尿病数据集（Pima‑Indians Diabetes Dataset）**
原始UCI公开存档已下架，脱敏备份数据可参考原始文献 Smith JW, et al. (1988) 合理申请获取。

2. **中国临床糖尿病数据集（Kaggle）**
可通过 kagglehub 下载：
```python
kagglehub.dataset_download("pkdarabi/diabetes‑dataset‑with‑18‑features")
