# Reproducible code for "基于XGBoost与LightGBM的糖尿病风险预测：双公开队列下的方法学评估与SHAP可解释性分析"

> This repository contains the Python code to reproduce the main results in the manuscript.
> **No raw datasets are included due to dataset copyright restrictions.**

## Environment
Python 3.7, see `requirements.txt` for package versions.
> Suggestion: Run scripts under a local folder without Chinese characters or spaces. All file paths inside scripts are relative paths.

## Data acquisition
1. Pima-Indians-Diabetes dataset: original UCI archive is offline, please obtain backup public de-identified copy from public resources.
2. Chinese clinical diabetes dataset: download from Kaggle:
`kagglehub.dataset_download("pkdarabi/diabetes-dataset-with-18-features")`

## Main scripts (these scripts produce **manuscript main results**)
1. `pima-deal.py`: Pima-Indian cohort modeling, produce Table 1, Pima ROC、DCA、SHAP.
2. `chinese_deal.py`: Chinese cohort modeling, produce Table2, Table3, risk‑stratification, ROC、DCA、SHAP.
3. `supplemental_analysis.py`: internal robustness validation: temporal‑split validation, MICE imputation comparison, gender subgroup analysis (Results section 3.6).

> Output figures and csv tables will be auto‑saved into local subfolders after execution.
> Note: **Table 4 (ablation experiment) cannot be directly reproduced by above three scripts**. The ablation experiment requires additional modification to fix classification threshold to 0.30 for all experimental groups.

## Ethics note
This work is secondary analysis of publicly available de‑identified datasets, no human subjects recruitment performed.
