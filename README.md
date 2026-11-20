# Cross-cohort evaluation of machine-learning models for prevalent diabetes identification

> Release note: v1.1.0 was deleted and is superseded by v1.1.1 (the Kaggle-derived data files were removed from the repository and its history).

Analysis code and data for the manuscript:
*Cross-cohort evaluation of machine-learning models for prevalent diabetes identification: ensemble selection, interpretability, and an empirically-documented caution regarding public clinical datasets*
(PEERJ submission; code–text audit dated 2026-11-20)

## 1. Data files

| File | Status | Description |
| ---- | ---- | ---- |
| `pima_diabetes.csv` | included | Pima Indians Diabetes Database (768 records, 8 predictors + outcome), an unmodified public UCI benchmark |
| `diabetes.csv` | **not included** | PUC raw records (Kaggle `pkdarabi/diabetes-dataset-with-18-features`, 4,303 rows) — must be downloaded independently (dataset terms of use) |
| `chinese_clean.csv` | **not included** | PUC after quality control (4,061 rows) — regenerate locally with `build_clean.py` |
| `split_indices_seed42.csv` | included | row indices of the seed-42 7:3 stratified train/test split (integers only) |

Consistent with the manuscript's Data Availability statement, **no Kaggle-derived records are redistributed in this repository**. To reproduce the cleaned cohort:

1. Download `diabetes.csv` (4,303 rows) from Kaggle: https://www.kaggle.com/datasets/pkdarabi/diabetes-dataset-with-18-features (or `kagglehub.dataset_download("pkdarabi/diabetes-dataset-with-18-features")`)
2. Place it in the repository root and run `python build_clean.py`
3. Verify the printed SHA-256 checksums against the expected values:
   - raw `diabetes.csv` (4,303 rows): `cced15f9f4aa0cd17089a2508e321f6a556e8a08b049404c5502c80fd9dd1351`
   - rebuilt `chinese_clean.csv` (4,061 rows): `aff50167d373744c64735bd270292f28b890a3579e80b17b039d45d484dccdfd`
   - positives 1,155 (28.4%); split 2,842 train / 1,219 test

The regenerated `chinese_clean.csv` is bit-identical to the one used in the analysis (same QC code path as `2_中国数据建模.py`), and `split_indices_seed42.csv` lets you confirm your split matches exactly.

`results_china/` contains model outputs and outcome labels only; no predictor values from the Kaggle source are included.

## 2. Main pipeline (run in order; working directory = repository root)

| # | Script | Manuscript section | Output |
| --- | --- | --- | --- |
| 1 | `1_Pima建模.py` | §3.2.1 / Table 1 (Pima) | `results_pima/` `figures_pima/` |
| 2 | `2_中国数据建模.py` | §3.2.2 / Table 2 / Table 5 (PUC) | `results_china/` `figures_china/` |
| 3 | `3_消融实验.py` | §3.2.5 / Table 4 / Figure 2 | `results_ablation/` |
| 4 | `4_跨数据集迁移.py` | §3.2.4 (needs 2) | `results_transfer/` |
| 5 | `5_数据可信度分析.py` | §3.6 / Table S1 | `results_credibility/` |
| 6 | `6_生成论文图表.py` | Figures 1–5 (needs 2, 3) | `figures_paper/` |
| 7 | `7_MICE敏感性分析.py` | §2.2.1 / Table S2 | `results_mice/` |
| 8–24 | `8_…py` ~ `24_…py` | supplementary checks (R1–R8, benchmark, TRIPOD checklist, docx write-back) | `results_extra/` etc. |
| **25** | **`25_OOF选模_全规则.py`** | **§2.4.1 OOF model selection, full lexicographic rule (recall≥0.92 → max F₂ → min missed → max specificity); 5-fold, everything redone inside each fold; threshold taken from pooled OOF predictions; never touches the test set** | `results_extra/OOF_select_1120/` + `logs/25_*.txt` |
| **26** | **`26_重跑_表1表2_全数值.py`** | **seed-42 rerun of Tables 1–2: all values + per-model confusion matrices + AUC 95% CI (2,000 bootstrap)** | `results_rerun_1120/` + `logs/26_*.txt` |

Shared helpers: `工具函数.py` (bootstrap CI, Brier, calibration, threshold sensitivity, benchmark) and `补充实验工具.py` (QC/feature engineering/model-grid helpers used by scripts 9–26). Neither is run standalone.

Environment: Python 3.7.3, scikit-learn 1.0.2, xgboost 1.6.2, lightgbm 4.6.0, numpy 1.21.6, pandas 1.3.5 (see logs for exact versions). Single fixed random seed 42 for splitting, CV, bootstrap and model randomness.

## 3. Reproducibility / audit artifacts (2026-11-20)

| Item | Location |
| ---- | ---- |
| Code ↔ manuscript discrepancy table (5 items + extras) | `核对报告_1120.md` |
| Old-vs-new results comparison | `新旧对比_1120.md` |
| OOF full-rule selection outputs + per-fold LASSO records | `results_extra/OOF_select_1120/` |
| Rerun Tables 1–2 with confusion matrices and AUC CIs | `results_rerun_1120/` |
| Run logs (timestamps, versions, seeds) | `logs/25_OOF选模_run_1120.txt`, `logs/26_重跑_run_1120.txt` |

Key audited points: the soft-voting ensemble has **five** members (LR/SVM/RF/XGBoost/LightGBM) in the code and in all reported numbers; model selection follows the out-of-fold lexicographic rule of §2.4.1 implemented in `25_OOF选模_全规则.py`; the seed-42 rerun reproduces the published test-set values exactly (see `新旧对比_1120.md`).

## 4. Legacy scripts

`pima-deal.py`, `chinese-deal.py`, `supplemental_analysis.py` belong to an earlier manuscript version (*Risk screening assessment of diabetes prevalence based on XGBoost and Voting*) and are kept for provenance; they are superseded by the numbered scripts above.

---

# 基于 XGBoost 与 Voting 的糖尿病患病风险筛查评估(旧稿存档)

本部分对应旧稿《基于XGBoost与Voting的糖尿病患病风险筛查评估》的 3 个脚本(`pima-deal.py`、`chinese_deal.py`、`supplemental_analysis.py`),已由上文编号脚本取代,保留备查。
