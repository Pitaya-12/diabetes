# Risk screening assessment of diabetes prevalence based on XGBoost and Voting

This repository contains all analysis scripts, trained model files and figure‑generation code for the manuscript:
> Risk screening assessment of diabetes prevalence based on XGBoost and Voting: a methodological evaluation with SHAP interpretability across two public cohorts.

⚠️ **Important Note about Datasets**
Due to copyright and platform license restrictions, we **cannot redistribute raw patient datasets in this repository**.
1. **Pima‑Indians Diabetes Dataset**: Original UCI archive has been removed. Archived de‑identified backup data can be obtained upon reasonable request referring to its original publication.
2. **Chinese clinical diabetes dataset (Kaggle)**: Access dataset via:
`kagglehub.dataset_download("pkdarabi/diabetes‑dataset‑with‑18‑features")`

## Repository Content
- `model_training/`: Main pipeline for 7 machine‑learning models (LR, SVM, RF, XGBoost, LightGBM, Soft‑Voting, Stacking)
- `feature_engineering.py`: Feature engineering pipeline & LASSO feature selection
- `supplemental_analysis.py`: Index‑based stability test, subgroup analysis, ablation study
- `plot_scripts/`: Scripts for plotting ROC, DCA, SHAP feature importance, ablation comparison figures
- `trained_models/`: Saved fitted model files (Pima‑XGBoost, Chinese‑cohort Voting etc.)

## Environment Requirements
python >= 3.9
scikit‑learn
xgboost
lightgbm
shap
pandas
numpy
imbalanced‑learn
matplotlib

## Reproduce workflow
1. Download two raw datasets from sources mentioned above
2. Put dataset files into local `data/` directory
3. Run main pipeline: `python model_training/main.py`

> All experiments adopt fixed random seed for full reproducibility.
> Ablation experiments use fixed classification threshold = 0.35.
