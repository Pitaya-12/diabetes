# -*- coding: utf-8 -*-
"""
5_数据可信度分析.py（v2：与主分析同口径）
论文 3.6 / 附表S1 —— 数据可信度敏感性分析
本次修订（2026-11-09）：
  · 对照组(未过滤)与过滤组使用【完全相同的主分析管线】
    （质控→特征工程→7:3分层划分(种子42)→LASSO→5模型网格搜索→Voting）
  · 两组统一使用主分析的最优阈值 0.35（不再各自调阈值）
  · 对照组应精确复现主分析结果：AUC 0.965 / 召回 92.5% / 特异度 88.3%
输出：results_credibility/
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             confusion_matrix)

from 工具函数 import bootstrap_auc_ci

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('results_credibility', exist_ok=True)

SEED = 42
THRESHOLD = 0.35        # 主分析 Voting 最优阈值，两组统一使用
SUSPICIOUS_MIN = 20     # FPG/BMI 取值出现 >20 次即标记可疑（论文 3.6 规则）

print("=" * 70)
print("数据可信度敏感性分析（v2：与主分析同口径，阈值固定 0.35）")
print("=" * 70)


# ====================== 1. 质控（与 2 号主分析脚本完全一致） ======================
def qc_full():
    df = pd.read_csv('diabetes.csv')
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].replace(4.860753, np.nan)
    df = df[(df['BMI'] > 10) & (df['BMI'] < 60) &
            (df['FPG'] > 2.8) & (df['SBP'] > 0) & (df['DBP'] > 0) &
            (df['Age'] >= 18)]
    core_cols = ['FPG', 'BMI', 'Age', 'SBP', 'DBP', 'Chol']
    df = df[df[core_cols].isnull().sum(axis=1) < 3]
    for col in core_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)


def add_features(X):
    X = X.copy()
    X['Glucose_BMI'] = X['FPG'] * X['BMI'] / 100
    X['Age_Gender'] = X['Age'] * X['Gender'] / 10
    X['SBP_DBP_Ratio'] = X['SBP'] / (X['DBP'] + 1)
    X['BMI_Age'] = X['BMI'] * X['Age'] / 100
    X['HDL_LDL_Ratio'] = X['HDL'] / (X['LDL'] + 1)
    X['Metabolic_Risk'] = ((X['FPG'] > 5.6).astype(int) * 2 +
                           (X['BMI'] > 28).astype(int) * 2 +
                           (X['SBP'] > 130).astype(int) +
                           (X['Age'] > 50).astype(int))
    X['Chol_Log'] = np.log1p(X['Chol'])
    X['Tri_Log'] = np.log1p(X['Tri'])
    return X


# ====================== 2. 主分析建模管线（与 2 号脚本完全一致） ======================
def run_main_voting(df, tag):
    """质控后数据 -> 特征工程 -> 7:3(种子42) -> LASSO -> 5模型网格搜索 -> Voting
    阈值固定 0.35（主分析最优阈值）"""
    X = add_features(df.drop('Diabetes', axis=1))
    y = df['Diabetes'].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)

    # LASSO（仅用训练集）
    sc = StandardScaler()
    Xtr_s0 = sc.fit_transform(Xtr)
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000)
    lasso.fit(Xtr_s0, ytr)
    coef = pd.DataFrame({'feature': X.columns, 'coef': lasso.coef_})
    keep = coef[coef['coef'].abs() > 0.001]['feature'].tolist() or list(X.columns)
    Xtr, Xte = Xtr[keep], Xte[keep]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # 5 个基模型网格搜索（网格与 2 号脚本一致）
    models = {}
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=3000, random_state=SEED, class_weight='balanced'),
        {'C': [0.01, 0.05, 0.1, 0.5, 1, 2]}, cv=5, scoring='roc_auc', n_jobs=1)
    lr_grid.fit(Xtr_s, ytr)
    models['LR'] = lr_grid.best_estimator_

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=SEED, class_weight='balanced', n_jobs=1),
        {'n_estimators': [100, 200], 'max_depth': [7, 10], 'min_samples_split': [2, 5]},
        cv=5, scoring='roc_auc', n_jobs=1)
    rf_grid.fit(Xtr, ytr)
    models['RF'] = rf_grid.best_estimator_

    spw = (ytr == 0).sum() / (ytr == 1).sum()
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(eval_metric='logloss', random_state=SEED,
                          scale_pos_weight=spw, use_label_encoder=False),
        {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
        cv=5, scoring='roc_auc', n_jobs=1)
    xgb_grid.fit(Xtr, ytr)
    models['XGB'] = xgb_grid.best_estimator_

    lgb_grid = GridSearchCV(
        lgb.LGBMClassifier(random_state=SEED, verbose=-1, class_weight='balanced'),
        {'n_estimators': [100, 200], 'max_depth': [5, 7], 'learning_rate': [0.05, 0.1]},
        cv=5, scoring='roc_auc', n_jobs=1)
    lgb_grid.fit(Xtr, ytr)
    models['LGB'] = lgb_grid.best_estimator_

    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=SEED, class_weight='balanced'),
        {'C': [0.5, 1, 2], 'gamma': ['scale', 'auto']}, cv=5, scoring='roc_auc', n_jobs=1)
    svm_grid.fit(Xtr_s, ytr)
    models['SVM'] = svm_grid.best_estimator_

    # Voting（与主分析一致：5 个网格搜索后的基模型等权软投票，训练于标准化特征）
    voting = VotingClassifier(
        estimators=[(k, v) for k, v in models.items()], voting='soft', n_jobs=1)
    voting.fit(Xtr_s, ytr)

    prob = voting.predict_proba(Xte_s)[:, 1]
    yp = (prob > THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
    auc_pt, auc_lo, auc_hi = bootstrap_auc_ci(yte, prob)
    prec = precision_score(yte, yp)
    rec = recall_score(yte, yp)
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0

    print(f"\n[{tag}]")
    print(f"  n_train={len(Xtr)}, n_test={len(Xte)}, LASSO 保留 {len(keep)}/{X.shape[1]} 特征")
    print(f"  AUC={auc_pt:.3f} (95%CI {auc_lo:.3f}-{auc_hi:.3f}), 召回率={rec:.3f}, "
          f"特异度={tn/(tn+fp):.3f}, PPV={prec:.3f}, 漏诊={fn}")
    return {'experiment': tag, 'n_train': len(Xtr), 'n_test': len(Xte),
            'threshold': THRESHOLD, 'AUC': round(auc_pt, 4),
            'AUC_CI': f'({auc_lo:.3f}-{auc_hi:.3f})', 'Recall': round(rec, 4),
            'Specificity': round(tn / (tn + fp), 4), 'PPV': round(prec, 4),
            'F2': round(f2, 4), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}


# ====================== 3. 数据准备：质控 + 去重 + 可疑标记 ======================
df = qc_full()
print(f"\n质控后 {len(df)} 行")
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print(f"全行重复 {n_dup} 条，去重后 {len(df)} 行")

# 附表S1：高频重复取值
rows = []
for label, c in [('FPG', 'FPG'), ('BMI', 'BMI')]:
    for val, cnt in df[c].value_counts().head(10).items():
        rows.append({'indicator': label, 'value': val, 'count': int(cnt),
                     'percent': f'{100 * cnt / len(df):.2f}%'})
pd.DataFrame(rows).to_csv('results_credibility/table_S2_frequent_values.csv',
                          index=False, encoding='utf-8-sig')

# 可疑规则：FPG 或 BMI 取值出现 > 20 次
mask = df['FPG'].isin(df['FPG'].value_counts()[lambda s: s > SUSPICIOUS_MIN].index) | \
       df['BMI'].isin(df['BMI'].value_counts()[lambda s: s > SUSPICIOUS_MIN].index)
print(f"可疑样本 {mask.sum()} 条 ({100 * mask.mean():.1f}%)，剩余 {int((~mask).sum())} 条")
df_clean = df[~mask].reset_index(drop=True)

# ====================== 4. 两组同口径对比 ======================
res_ctrl = run_main_voting(df, 'original (deduped, main pipeline)')
res_filt = run_main_voting(df_clean, 'suspicious removed (main pipeline)')

comp = pd.DataFrame([res_ctrl, res_filt])
comp.to_csv('results_credibility/credibility_sensitivity_results.csv',
            index=False, encoding='utf-8-sig')

dauc = res_ctrl['AUC'] - res_filt['AUC']
drec = (res_ctrl['Recall'] - res_filt['Recall']) * 100
print(f"\n===== 同口径对比（阈值均为主分析最优 0.35）=====")
print(comp.to_string(index=False))
print(f"\nAUC 下降 {dauc:.3f}；召回率下降 {drec:.1f} 个百分点")
print("（对照组应精确复现主分析：AUC 0.965 / 召回 92.5% / 特异度 88.3%）")
print("\n已保存: results_credibility/credibility_sensitivity_results.csv")
