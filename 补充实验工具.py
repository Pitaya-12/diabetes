# -*- coding: utf-8 -*-
"""
补充实验工具.py — R1~R4 补充实验共用函数（被 9~12 号脚本调用，不用单独运行）
与 1_Pima建模.py / 2_中国数据建模.py 的主分析管线【同口径】：
质控 → 特征工程 → 7:3 分层划分(种子42) → LASSO(仅训练集) → 5 模型网格搜索 → Voting/Stacking
本模块不写死任何结果数字，全部由运行时计算。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             confusion_matrix)
import xgboost as xgb
import lightgbm as lgb

SEED = 42
THR_GRID = np.arange(0.10, 0.71, 0.05)   # 与主分析一致的 F2 阈值搜索网格
USE_SCALED = {'LR', 'SVM', 'Voting', 'Stacking'}   # 与主分析一致


# ============================================================
# 1. 质控与特征工程（逐行复刻 2_中国数据建模.py / 1_Pima建模.py）
# ============================================================
def qc_puc():
    """PUC 质控：4303 -> 4061（与 2 号脚本完全一致）"""
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


def add_features_puc(X):
    """PUC 特征工程：8 个构造特征（与 2 号脚本完全一致）"""
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


def qc_pima(split_seed=42):
    """Pima 清洗（与 1 号脚本完全一致），返回含构造特征的 X、y。
    口径修正 2026-11-10：0 值视作缺失，以【非零中位数】填补，且中位数只在
    训练集（7:3 分层、split_seed，默认 42 与主分析一致）上估计后应用到全量。
    调用方（9/18/19 号）随后的 train_test_split(split_seed=42) 与这里的训练行完全一致。"""
    df = pd.read_csv('pima_diabetes.csv')
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    idx_train, _ = train_test_split(df.index, test_size=0.3, random_state=split_seed,
                                    stratify=df['Outcome'])
    for col in zero_cols:
        med = df.loc[idx_train, col]
        df[col] = df[col].replace(0, med[med != 0].median())
    df = df[(df['BloodPressure'] > 0) & (df['BMI'] > 10)]

    df['Glucose_BMI'] = df['Glucose'] * df['BMI'] / 100
    df['Age_Pregnancies'] = df['Age'] * df['Pregnancies'] / 10
    df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1)
    df['BMI_Age'] = df['BMI'] * df['Age'] / 100
    df['SkinThickness_BMI'] = df['SkinThickness'] / (df['BMI'] + 1)
    df['Metabolic_Risk'] = ((df['Glucose'] > 99).astype(int) * 2 +
                            (df['BMI'] > 30).astype(int) * 2 +
                            (df['BloodPressure'] > 80).astype(int) +
                            (df['Age'] > 40).astype(int))
    df['Insulin_Log'] = np.log1p(df['Insulin'])
    df['DPF_Log'] = np.log1p(df['DiabetesPedigreeFunction'])
    return df.drop('Outcome', axis=1), df['Outcome']


# ============================================================
# 2. LASSO 特征筛选（与主分析一致：LassoCV 内部 5 折 CV，仅拟合传入数据）
# ============================================================
def lasso_select(Xtr, ytr):
    """在 Xtr 上做标准化 + LassoCV(cv=5) 筛选；返回保留特征名列表"""
    sc = StandardScaler()
    Xtr_s0 = sc.fit_transform(Xtr)
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000)
    lasso.fit(Xtr_s0, ytr)
    coef = pd.DataFrame({'feature': Xtr.columns, 'coef': lasso.coef_})
    keep = coef[coef['coef'].abs() > 0.001]['feature'].tolist()
    if not keep:
        keep = Xtr.columns.tolist()
    return keep, coef


# ============================================================
# 3. 5 个基模型网格搜索（网格与 1/2 号脚本完全一致）
# ============================================================
def train_base_models(Xtr_raw, Xtr_s, ytr, cohort):
    """返回 {'LR':…, 'RF':…, 'XGB':…, 'LGB':…, 'SVM':…}（均为网格搜索后的最优估计器）
    cohort: 'pima' 或 'puc'（LGB 网格与主分析对应脚本一致）"""
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
    rf_grid.fit(Xtr_raw, ytr)
    models['RF'] = rf_grid.best_estimator_

    spw = (ytr == 0).sum() / (ytr == 1).sum()
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(eval_metric='logloss', random_state=SEED,
                          scale_pos_weight=spw, use_label_encoder=False),
        {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
        cv=5, scoring='roc_auc', n_jobs=1)
    xgb_grid.fit(Xtr_raw, ytr)
    models['XGB'] = xgb_grid.best_estimator_

    if cohort == 'pima':
        lgb_space = {'n_estimators': [100, 200], 'max_depth': [5, 7]}
    else:
        lgb_space = {'n_estimators': [100, 200], 'max_depth': [5, 7],
                     'learning_rate': [0.05, 0.1]}
    lgb_grid = GridSearchCV(
        lgb.LGBMClassifier(random_state=SEED, verbose=-1, class_weight='balanced'),
        lgb_space, cv=5, scoring='roc_auc', n_jobs=1)
    lgb_grid.fit(Xtr_raw, ytr)
    models['LGB'] = lgb_grid.best_estimator_

    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=SEED, class_weight='balanced'),
        {'C': [0.5, 1, 2], 'gamma': ['scale', 'auto']}, cv=5, scoring='roc_auc', n_jobs=1)
    svm_grid.fit(Xtr_s, ytr)
    models['SVM'] = svm_grid.best_estimator_
    return models


def build_voting(models, Xtr_s, ytr):
    """软投票（5 个基模型等权，与主分析一致），拟合于标准化特征"""
    voting = VotingClassifier(
        estimators=[(k, v) for k, v in models.items()], voting='soft', n_jobs=1)
    voting.fit(Xtr_s, ytr)
    return voting


def build_stacking(models, Xtr_s, ytr):
    """Stacking（LR/RF/XGB 基学习器 + LR 元学习器，与主分析一致）"""
    stacking = StackingClassifier(
        estimators=[('LR', models['LR']), ('RF', models['RF']), ('XGB', models['XGB'])],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5, stack_method='predict_proba')
    stacking.fit(Xtr_s, ytr)
    return stacking


# ============================================================
# 4. 预测与评估（阈值规则与主分析一致：0.10~0.70 步长 0.05 最大化 F2，prob > thr）
# ============================================================
def predict_all(models, voting, stacking, X_raw, X_s):
    """返回 {模型名: 阳性概率数组}，顺序 LR/RF/XGB/LGB/SVM/Voting/Stacking"""
    probs = {}
    for name, m in models.items():
        Xte = X_s if name in USE_SCALED else X_raw
        probs[name] = m.predict_proba(Xte)[:, 1]
    probs['Voting'] = voting.predict_proba(X_s)[:, 1]
    probs['Stacking'] = stacking.predict_proba(X_s)[:, 1]
    return probs


def f2_best_threshold(y, prob):
    """在给定概率上扫描 THR_GRID，返回 (最优阈值, 该阈值下 F2)。与主分析同规则。"""
    best_f2, best_thr = 0, 0.5
    for thr in THR_GRID:
        yp = (prob > thr).astype(int)
        r, p = recall_score(y, yp), precision_score(y, yp)
        if p + r > 0:
            f2 = 5 * p * r / (4 * p + r)
            if f2 > best_f2:
                best_f2, best_thr = f2, thr
    return best_thr, best_f2


def metrics_at(y, prob, thr):
    """给定阈值计算完整指标（含混淆矩阵）"""
    yp = (prob > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    rec = recall_score(y, yp)
    prec = precision_score(y, yp)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f2 = 5 * prec * rec / (4 * prec + rec) if (prec + rec) > 0 else 0
    return {'Threshold': float(thr), 'AUC': roc_auc_score(y, prob),
            'Recall': rec, 'Specificity': spec, 'PPV': prec, 'F2': f2,
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}


# ============================================================
# 5. 一次完整管线（主分析同口径），供 R2/R3 调用
# ============================================================
def run_pipeline(X_all, y_all, split_seed, cohort, drop_cols=None,
                 want_stacking=True):
    """完整主分析管线：划分 -> LASSO -> 网格搜索 -> Voting(+Stacking)
    split_seed: 划分随机种子（主分析=42；R3 用 0~4）
    drop_cols:  剔除的候选特征（R2 去糖敏感性用），在划分前从候选集中删除
    返回 dict：含所选特征、各模型概率（测试集）、最优阈值等
    """
    X = X_all.drop(columns=[c for c in (drop_cols or []) if c in X_all.columns])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_all, test_size=0.3, random_state=split_seed, stratify=y_all)

    keep, coef = lasso_select(Xtr, ytr)
    Xtr_raw, Xte_raw = Xtr[keep], Xte[keep]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_raw)
    Xte_s = scaler.transform(Xte_raw)

    models = train_base_models(Xtr_raw, Xtr_s, ytr, cohort)
    voting = build_voting(models, Xtr_s, ytr)
    stacking = build_stacking(models, Xtr_s, ytr) if want_stacking else None

    probs = {}
    for name, m in models.items():
        Xv = Xte_s if name in USE_SCALED else Xte_raw
        probs[name] = m.predict_proba(Xv)[:, 1]
    probs['Voting'] = voting.predict_proba(Xte_s)[:, 1]
    if stacking is not None:
        probs['Stacking'] = stacking.predict_proba(Xte_s)[:, 1]

    return {'y_test': yte, 'probs': probs, 'keep': keep, 'coef': coef,
            'n_train': len(ytr), 'n_test': len(yte),
            'Xtr_raw': Xtr_raw, 'Xtr_s': Xtr_s, 'ytr': ytr,
            'Xte_raw': Xte_raw, 'Xte_s': Xte_s,
            'models': models, 'voting': voting, 'stacking': stacking}
