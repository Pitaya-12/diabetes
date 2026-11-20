# -*- coding: utf-8 -*-
"""
8_推理耗时单会话复测.py —— 论文 3.7 推理耗时（同一 session 内重测）
本次修订（2026-11-09）：旧稿两次 session 的批量耗时相差 28%，超出"约20%"的说法。
本脚本在【同一个 Python 会话】内：
  1) 按主分析管线训练两个最终模型（Pima XGBoost / PUC Voting）
  2) 同一测量会话内各重复 3 轮（每轮单样本 1000 次、批量 100 样本）
     取均值±标准差，同时报告模型文件大小
输出：results_benchmark/session_benchmark.csv
"""
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
os.makedirs('results_benchmark', exist_ok=True)
SEED = 42
N_ITER = 1000      # 每轮单样本推理次数
N_REPS = 3         # 测量轮数
BATCH = 100

print("=" * 70)
print("推理耗时单会话复测（论文 3.7）")
print("=" * 70)


# ====================== Pima 管线（与 1 号脚本一致）→ XGBoost ======================
def train_pima():
    df = pd.read_csv('pima_diabetes.csv')
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, df[col].median())
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
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    sc = StandardScaler()
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000)
    lasso.fit(sc.fit_transform(Xtr), ytr)
    coef = pd.DataFrame({'feature': X.columns, 'coef': lasso.coef_})
    keep = coef[coef['coef'].abs() > 0.001]['feature'].tolist() or list(X.columns)
    Xtr, Xte = Xtr[keep], Xte[keep]
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    grid = GridSearchCV(
        xgb.XGBClassifier(eval_metric='logloss', random_state=SEED,
                          scale_pos_weight=spw, use_label_encoder=False),
        {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
        cv=5, scoring='roc_auc', n_jobs=1)
    grid.fit(Xtr, ytr)
    print(f"Pima XGBoost 训练完成（LASSO {len(keep)} 特征，"
          f"最优参数 {grid.best_params_}）")
    return grid.best_estimator_, Xte   # 树模型用未标准化特征


# ====================== PUC 管线（与 2 号脚本一致）→ Voting ======================
def train_puc():
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
    df = df.reset_index(drop=True)
    X = df.drop('Diabetes', axis=1)
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
    y = df['Diabetes']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    sc = StandardScaler()
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000)
    lasso.fit(sc.fit_transform(Xtr), ytr)
    coef = pd.DataFrame({'feature': X.columns, 'coef': lasso.coef_})
    keep = coef[coef['coef'].abs() > 0.001]['feature'].tolist() or list(X.columns)
    Xtr, Xte = Xtr[keep], Xte[keep]
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    models = {}
    g = GridSearchCV(LogisticRegression(max_iter=3000, random_state=SEED, class_weight='balanced'),
                     {'C': [0.01, 0.05, 0.1, 0.5, 1, 2]}, cv=5, scoring='roc_auc', n_jobs=1)
    g.fit(Xtr_s, ytr); models['LR'] = g.best_estimator_
    g = GridSearchCV(RandomForestClassifier(random_state=SEED, class_weight='balanced', n_jobs=1),
                     {'n_estimators': [100, 200], 'max_depth': [7, 10], 'min_samples_split': [2, 5]},
                     cv=5, scoring='roc_auc', n_jobs=1)
    g.fit(Xtr, ytr); models['RF'] = g.best_estimator_
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    g = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss', random_state=SEED,
                                       scale_pos_weight=spw, use_label_encoder=False),
                     {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
                     cv=5, scoring='roc_auc', n_jobs=1)
    g.fit(Xtr, ytr); models['XGB'] = g.best_estimator_
    g = GridSearchCV(lgb.LGBMClassifier(random_state=SEED, verbose=-1, class_weight='balanced'),
                     {'n_estimators': [100, 200], 'max_depth': [5, 7], 'learning_rate': [0.05, 0.1]},
                     cv=5, scoring='roc_auc', n_jobs=1)
    g.fit(Xtr, ytr); models['LGB'] = g.best_estimator_
    g = GridSearchCV(SVC(probability=True, random_state=SEED, class_weight='balanced'),
                     {'C': [0.5, 1, 2], 'gamma': ['scale', 'auto']}, cv=5, scoring='roc_auc', n_jobs=1)
    g.fit(Xtr_s, ytr); models['SVM'] = g.best_estimator_

    voting = VotingClassifier(estimators=[(k, v) for k, v in models.items()],
                              voting='soft', n_jobs=1)
    voting.fit(Xtr_s, ytr)
    print(f"PUC Voting 训练完成（LASSO {len(keep)} 特征）")
    return voting, Xte_s   # Voting 用标准化特征


# ====================== 同一会话内测量 ======================
def measure(model, X, name):
    single = X.iloc[0:1] if hasattr(X, 'iloc') else X[0:1]
    batch = X.iloc[:BATCH] if hasattr(X, 'iloc') else X[:BATCH]
    for _ in range(20):
        model.predict_proba(single)          # 预热
    single_reps, batch_reps = [], []
    for r in range(N_REPS):
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            model.predict_proba(single)
        single_reps.append((time.perf_counter() - t0) / N_ITER * 1000)
        t0 = time.perf_counter()
        model.predict_proba(batch)
        batch_reps.append((time.perf_counter() - t0) * 1000)
    size_mb = len(pickle.dumps(model)) / (1024 * 1024)
    row = {'model': name,
           'single_ms_mean': round(float(np.mean(single_reps)), 3),
           'single_ms_sd': round(float(np.std(single_reps)), 3),
           'batch100_ms_mean': round(float(np.mean(batch_reps)), 3),
           'batch100_ms_sd': round(float(np.std(batch_reps)), 3),
           'model_size_mb': round(size_mb, 2),
           'reps': N_REPS, 'iters_per_rep': N_ITER}
    print(f"\n[{name}] 单样本 {row['single_ms_mean']:.3f}±{row['single_ms_sd']:.3f} ms | "
          f"批量({BATCH}) {row['batch100_ms_mean']:.3f}±{row['batch100_ms_sd']:.3f} ms | "
          f"模型 {row['model_size_mb']:.2f} MB")
    return row


# ====================== 主流程：训练两个模型后立即测量 ======================
t_start = time.time()
model_p, Xte_p = train_pima()
model_c, Xte_c = train_puc()
print(f"\n训练总耗时 {time.time()-t_start:.0f} 秒。开始同一会话测量...")

rows = [measure(model_p, Xte_p, 'Pima XGBoost'),
        measure(model_c, Xte_c, 'PUC Voting')]

pd.DataFrame(rows).to_csv('results_benchmark/session_benchmark.csv', index=False)
print("\n已保存: results_benchmark/session_benchmark.csv")
print("（同一会话、同一环境一次测得；论文 3.7 应使用本组数字或如实给出波动范围）")
