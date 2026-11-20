# -*- coding: utf-8 -*-
"""
3_消融实验.py（v2：与主分析同口径）
论文 3.2.5 / 表4 / 图2 —— 消融实验
本次修订（2026-11-09）：
  · 预处理改用主分析质控（4061 行，与 2 号脚本完全一致），
    不再使用旧版"先填补后过滤"的 4303 行口径
  · LASSO 同源：LassoCV 5 折 CV 选 λ，与主分析一致
    → Version A 的特征选择应与主分析相同（25 候选中保留 22 个）
  · 模型仍为固定超参 Voting（论文已披露"固定超参、不重调"），统一阈值 0.35
输出：results_ablation/ + figures_ablation/
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('results_ablation', exist_ok=True)
os.makedirs('figures_ablation', exist_ok=True)

SEED = 42
THRESHOLD = 0.35   # 主分析 Voting 最优阈值，4 个版本统一使用

print("=" * 70)
print("消融实验 v2（与主分析同口径；Voting 固定超参，统一阈值 0.35）")
print("=" * 70)


# ====================== 预处理（与 2 号主分析脚本完全一致） ======================
def preprocess(with_feature_eng):
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
    if with_feature_eng:
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
    return X, df['Diabetes']


def lasso_select(X_train, y_train):
    """LASSO 筛选：LassoCV 5 折 CV（与主分析一致）"""
    scaler = StandardScaler()
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000)
    lasso.fit(scaler.fit_transform(X_train), y_train)
    coef = pd.DataFrame({'feature': X_train.columns, 'coef': lasso.coef_})
    keep = coef[coef['coef'].abs() > 0.001]['feature'].tolist()
    return keep if keep else list(X_train.columns), coef


def train_voting(X_train, X_test, y_train, y_test):
    """Voting 集成（基模型超参数固定、不调参），阈值 0.35"""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    voting = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=7, random_state=SEED)),
            ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                      random_state=SEED, use_label_encoder=False,
                                      eval_metric='logloss')),
            ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.1,
                                       random_state=SEED, verbose=-1)),
            ('svm', SVC(C=0.5, kernel='rbf', probability=True, random_state=SEED)),
        ], voting='soft')
    voting.fit(Xtr, y_train)

    prob = voting.predict_proba(Xte)[:, 1]
    yp = (prob > THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
    prec = precision_score(y_test, yp)
    rec = recall_score(y_test, yp)
    return {
        'auc': roc_auc_score(y_test, prob), 'recall': rec,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'miss_rate': fn / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        'f2': 5 * prec * rec / (4 * prec + rec) if (prec + rec) > 0 else 0,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


# ====================== 4 个版本 ======================
configs = [
    ('A: 完整方案(特征工程+LASSO)', True, True),
    ('B: 无特征工程(原始+LASSO)', False, True),
    ('C: 无LASSO(特征工程+全特征)', True, False),
    ('D: 基线(原始+全特征)', False, False),
]

results = []
for name, with_fe, use_lasso in configs:
    X, yy = preprocess(with_fe)
    Xtr, Xte, ytr, yte = train_test_split(
        X, yy, test_size=0.3, random_state=SEED, stratify=yy)
    if use_lasso:
        keep, coef = lasso_select(Xtr, ytr)
        Xtr, Xte = Xtr[keep], Xte[keep]
    r = train_voting(Xtr, Xte, ytr, yte)
    r.update({'version': name, 'n_features': Xtr.shape[1]})
    results.append(r)
    print(f"{name}: 特征数={Xtr.shape[1]}, AUC={r['auc']:.3f}, "
          f"召回率={r['recall']:.3f}, 漏诊率={r['miss_rate']:.1f}%")

df = pd.DataFrame(results)[['version', 'n_features', 'auc', 'recall',
                            'specificity', 'miss_rate', 'f2',
                            'tn', 'fp', 'fn', 'tp']]
df.to_csv('results_ablation/ablation_results.csv', index=False, encoding='utf-8-sig')

# ====================== 图2：消融对比图 ======================
labels = ['D\nbaseline', 'B\nno FE', 'C\nno LASSO', 'A\nfull']
order = [3, 1, 2, 0]
aucs = [df.loc[i, 'auc'] for i in order]
recs = [df.loc[i, 'recall'] for i in order]
miss = [df.loc[i, 'miss_rate'] for i in order]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(4)
w = 0.35
b1 = axes[0].bar(x - w / 2, aucs, w, label='AUC', color='#3498db', edgecolor='white')
b2 = axes[0].bar(x + w / 2, recs, w, label='Recall', color='#2ecc71', edgecolor='white')
axes[0].set_ylabel('Score')
axes[0].set_title(f'Ablation study: AUC vs Recall (threshold={THRESHOLD})')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
axes[0].legend(loc='lower right'); axes[0].set_ylim(0.80, 0.98)
axes[0].grid(True, alpha=0.2, axis='y')
for bar, v in list(zip(b1, aucs)) + list(zip(b2, recs)):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{v:.3f}', ha='center', fontsize=9)

b3 = axes[1].bar(x, miss, color=['#e74c3c', '#e67e22', '#f39c12', '#27ae60'],
                 edgecolor='white')
axes[1].set_ylabel('Miss Rate (%)')
axes[1].set_title(f'Ablation study: Miss Rate (threshold={THRESHOLD})')
axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
axes[1].set_ylim(0, 16); axes[1].grid(True, alpha=0.2, axis='y')
for bar, v in zip(b3, miss):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures_ablation/ablation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n已保存: results_ablation/ablation_results.csv, figures_ablation/ablation_comparison.png")
print("（消融各版本固定阈值 0.35、固定超参，不能与主实验表2的 F2 优化结果直接比较）")
