# -*- coding: utf-8 -*-
"""
4_跨数据集迁移.py（v2：评估集配对）
论文 3.2.4 —— 双向跨队列迁移实验
本次修订（2026-11-09）：
  · 方向 A 与对照 A0 统一在【同一个 PUC 测试集（n=1,219）】上评估
    （旧版 A 用全量 4,061、A0 用测试集 1,219，两底数不同，"损失 0.32" 不可比）
  · 另附 A-full 行（Pima 模型在 PUC 全量 4,061 上评估）作为补充参考
  · CSV 增加 n_eval 列标明各实验的评估样本数
运行前请先运行 2_中国数据建模.py（生成 chinese_clean.csv）
输出：results_transfer/transportability_results.csv
"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss

SEED = 42
N_BOOT = 2000
THRESH_PIMA = 0.20   # Pima 模型最优阈值（论文 表1）
THRESH_CN = 0.35     # PUC 模型最优阈值（论文 表2）
os.makedirs('results_transfer', exist_ok=True)

XGB_PIMA = dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                random_state=SEED, eval_metric='logloss')  # 2026-11-10 跟随表1 新口径 XGB 网格最优（grid_best_params.csv）
XGB_CN = dict(n_estimators=200, max_depth=5, learning_rate=0.1,
              random_state=SEED, eval_metric='logloss')
FEATS = ['Age', 'Glucose_mmol', 'BMI', 'DBP']


def bootstrap_ci(y_true, y_prob, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return tuple(np.percentile(stats, [2.5, 97.5]))


def load_pima():
    df = pd.read_csv('pima_diabetes.csv')
    for c in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[c] = df[c].replace(0, np.nan)
    out = pd.DataFrame({
        'Age': df['Age'].astype(float),
        'Glucose_mmol': df['Glucose'] / 18.02,
        'BMI': df['BMI'].astype(float),
        'DBP': df['BloodPressure'].astype(float),
        'y': df['Outcome'].astype(int)})
    return out.dropna(subset=['y']).reset_index(drop=True)


def load_chinese():
    df = pd.read_csv('chinese_clean.csv')
    out = pd.DataFrame({
        'Age': df['Age'].astype(float),
        'Glucose_mmol': df['FPG'].astype(float),
        'BMI': df['BMI'].astype(float),
        'DBP': df['DBP'].astype(float),
        'y': df['Diabetes'].astype(int)})
    return out.reset_index(drop=True)


def impute_fit(df):
    return df[['Age', 'Glucose_mmol', 'BMI', 'DBP']].median()


def impute_apply(df, med):
    df = df.copy()
    for c in ['Age', 'Glucose_mmol', 'BMI', 'DBP']:
        df[c] = df[c].fillna(med[c])
    return df


def fit_xgb(params, X, y):
    p = dict(params)
    p['scale_pos_weight'] = (y == 0).sum() / max((y == 1).sum(), 1)
    return XGBClassifier(**p).fit(X, y)


def evaluate(tag, y_true, prob, thresh):
    pred = (prob >= thresh).astype(int)
    auc = roc_auc_score(y_true, prob)
    lo, hi = bootstrap_ci(y_true, prob)
    return {'experiment': tag, 'n_eval': len(y_true), 'AUC': auc,
            'Recall': recall_score(y_true, pred),
            'Brier': brier_score_loss(y_true, prob),
            'AUC_CI': f'({lo:.3f}-{hi:.3f})'}


print("=" * 70)
print("双向跨队列迁移实验 v2（评估集配对，论文 3.2.4）")
print("=" * 70)

pima = load_pima()
cn = load_chinese()
pima_tr, pima_te = train_test_split(pima, test_size=0.3, stratify=pima['y'], random_state=SEED)
cn_tr, cn_te = train_test_split(cn, test_size=0.3, stratify=cn['y'], random_state=SEED)
print(f"PUC: 训练 {len(cn_tr)} / 测试 {len(cn_te)}；Pima: 训练 {len(pima_tr)} / 测试 {len(pima_te)}")

results = []

# ---- 训练两个来源模型 ----
med_p = impute_fit(pima_tr)
model_p = fit_xgb(XGB_PIMA, impute_apply(pima_tr, med_p)[FEATS], pima_tr['y'])
med_c = impute_fit(cn_tr)
model_c = fit_xgb(XGB_CN, impute_apply(cn_tr, med_c)[FEATS], cn_tr['y'])

# ---- A: Pima 训练 -> PUC 测试集（与 A0 同一评估集） ----
prob_a = model_p.predict_proba(impute_apply(cn_te, med_p)[FEATS])[:, 1]
results.append(evaluate('A: Pima训练 -> PUC测试集', cn_te['y'], prob_a, THRESH_PIMA))

# ---- A0 对照: PUC 训练 -> PUC 测试集（同一评估集） ----
prob_a0 = model_c.predict_proba(impute_apply(cn_te, med_c)[FEATS])[:, 1]
results.append(evaluate('A0(对照): PUC训练 -> PUC测试集', cn_te['y'], prob_a0, THRESH_CN))

# ---- A-full（补充）: Pima 训练 -> PUC 全量（无训练污染，可比性弱，仅供参考） ----
prob_af = model_p.predict_proba(impute_apply(cn, med_p)[FEATS])[:, 1]
results.append(evaluate('A-full(补充): Pima训练 -> PUC全量', cn['y'], prob_af, THRESH_PIMA))

# ---- B: PUC 训练 -> Pima 测试集（与 B0 同一评估集） ----
prob_b = model_c.predict_proba(impute_apply(pima_te, med_c)[FEATS])[:, 1]
results.append(evaluate('B: PUC训练 -> Pima测试集', pima_te['y'], prob_b, THRESH_CN))

# ---- B0 对照: Pima 训练 -> Pima 测试集（同一评估集） ----
prob_b0 = model_p.predict_proba(impute_apply(pima_te, med_p)[FEATS])[:, 1]
results.append(evaluate('B0(对照): Pima训练 -> Pima测试集', pima_te['y'], prob_b0, THRESH_PIMA))

res = pd.DataFrame(results)
res.to_csv('results_transfer/transportability_results.csv', index=False, encoding='utf-8-sig')
print(res.to_string(index=False))

loss_a = res.loc[res['experiment'].str.startswith('A0'), 'AUC'].iloc[0] - \
         res.loc[res['experiment'].str.startswith('A:'), 'AUC'].iloc[0]
print(f"\n配对比较（同为 PUC 测试集 n={len(cn_te)}）：A 相对 A0 的 AUC 损失 = {loss_a:.3f}")
print("结论：迁移存在方向不对称（论文 3.2.4）")
print("已保存: results_transfer/transportability_results.csv")
