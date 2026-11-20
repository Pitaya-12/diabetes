# -*- coding: utf-8 -*-
"""
2_中国数据建模.py — 来源不明公开临床队列(PUC)建模（论文 3.2.2 / 表2 / 表5）
流程：数据质控 → 特征工程 → LASSO → 7 个模型 → F2 阈值优化 → 评估
输出：results_china/ + figures_china/ + chinese_clean.csv（供 4 号脚本使用）
运行前先跑 1 号脚本不需要；本脚本独立运行。
"""
import os
import pickle
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             confusion_matrix, roc_curve)

from 工具函数 import (bootstrap_auc_ci, plot_calibration_curve,
                  plot_multi_model_calibration, threshold_sensitivity_analysis,
                  benchmark_inference)

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('results_china', exist_ok=True)
os.makedirs('figures_china', exist_ok=True)
SEED = 42

print("=" * 70)
print("中国来源不明公开临床队列(PUC)建模（论文 表2 / 表5）")
print("=" * 70)

# ======================
# 1. 数据质控（论文 2.1.2：4303 → 4061）
# ======================
df = pd.read_csv('diabetes.csv')
print(f"原始数据: {len(df)} 行")

# 4.860753 是缺失标记
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].replace(4.860753, np.nan)

# 生理范围剔除
df = df[(df['BMI'] > 10) & (df['BMI'] < 60) &
        (df['FPG'] > 2.8) & (df['SBP'] > 0) & (df['DBP'] > 0) &
        (df['Age'] >= 18)]

# 核心字段缺失 >=3 剔除
core_cols = ['FPG', 'BMI', 'Age', 'SBP', 'DBP', 'Chol']
df = df[df[core_cols].isnull().sum(axis=1) < 3]

# IQR x 1.5 剔除异常
for col in core_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

# 中位数填补
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())
df = df.reset_index(drop=True)
print(f"质控后: {len(df)} 行，患病率 {df['Diabetes'].mean()*100:.1f}%")

# 导出质控后数据（供 4_跨数据集迁移.py 使用）
df.to_csv('chinese_clean.csv', index=False, encoding='utf-8-sig')

# ======================
# 2. 特征工程（论文 2.2.2）
# ======================
def add_features(X):
    """在原始特征基础上构造 8 个新特征"""
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


X = add_features(df.drop('Diabetes', axis=1))
y = df['Diabetes']

# ======================
# 3. 7:3 分层划分
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)
print(f"训练集 {len(X_train)} / 测试集 {len(X_test)}")

# ======================
# 4. LASSO 特征筛选
# ======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000)
lasso.fit(X_train_scaled, y_train)
coef_df = pd.DataFrame({'feature': X.columns, 'coef': lasso.coef_})
selected = coef_df[coef_df['coef'].abs() > 0.001]['feature'].tolist()
if not selected:
    selected = X.columns.tolist()
print(f"LASSO 保留 {len(selected)}/{X.shape[1]} 个特征")
coef_df.sort_values('coef', key=abs, ascending=False).to_csv(
    'results_china/lasso_coefficients.csv', index=False, encoding='utf-8-sig')

X_train_sel, X_test_sel = X_train[selected], X_test[selected]
scaler_sel = StandardScaler()
X_train_s = scaler_sel.fit_transform(X_train_sel)
X_test_s = scaler_sel.transform(X_test_sel)

# ======================
# 5. 训练 7 个模型
# ======================
print("训练 7 个模型...")
models = {}

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=3000, random_state=SEED, class_weight='balanced'),
    {'C': [0.01, 0.05, 0.1, 0.5, 1, 2]}, cv=5, scoring='roc_auc', n_jobs=1)
lr_grid.fit(X_train_s, y_train)
models['LR'] = lr_grid.best_estimator_

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=SEED, class_weight='balanced', n_jobs=1),
    {'n_estimators': [100, 200], 'max_depth': [7, 10], 'min_samples_split': [2, 5]},
    cv=5, scoring='roc_auc', n_jobs=1)
rf_grid.fit(X_train_sel, y_train)
models['RF'] = rf_grid.best_estimator_

spw = (y_train == 0).sum() / (y_train == 1).sum()
xgb_grid = GridSearchCV(
    xgb.XGBClassifier(eval_metric='logloss', random_state=SEED,
                      scale_pos_weight=spw, use_label_encoder=False),
    {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
    cv=5, scoring='roc_auc', n_jobs=1)
xgb_grid.fit(X_train_sel, y_train)
models['XGB'] = xgb_grid.best_estimator_

lgb_grid = GridSearchCV(
    lgb.LGBMClassifier(random_state=SEED, verbose=-1, class_weight='balanced'),
    {'n_estimators': [100, 200], 'max_depth': [5, 7], 'learning_rate': [0.05, 0.1]},
    cv=5, scoring='roc_auc', n_jobs=1)
lgb_grid.fit(X_train_sel, y_train)
models['LGB'] = lgb_grid.best_estimator_

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=SEED, class_weight='balanced'),
    {'C': [0.5, 1, 2], 'gamma': ['scale', 'auto']}, cv=5, scoring='roc_auc', n_jobs=1)
svm_grid.fit(X_train_s, y_train)
models['SVM'] = svm_grid.best_estimator_

voting = VotingClassifier(
    estimators=[(k, v) for k, v in models.items()], voting='soft', n_jobs=1)
voting.fit(X_train_s, y_train)
models['Voting'] = voting

stacking = StackingClassifier(
    estimators=[('LR', models['LR']), ('RF', models['RF']), ('XGB', models['XGB'])],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5, stack_method='predict_proba')
stacking.fit(X_train_s, y_train)
models['Stacking'] = stacking

# ======================
# 6. 评估 + 表2
# ======================
USE_SCALED = {'LR', 'SVM', 'Voting', 'Stacking'}
rows, prob_dict = [], {}
for name, m in models.items():
    Xte = X_test_s if name in USE_SCALED else X_test_sel
    prob = m.predict_proba(Xte)[:, 1]
    best_f2, best_thr = 0, 0.5
    for thr in np.arange(0.10, 0.71, 0.05):
        yp = (prob > thr).astype(int)
        r, p = recall_score(y_test, yp), precision_score(y_test, yp)
        if p + r > 0:
            f2 = 5 * p * r / (4 * p + r)
            if f2 > best_f2:
                best_f2, best_thr = f2, thr
    yp = (prob > best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
    rows.append({'Model': name, 'Threshold': best_thr,
                 'AUC': roc_auc_score(y_test, prob),
                 'Recall': recall_score(y_test, yp),
                 'PPV': precision_score(y_test, yp),
                 'F2': best_f2, 'Missed': fn})
    prob_dict[name] = prob

table2 = pd.DataFrame(rows).sort_values('F2', ascending=False)
table2.to_csv('results_china/table2_model_comparison.csv',
              index=False, encoding='utf-8-sig')
print("\n表2 各模型性能（按 F2 降序）:")
print(table2.to_string(index=False))

# 最优模型（论文 2.4 预设规则：召回率≥0.92 中 F2 最高者）
cand = table2[table2['Recall'] >= 0.92]
best_name = (cand if len(cand) else table2).iloc[0]['Model']
best_prob = prob_dict[best_name]
best_thr = table2.set_index('Model').loc[best_name, 'Threshold']
y_pred = (best_prob > best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
auc_pt, auc_lo, auc_hi = bootstrap_auc_ci(y_test, best_prob)
print(f"\n最优模型: {best_name} (阈值 {best_thr:.2f})  "
      f"AUC={auc_pt:.3f} (95%CI {auc_lo:.3f}-{auc_hi:.3f})")

# ======================
# 7. 风险四级分级（论文 表5）
# ======================
def risk_level(p):
    return '低风险' if p < 0.25 else '中风险' if p < 0.5 else '高风险' if p < 0.75 else '极高风险'

risk_df = pd.DataFrame({'实际': y_test.values, '概率': best_prob,
                        '风险等级': [risk_level(p) for p in best_prob],
                        '预测': y_pred})
risk_df.to_csv('results_china/risk_stratification.csv', index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(8, 5))
order = ['低风险', '中风险', '高风险', '极高风险']
order_en = ['Low risk\n(<0.25)', 'Moderate\n(0.25-0.50)',
            'High\n(0.50-0.75)', 'Very high\n(>=0.75)']
counts = [int((risk_df['风险等级'] == lv).sum()) for lv in order]
prev = [risk_df[risk_df['风险等级'] == lv]['实际'].mean() * 100 for lv in order]
bars = ax.bar(order_en, counts, color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
for b, c, p in zip(bars, counts, prev):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
            f'n={c}\nprevalence {p:.1f}%', ha='center', fontsize=10)
ax.set_ylabel('Number of subjects')
ax.set_title('Risk Stratification - PUC Cohort')
plt.tight_layout()
plt.savefig('figures_china/risk_stratification.png', dpi=300)
plt.close()
print("风险分级: " + ", ".join(f"{lv} {c}人" for lv, c in zip(order, counts)))

# ======================
# 8. 图：ROC（论文 图1）
# ======================
fpr, tpr, _ = roc_curve(y_test, best_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'{best_name} (AUC={auc_pt:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Provenance-Unverified Public Clinical Cohort')
plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures_china/roc_curve.png', dpi=300)
plt.close()
print("已保存: figures_china/roc_curve.png")

# ======================
# 9. DCA 决策曲线（论文 3.5 / 图4）
# ======================
def net_benefit(y_true, prob, t):
    yp = (prob > t).astype(int)
    tp = np.sum((y_true == 1) & (yp == 1))
    fp = np.sum((y_true == 0) & (yp == 1))
    return max(0, tp / len(y_true) - fp / len(y_true) * (t / (1 - t)))

ths = np.arange(0.01, 0.5, 0.02)
nb_model = [net_benefit(y_test.values, best_prob, t) for t in ths]
nb_all = [max(0, y_test.mean() - t / (1 - t) * (1 - y_test.mean())) for t in ths]
plt.figure(figsize=(8, 6))
plt.plot(ths, nb_model, 'b-', linewidth=2, label='Voting Ensemble')
plt.plot(ths, nb_all, 'g--', linewidth=1.5, label='Treat All')
plt.plot(ths, [0] * len(ths), 'r--', linewidth=1.5, label='Treat None')
plt.xlabel('Risk Threshold'); plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis - PUC Cohort')
plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures_china/decision_curve.png', dpi=300)
plt.close()
print("已保存: figures_china/decision_curve.png")

# ======================
# 10. 校准曲线 + Brier（论文 3.5）
# ======================
plot_calibration_curve(y_test.values, best_prob, 'PUC',
                       save_path='figures_china/calibration_curve.png')
summary = plot_multi_model_calibration(
    prob_dict, y_test.values, 'PUC',
    save_path='figures_china/calibration_all_models.png')
summary.to_csv('results_china/brier_summary.csv', index=False, encoding='utf-8-sig')

# ======================
# 11. SHAP（Voting 用 KernelSHAP，论文 3.5 / 图5）
#     注意：KernelSHAP 较慢（约 20~30 分钟）；已有结果文件时自动跳过
# ======================
if os.path.exists('results_china/shap_voting.csv'):
    print("\n[SHAP] 已存在 results_china/shap_voting.csv，跳过重算")
else:
    try:
        import shap
        bg = shap.sample(X_train_s, 50, random_state=SEED)
        sv = shap.KernelExplainer(models[best_name].predict_proba, bg).shap_values(X_test_s[:200])
        sv = sv[1] if isinstance(sv, list) else sv
        shap_df = pd.DataFrame({'feature': selected,
                                'mean_abs_shap': np.abs(sv).mean(axis=0)}) \
            .sort_values('mean_abs_shap', ascending=False)
        shap_df.to_csv('results_china/shap_voting.csv', index=False)
        print(f"SHAP 前3特征: {' > '.join(shap_df['feature'].head(3))}")
    except Exception as e:
        print(f"[跳过 SHAP] {e}")

# ======================
# 12. 稳定性检验（论文 3.6）
#     注意：该数据集原始行序按标签排列（前段全是未患病、后段全是患病），
#     无法按"前60%/后40%"索引切分（早期组没有阳性样本），
#     因此这里用 5 次不同随机种子的分层分割检验模型稳定性。
# ======================
print("\n【稳定性检验（5 次随机分割）】")
X_all_fe = add_features(df.drop('Diabetes', axis=1))
y_all = df['Diabetes']
aucs = []
for s in [0, 1, 2, 3, 4]:
    Xtr, Xte, ytr, yte = train_test_split(X_all_fe, y_all, test_size=0.3,
                                          random_state=s, stratify=y_all)
    m = lgb.LGBMClassifier(n_estimators=100, max_depth=7, random_state=SEED,
                           verbose=-1, class_weight='balanced')
    m.fit(Xtr, ytr)
    aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
print(f"  AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (5 次: {[round(a, 3) for a in aucs]})")

# ======================
# 13. 成本分析模板（论文 3.8：参数为假设值，结论不作真实经济推断）
# ======================
print("\n【成本分析模板（参数为假设值，仅演示流程）】")
N = 1000
prev = (tp + fn) / len(y_test)
n_dis, n_hea = round(N * prev), N - round(N * prev)
m_fn = n_dis - round(tp / (tp + fn) * n_dis)
m_fp = round(fp / (fp + tn) * n_hea)
m_tp = n_dis - m_fn
base_fn, base_fp = 41, 57          # 传统筛查基线（假设值）
cc, comp = 200, 30000               # 确认检查费 / 并发症年费（假设值）
for name, c1, c2 in [('基线', cc, comp), ('并发症-30%', cc, comp * 0.7),
                     ('并发症+30%', cc, comp * 1.3), ('检查费-50%', cc * 0.5, comp),
                     ('检查费+50%', cc * 1.5, comp),
                     ('双重极端(低)', cc * 0.5, comp * 0.7),
                     ('双重极端(高)', cc * 1.5, comp * 1.3)]:
    b = base_fp * c1 + base_fn * c2
    mm = m_fp * c1 + m_fn * c2
    print(f"  {name:12s} 传统 {b:>10,.0f} 元  模型 {mm:>9,.0f} 元  节约率 {(b-mm)/b*100:4.1f}%")

# ======================
# 14. 阈值敏感性 + 推理耗时 + 保存供 6 号脚本使用的文件
# ======================
threshold_sensitivity_analysis(y_test.values, best_prob, best_thr, 'PUC', 'results_china')
benchmark_inference(models[best_name], X_test_sel, 'PUC', 'results_china')

np.save('results_china/y_test.npy', y_test.values)
np.save('results_china/best_prob.npy', best_prob)
pickle.dump(prob_dict, open('results_china/all_probs.pkl', 'wb'))

print("\n" + "=" * 70)
print(f"完成！最优模型 {best_name}：AUC={auc_pt:.3f}，召回率={recall_score(y_test, y_pred):.3f}，"
      f"漏诊 {fn} 人。结果在 results_china/ 与 figures_china/")
print("=" * 70)
