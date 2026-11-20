# -*- coding: utf-8 -*-
"""
1_Pima建模.py — Pima 印第安人队列建模（论文 3.2.1 / 表1）
流程：数据清洗 → 特征工程 → LASSO 筛选 → 7 个模型 → F2 阈值优化 → 评估
输出：results_pima/ (表格) + figures_pima/ (ROC、DCA、校准、SHAP、风险分级)
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             confusion_matrix, roc_curve, auc)

from 工具函数 import (bootstrap_auc_ci, plot_calibration_curve,
                  plot_multi_model_calibration, threshold_sensitivity_analysis,
                  benchmark_inference)

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('results_pima', exist_ok=True)
os.makedirs('figures_pima', exist_ok=True)
SEED = 42

print("=" * 70)
print("Pima 队列建模（论文 表1）")
print("=" * 70)

# ======================
# 1. 数据清洗（论文 2.2.1：0 值视作缺失，非零中位数填补）
#    口径修正 2026-11-10：中位数只在训练集上估计（防泄漏），且按非零值计算
#    （旧口径 replace(0, 含0中位数) 会把 Insulin 近半样本填成 30.5，明显失真）
# ======================
df = pd.read_csv('pima_diabetes.csv')
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# 先按主分析同款 7:3 分层划分定位训练行（与下方 train_test_split 选出完全同一批行）
_idx_train, _ = train_test_split(df.index, test_size=0.3, random_state=SEED,
                                 stratify=df['Outcome'])
for col in zero_cols:
    _med = df.loc[_idx_train, col]
    df[col] = df[col].replace(0, _med[_med != 0].median())
df = df[(df['BloodPressure'] > 0) & (df['BMI'] > 10)]

# ======================
# 2. 特征工程（论文 2.2.2）
# ======================
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
print(f"样本数: {len(df)}，特征数: {X.shape[1]}")

# ======================
# 3. 7:3 分层划分（论文 2.2.3）
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)

# ======================
# 4. LASSO 特征筛选（论文 2.3）
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
coef_df.to_csv('results_pima/lasso_coefficients.csv', index=False, encoding='utf-8-sig')

X_train_sel = X_train[selected]
X_test_sel = X_test[selected]
scaler_sel = StandardScaler()
X_train_s = scaler_sel.fit_transform(X_train_sel)
X_test_s = scaler_sel.transform(X_test_sel)

# ======================
# 5. 训练 7 个模型（论文 2.4）
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
    {'n_estimators': [100, 200], 'max_depth': [5, 7]}, cv=5, scoring='roc_auc', n_jobs=1)
lgb_grid.fit(X_train_sel, y_train)
models['LGB'] = lgb_grid.best_estimator_

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=SEED, class_weight='balanced'),
    {'C': [0.5, 1, 2], 'gamma': ['scale', 'auto']}, cv=5, scoring='roc_auc', n_jobs=1)
svm_grid.fit(X_train_s, y_train)
models['SVM'] = svm_grid.best_estimator_

# 保存网格搜索最优超参（4 号迁移脚本等按表1 联动处使用）
pd.DataFrame([{'model': n, 'best_params': str(g.best_params_)}
              for n, g in [('LR', lr_grid), ('RF', rf_grid), ('XGB', xgb_grid),
                           ('LGB', lgb_grid), ('SVM', svm_grid)]]
             ).to_csv('results_pima/grid_best_params.csv', index=False, encoding='utf-8-sig')

# 集成：软投票 + Stacking
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
# 6. 评估：每模型找 F2 最优阈值（论文 2.4 预设规则）
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

table1 = pd.DataFrame(rows).sort_values('F2', ascending=False)
table1.to_csv('results_pima/table1_model_comparison.csv',
              index=False, encoding='utf-8-sig')
print("\n表1 各模型性能（按 F2 降序）:")
print(table1.to_string(index=False))

# 最优模型（论文预设规则：F2 最高者）
best_name = table1.iloc[0]['Model']
best_prob = prob_dict[best_name]
best_thr = table1.iloc[0]['Threshold']
y_pred = (best_prob > best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
auc_pt, auc_lo, auc_hi = bootstrap_auc_ci(y_test, best_prob)
print(f"\n最优模型: {best_name}  AUC={auc_pt:.3f} (95%CI {auc_lo:.3f}-{auc_hi:.3f})")

# ======================
# 7. 风险四级分级（论文 3.4）
# ======================
def risk_level(p):
    return '低风险' if p < 0.25 else '中风险' if p < 0.5 else '高风险' if p < 0.75 else '极高风险'

risk_df = pd.DataFrame({'实际': y_test.values, '概率': best_prob,
                        '风险等级': [risk_level(p) for p in best_prob],
                        '预测': y_pred})
risk_df.to_csv('results_pima/risk_stratification.csv', index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(8, 5))
order = ['低风险', '中风险', '高风险', '极高风险']
order_en = ['Low risk\n(<0.25)', 'Moderate\n(0.25-0.50)',
            'High\n(0.50-0.75)', 'Very high\n(>=0.75)']
counts = [int((risk_df['风险等级'] == lv).sum()) for lv in order]
prev = [risk_df[risk_df['风险等级'] == lv]['实际'].mean() * 100 for lv in order]
bars = ax.bar(order_en, counts, color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
for b, c, p in zip(bars, counts, prev):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
            f'n={c}\nprevalence {p:.1f}%', ha='center', fontsize=10)
ax.set_ylabel('Number of subjects')
ax.set_title('Risk Stratification - Pima Cohort')
plt.tight_layout()
plt.savefig('figures_pima/risk_stratification.png', dpi=300)
plt.close()

# ======================
# 8. 图：ROC
# ======================
fpr, tpr, _ = roc_curve(y_test, best_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'{best_name} (AUC={auc_pt:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Pima Cohort')
plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures_pima/roc_curve.png', dpi=300)
plt.close()
print("已保存: figures_pima/roc_curve.png")

# ======================
# 9. 图：DCA 决策曲线（论文 3.5）
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
plt.plot(ths, nb_model, 'b-', linewidth=2, label='Model')
plt.plot(ths, nb_all, 'g--', linewidth=1.5, label='Treat All')
plt.plot(ths, [0] * len(ths), 'r--', linewidth=1.5, label='Treat None')
plt.xlabel('Risk Threshold'); plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis - Pima Cohort')
plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures_pima/decision_curve.png', dpi=300)
plt.close()
print("已保存: figures_pima/decision_curve.png")

# ======================
# 10. 校准曲线 + Brier（论文 3.5）
# ======================
res = plot_calibration_curve(y_test.values, best_prob, 'Pima',
                             save_path='figures_pima/calibration_curve.png')
summary = plot_multi_model_calibration(
    prob_dict, y_test.values, 'Pima',
    save_path='figures_pima/calibration_all_models.png')
summary.to_csv('results_pima/brier_summary.csv', index=False, encoding='utf-8-sig')

# ======================
# 11. SHAP 可解释性（论文 3.5 / 图S4）
# ======================
try:
    import shap
    if best_name in ('RF', 'XGB', 'LGB'):
        sv = shap.TreeExplainer(models[best_name]).shap_values(X_test_sel)
    else:
        bg = shap.sample(X_train_s, 50, random_state=SEED)
        sv = shap.KernelExplainer(models[best_name].predict_proba, bg).shap_values(X_test_s[:100])
    if isinstance(sv, list):
        sv = sv[1]
    shap.summary_plot(sv, X_test_sel, feature_names=selected,
                      plot_type='bar', show=False)
    plt.tight_layout(); plt.savefig('figures_pima/shap_importance.png', dpi=300); plt.close()
    shap.summary_plot(sv, X_test_sel, feature_names=selected, show=False)
    plt.tight_layout(); plt.savefig('figures_pima/shap_summary.png', dpi=300); plt.close()
    print("已保存: figures_pima/shap_importance.png, shap_summary.png")
except Exception as e:
    print(f"[跳过 SHAP] {e}")

# ======================
# 12. 阈值敏感性 + 推理耗时（论文 3.3 / 3.7）
# ======================
threshold_sensitivity_analysis(y_test.values, best_prob, best_thr, 'Pima', 'results_pima')
benchmark_inference(models[best_name], X_test_sel, 'Pima', 'results_pima')

print("\n" + "=" * 70)
print(f"完成！最优模型 {best_name}：AUC={auc_pt:.3f}，召回率={recall_score(y_test, y_pred):.3f}，"
      f"漏诊 {fn} 人。结果在 results_pima/ 与 figures_pima/")
print("=" * 70)
