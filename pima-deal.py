# ======================
# pima
# ======================
import pandas as pd
import numpy as np
import warnings
import pickle
import os
import joblib
from datetime import datetime

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report, roc_curve, auc)

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('C:\\temp', exist_ok=True)
os.environ['TEMP'] = 'C:\\temp'
os.environ['TMP'] = 'C:\\temp'
warnings.filterwarnings('ignore')

print("=" * 80)
print("糖尿病风险预警模型")
print("=" * 80)

# ======================
# 1. 数据读取与特征工程
# ======================
print("\n【1/7】数据读取与特征工程...")
df = pd.read_csv("pima_diabetes.csv")
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    median_val = df[col].median()
    df[col] = df[col].replace(0, median_val)
df = df[(df['BloodPressure'] > 0) & (df['BMI'] > 10)]

df['Glucose_BMI'] = df['Glucose'] * df['BMI'] / 100
df['Age_Pregnancies'] = df['Age'] * df['Pregnancies'] / 10
df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1)
df['BMI_Age'] = df['BMI'] * df['Age'] / 100
df['SkinThickness_BMI'] = df['SkinThickness'] / (df['BMI'] + 1)
df['Metabolic_Risk'] = ((df['Glucose'] > 99).astype(int) * 2 + (df['BMI'] > 30).astype(int) * 2 +
                        (df['BloodPressure'] > 80).astype(int) + (df['Age'] > 40).astype(int))
df['Insulin_Log'] = np.log1p(df['Insulin'])
df['DPF_Log'] = np.log1p(df['DiabetesPedigreeFunction'])

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
feature_names = X.columns.tolist()

# ======================
# 2. 数据划分
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================
# 3. LASSO 特征筛选
# ======================
print("\n【3/7】LASSO特征筛选...")
lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso.fit(X_train_scaled, y_train)
coef_df = pd.DataFrame({'feature': feature_names, 'coef': lasso.coef_})
selected_features = coef_df[coef_df['coef'].abs() > 0.001]['feature'].tolist()
if len(selected_features) == 0:
    selected_features = feature_names

print(f"  筛选出 {len(selected_features)} 个核心特征")
for i, row in coef_df.nlargest(8, 'coef', keep='all').iterrows():
    print(f"    {row['feature']:25s}: {row['coef']:.4f}")

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]
scaler_sel = StandardScaler()
X_train_scaled_sel = scaler_sel.fit_transform(X_train_sel)
X_test_scaled_sel = scaler_sel.transform(X_test_sel)

# ======================
# 4. 模型训练
# ======================
print("\n【4/7】模型训练...")
models = {}

# LR
lr = LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced')
lr_params = {'C': [0.01, 0.05, 0.1, 0.5, 1, 2]}
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='roc_auc', n_jobs=1)
lr_grid.fit(X_train_scaled_sel, y_train)  # 添加训练
models['LR'] = lr_grid.best_estimator_
print(f"  LR 完成: C={lr_grid.best_estimator_.C:.2f}")

# RF
rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1)
rf_params = {'n_estimators': [100, 200], 'max_depth': [7, 10], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_jobs=1)
rf_grid.fit(X_train_scaled_sel, y_train)  # 添加训练
models['RF'] = rf_grid.best_estimator_
print(f"  RF 完成: depth={rf_grid.best_estimator_.max_depth}")

# XGB
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42,
                              scale_pos_weight=scale_pos_weight, use_label_encoder=False)
xgb_grid = GridSearchCV(xgb_model, {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
                        cv=5, scoring='roc_auc', n_jobs=1)
xgb_grid.fit(X_train_sel, y_train)
models['XGB'] = xgb_grid.best_estimator_
print(f"  XGB 完成: depth={xgb_grid.best_estimator_.max_depth}")

# LGB
lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
lgb_grid = GridSearchCV(lgb_model, {'n_estimators': [100, 200], 'max_depth': [5, 7]}, cv=5, scoring='roc_auc', n_jobs=1)
lgb_grid.fit(X_train_sel, y_train)
models['LGB'] = lgb_grid.best_estimator_
print(f"  LGB 完成")

# SVM
svm = SVC(probability=True, random_state=42, class_weight='balanced')
svm_params = {'C': [0.5, 1, 2], 'gamma': ['scale', 'auto']}
svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='roc_auc', n_jobs=1)
svm_grid.fit(X_train_scaled_sel, y_train)  # SVM使用标准化数据
models['SVM'] = svm_grid.best_estimator_
print(f"  SVM 完成: C={svm_grid.best_estimator_.C:.2f}")

# ======================
# 5. 集成模型
# ======================
print("\n【5/7】构建集成模型...")

# Voting
voting_soft = VotingClassifier(estimators=[(k, v) for k, v in models.items()], voting='soft', n_jobs=1)
voting_soft.fit(X_train_scaled_sel, y_train)
models['Voting'] = voting_soft
print("  Voting 完成")

# Stacking
stacking = StackingClassifier(
    estimators=[('LR', models['LR']), ('RF', models['RF']), ('XGB', models['XGB'])],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5,
    stack_method='predict_proba'
)
stacking.fit(X_train_scaled_sel, y_train)
models['Stacking'] = stacking
print("  Stacking 完成")

# ======================
# 6. 模型评估
# ======================
print("\n【6/7】模型评估...")


def evaluate_model_with_threshold(model, X_test_data, y_test, use_scaled, X_test_orig=None):
    if use_scaled:
        y_prob = model.predict_proba(X_test_data)[:, 1]
    else:
        y_prob = model.predict_proba(X_test_data)[:, 1]

    best_f2 = 0
    best_thr = 0.3
    for thr in np.arange(0.10, 0.71, 0.05):
        yp = (y_prob > thr).astype(int)
        r = recall_score(y_test, yp)
        p = precision_score(y_test, yp)
        if p + r > 0:
            f2 = 5 * p * r / (4 * p + r)
            if f2 > best_f2:
                best_f2 = f2
                best_thr = thr

    yp = (y_prob > best_thr).astype(int)
    return {
        'AUC': roc_auc_score(y_test, y_prob),
        'Recall': recall_score(y_test, yp),
        'Precision': precision_score(y_test, yp),
        'F2': best_f2,
        'Threshold': best_thr
    }, y_prob


results = []
prob_dict = {}

for name, m in models.items():
    use_scaled = name in ['LR', 'SVM', 'Voting', 'Stacking']
    res, prob = evaluate_model_with_threshold(
        m,
        X_test_scaled_sel if use_scaled else X_test_sel,
        y_test,
        use_scaled
    )
    res['Model'] = name
    results.append(res)
    prob_dict[name] = prob

res_df = pd.DataFrame(results).sort_values('F2', ascending=False)
best_model_name = res_df.iloc[0]['Model']
best_model = models[best_model_name]
best_prob = prob_dict[best_model_name]
best_thr = res_df.iloc[0]['Threshold']
y_pred_best = (best_prob > best_thr).astype(int)

print(f"\n🏆 最佳模型: {best_model_name}")
print(f"   最优阈值: {best_thr:.2f}")
print(f"   AUC: {roc_auc_score(y_test, best_prob):.4f}")
print(f"   召回率: {recall_score(y_test, y_pred_best):.4f}")

# ======================
# 7. 交叉验证与置信区间
# ======================
print("\n  计算交叉验证...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
try:
    # 尝试用标准化数据
    cv_scores = cross_val_score(best_model, X_train_scaled_sel, y_train,
                                cv=cv, scoring='roc_auc', n_jobs=1)
except:
    # 树模型用原始数据
    cv_scores = cross_val_score(best_model, X_train_sel, y_train,
                                cv=cv, scoring='roc_auc', n_jobs=1)

print(f"  5折交叉验证 AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ======================
# 8. 风险分级
# ======================
def risk_level(p):
    if p < 0.25:
        return "低风险"
    elif p < 0.5:
        return "中风险"
    elif p < 0.75:
        return "高风险"
    else:
        return "极高风险"


risk_levels = [risk_level(p) for p in best_prob]
cm = confusion_matrix(y_test, y_pred_best)

# ======================
# 9. SHAP可解释性
# ======================
print("\n  生成 SHAP 可解释性分析...")


def run_shap():
    try:
        import shap

        # 根据模型类型选择不同的Explainer
        model_name_lower = best_model_name.lower()

        if 'rf' in model_name_lower or 'xgb' in model_name_lower or 'lgb' in model_name_lower:
            # 树模型使用TreeExplainer
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test_sel)
        elif 'lr' in model_name_lower:
            # 线性模型使用LinearExplainer
            explainer = shap.LinearExplainer(best_model, X_train_scaled_sel)
            shap_values = explainer.shap_values(X_test_scaled_sel)
        else:
            # 集成模型使用KernelExplainer（较慢但通用）
            background = shap.sample(X_train_scaled_sel, 50)
            explainer = shap.KernelExplainer(best_model.predict_proba, background)
            shap_values = explainer.shap_values(X_test_scaled_sel[:50])
            shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

        # 处理多分类输出
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # 特征重要性条形图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_sel, feature_names=selected_features,
                          plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig("figures/shap_importance.png", dpi=300)
        plt.close()

        # SHAP摘要图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_sel, feature_names=selected_features, show=False)
        plt.tight_layout()
        plt.savefig("figures/shap_summary.png", dpi=300)
        plt.close()

        print("  ✅ SHAP 图表已保存: figures/shap_importance.png, figures/shap_summary.png")

    except ImportError:
        print("  ⚠️ shap库未安装，跳过SHAP分析。安装命令: pip install shap")
    except Exception as e:
        print(f"  ⚠️ SHAP分析跳过: {e}")


run_shap()


# ======================
# 10. 精细阈值优化
# ======================
def threshold_tuning(y_true, y_prob):
    rows = []
    for t in np.arange(0.15, 0.55, 0.02):
        yp = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
        miss = fn / (tp + fn) if (tp + fn) > 0 else 0
        pre = precision_score(y_true, yp) if np.sum(yp) > 0 else 0
        f2 = 5 * pre * sen / (4 * pre + sen + 1e-8)
        rows.append([round(t, 2), round(sen, 3), round(spe, 3), round(miss, 3), round(f2, 3)])

    df_thr = pd.DataFrame(rows, columns=['阈值', '敏感度', '特异度', '漏诊率', 'F2'])
    df_thr = df_thr.sort_values('F2', ascending=False)
    df_thr.to_csv("results/threshold_optimize.csv", index=False, encoding='utf-8-sig')

    print("\n===== 最优阈值推荐 =====")
    print(df_thr.head(6).to_string(index=False))
    return df_thr


threshold_tuning(y_test, best_prob)

# ======================
# 11. 模型保存
# ======================
joblib.dump(best_model, '../统计建模/models/diabetes_best_model.pkl')
joblib.dump(scaler_sel, '../统计建模/models/scaler.pkl')
joblib.dump(selected_features, '../统计建模/models/selected_features.pkl')

print("\n✅ 模型已保存至 models/ 文件夹")

# ======================
# 12. 生成ROC曲线
# ======================
print("\n  生成 ROC 曲线...")
fpr, tpr, _ = roc_curve(y_test, best_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'{best_model_name} (AUC={roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve-pima', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curve.png', dpi=300)
plt.close()
print("  ✅ ROC曲线已保存: figures-pima/roc_curve.png")

# ======================
# 13. 生成决策曲线
# ======================
print("\n  生成 决策曲线分析(DCA)...")
def net_benefit(y_true, y_prob, threshold):
    y_pred = (y_prob > threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    n = len(y_true)
    if threshold == 1:
        nb = 0
    else:
        nb = tp / n - fp / n * (threshold / (1 - threshold))
    return max(0, nb)


thresholds_dca = np.arange(0.01, 0.5, 0.02)
nb_model = [net_benefit(y_test, best_prob, t) for t in thresholds_dca]
nb_all = [max(0, y_test.mean() - t / (1 - t) * (1 - y_test.mean())) for t in thresholds_dca]
nb_none = [0] * len(thresholds_dca)

plt.figure(figsize=(8, 6))
plt.plot(thresholds_dca, nb_model, 'b-', linewidth=2, label='本研究模型')
plt.plot(thresholds_dca, nb_all, 'g--', linewidth=1.5, label='全部干预')
plt.plot(thresholds_dca, nb_none, 'r--', linewidth=1.5, label='不干预')
plt.xlabel('Risk Threshold', fontsize=12)
plt.ylabel('Net Benefit', fontsize=12)
plt.title('Decision Curve Analysis (DCA)', fontsize=14)
plt.legend(['Our Model', 'Treat All', 'Treat None'], loc='upper right')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/decision_curve.png', dpi=300)
plt.close()
print("  ✅ DCA曲线已保存: figures/decision_curve.png")


# ======================
# 14. 生成临床报告
# ======================
def generate_report():
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    miss = fn / (tp + fn) if (tp + fn) > 0 else 0

    risk_counts = pd.Series(risk_levels).value_counts()

    txt = f"""
{'=' * 60}
糖尿病风险预警模型 - 临床筛查报告
{'=' * 60}
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
最佳模型：{best_model_name}
最优阈值：{best_thr:.2f}
测试样本：{len(y_test)} 例

{'=' * 60}
【模型性能指标】
{'=' * 60}
AUC：{roc_auc_score(y_test, best_prob):.4f}
准确率：{accuracy_score(y_test, y_pred_best):.3f}
敏感度（召回率）：{sen:.3f}
特异度：{spe:.3f}
精确率：{precision_score(y_test, y_pred_best):.3f}
漏诊率：{miss:.3f}
F2分数：{res_df.iloc[0]['F2']:.3f}

{'=' * 60}
【混淆矩阵】
{'=' * 60}
                预测未患病    预测患病
实际未患病        {tn}          {fp}
实际患病          {fn}          {tp}

{'=' * 60}
【风险分级结果】
{'=' * 60}
低风险（<0.25）：    {risk_counts.get('低风险', 0)} 例
中风险（0.25-0.5）：  {risk_counts.get('中风险', 0)} 例
高风险（0.5-0.75）：  {risk_counts.get('高风险', 0)} 例
极高风险（≥0.75）：   {risk_counts.get('极高风险', 0)} 例

{'=' * 60}
• 本预测结果仅供参考，不能替代医生诊断
• 建议结合临床症状和实验室检查综合判断
• 模型最优阈值为 {best_thr:.2f}，可根据实际情况调整

{'=' * 60}
报告结束
{'=' * 60}
"""
    with open("../统计建模/results/临床报告.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print("✅ 临床报告已保存: results/临床报告.txt")


generate_report()
# ======================
# 15.5 保存所有模型到统一文件夹
# ======================
print("\n【保存所有模型】")

# 创建模型保存目录
all_models_dir = "../统计建模/models/all_models/"
os.makedirs(all_models_dir, exist_ok=True)

# 保存所有训练好的模型
for name, model in models.items():
    model_path = os.path.join(all_models_dir, f"{name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"  ✅ 已保存: {name}_model.pkl")

# 保存标准化器和特征列表（供所有模型使用）
joblib.dump(scaler_sel, os.path.join(all_models_dir, "scaler.pkl"))
joblib.dump(selected_features, os.path.join(all_models_dir, "selected_features.pkl"))

# 额外保存最佳模型（单独一份）
joblib.dump(best_model, os.path.join(all_models_dir, "best_model.pkl"))

print(f"✅ 所有模型已保存至: {all_models_dir}")
# ======================
# 15. 保存结果表格
# ======================
res_df.to_csv("results/model_performance.csv", encoding="utf-8-sig", index=False)

# 风险分级详细表
risk_detail_df = pd.DataFrame({
    '实际结果': y_test.values,
    '预测概率': best_prob,
    '风险等级': risk_levels,
    '预测结果': y_pred_best
})
risk_detail_df.to_csv("results/risk_classification_detail.csv", encoding="utf-8-sig", index=False)

print("\n✅ 所有结果已保存至 results/ 文件夹")

# ======================
# 16. 最终总结
# ======================
print("\n" + "=" * 80)
print("最终结果汇总")
print("=" * 80)
print(f"""
【模型性能】
  最佳模型: {best_model_name}
  AUC: {roc_auc_score(y_test, best_prob):.4f}
  5折交叉验证 AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
  敏感度: {recall_score(y_test, y_pred_best):.3f}
  特异度: {cm[0, 0] / (cm[0, 0] + cm[0, 1]):.3f}
  漏诊率: {cm[1, 0] / (cm[1, 0] + cm[1, 1]):.1%} (仅{cm[1, 0]}人)

【风险分级】
  低风险: {(np.array(risk_levels) == '低风险').sum()}人
  中风险: {(np.array(risk_levels) == '中风险').sum()}人
  高风险: {(np.array(risk_levels) == '高风险').sum()}人
  极高风险: {(np.array(risk_levels) == '极高风险').sum()}人

【输出文件】
  📁 figures/  → ROC曲线、DCA曲线、SHAP图
  📁 models/   → 训练好的模型文件
  📁 results/  → 性能表格、风险分级、临床报告
""")

print("\n" + "=" * 80)
print("🎉 全部运行完成！")
print("=" * 80)

# ======================
# 使用各模型自己的最优阈值进行对比
# ======================

print("\n" + "=" * 80)
print("【模型详细性能对比（使用各自最优阈值）】")
print("=" * 80)

# 创建一个详细的性能对比表格
comparison_data = []

for name, model in models.items():
    # 判断是否需要标准化
    use_scaled = name in ['LR', 'SVM', 'Voting', 'Stacking']

    if use_scaled:
        y_prob = model.predict_proba(X_test_scaled_sel)[:, 1]
    else:
        y_prob = model.predict_proba(X_test_sel)[:, 1]

    # 为每个模型单独寻找最优阈值（以F2分数为目标）
    best_f2 = 0
    best_thr = 0.5
    for thr in np.arange(0.10, 0.71, 0.05):
        yp = (y_prob > thr).astype(int)
        r = recall_score(y_test, yp)
        p = precision_score(y_test, yp)
        if p + r > 0:
            f2 = 5 * p * r / (4 * p + r)
            if f2 > best_f2:
                best_f2 = f2
                best_thr = thr

    # 使用最优阈值进行预测
    y_pred_opt = (y_prob > best_thr).astype(int)

    # 计算各项指标
    auc = roc_auc_score(y_test, y_prob)
    recall_opt = recall_score(y_test, y_pred_opt)
    precision_opt = precision_score(y_test, y_pred_opt)

    # 计算F2分数
    if precision_opt + recall_opt > 0:
        f2_opt = 5 * precision_opt * recall_opt / (4 * precision_opt + recall_opt)
    else:
        f2_opt = 0

    # 计算漏诊数
    cm = confusion_matrix(y_test, y_pred_opt)
    fn = cm[1, 0]  # 漏诊数
    fp = cm[0, 1]  # 误诊数

    comparison_data.append({
        '模型名称': name,
        '最优阈值': best_thr,
        'AUC': auc,
        '召回率': recall_opt,
        '精确率': precision_opt,
        'F2分数': f2_opt,
        '漏诊数': fn,
        '误诊数': fp
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('F2分数', ascending=False)

print("\n【Pima数据集各模型详细性能对比表（各模型使用各自最优阈值）】")
print("-" * 110)
print(f"{'模型名称':<15} {'最优阈值':<8} {'AUC':<8} {'召回率':<8} {'精确率':<8} {'F2分数':<8} {'漏诊数':<6}")
print("-" * 110)

for _, row in comparison_df.iterrows():
    print(
        f"{row['模型名称']:<15} {row['最优阈值']:.2f}      {row['AUC']:.4f}   {row['召回率']:.3f}     {row['精确率']:.3f}     {row['F2分数']:.3f}     {row['漏诊数']:d}")

print("-" * 110)

# 特别对比 XGBoost 和 Stacking
print("\n【XGBoost vs Stacking 详细对比（使用各自最优阈值）】")
print("=" * 70)

xgb_row = comparison_df[comparison_df['模型名称'] == 'XGB'].iloc[0]
stack_row = comparison_df[comparison_df['模型名称'] == 'Stacking'].iloc[0]

print(f"\n{'指标':<15} {'XGBoost':<18} {'Stacking':<18} {'谁更优':<10}")
print("-" * 65)
print(f"{'最优阈值':<15} {xgb_row['最优阈值']:.2f}              {stack_row['最优阈值']:.2f}              -")
print(
    f"{'AUC':<15} {xgb_row['AUC']:.4f}           {stack_row['AUC']:.4f}           {'Stacking' if stack_row['AUC'] > xgb_row['AUC'] else 'XGBoost'}")
print(
    f"{'召回率':<15} {xgb_row['召回率']:.3f}            {stack_row['召回率']:.3f}            {'XGBoost' if xgb_row['召回率'] > stack_row['召回率'] else 'Stacking'}")
print(
    f"{'精确率':<15} {xgb_row['精确率']:.3f}            {stack_row['精确率']:.3f}            {'Stacking' if stack_row['精确率'] > xgb_row['精确率'] else 'XGBoost'}")
print(
    f"{'F2分数':<15} {xgb_row['F2分数']:.4f}           {stack_row['F2分数']:.4f}           {'XGBoost' if xgb_row['F2分数'] > stack_row['F2分数'] else 'Stacking'}")
print(
    f"{'漏诊数':<15} {int(xgb_row['漏诊数']):d}               {int(stack_row['漏诊数']):d}               {'XGBoost' if xgb_row['漏诊数'] < stack_row['漏诊数'] else 'Stacking'}")

print("=" * 70)

# 最终结论
print("\n【最终结论】")
print("=" * 60)
best_model_by_f2 = comparison_df.iloc[0]['模型名称']
best_f2_value = comparison_df.iloc[0]['F2分数']
print(f"🏆 按F2分数排序，最优模型为: {best_model_by_f2} (F2={best_f2_value:.4f})")

# 根据第6节已经确定的最优模型
print(f"🏆 第6节评估确定的最优模型为: {best_model_name} (最优阈值={best_thr:.2f})")
print(f"   该模型召回率={recall_score(y_test, y_pred_best):.3f}, F2={res_df.iloc[0]['F2']:.3f}")

if best_model_name == best_model_by_f2:
    print("✅ 两者结论一致！")
else:
    print("⚠️ 两者结论不一致，建议以第6节优化阈值后的结果为准")

print("=" * 60)


# ======================
# 阈值敏感性分析
# ======================

def threshold_sensitivity_analysis(
        y_true,  # 真实标签
        y_prob,  # 预测概率
        current_threshold,  # 当前使用的阈值
        dataset_name,  # 数据集名称（用于文件名）
        output_dir,  # 输出目录
        thresholds=None,  # 自定义阈值列表，默认0.10-0.70步长0.05
        figsize=(14, 5),  # 图片尺寸
        dpi=300  # 图片分辨率
):
    """
    阈值敏感性分析：在多个阈值下计算性能指标并可视化
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    if thresholds is None:
        thresholds = np.arange(0.10, 0.71, 0.05)

    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        f2 = 5 * precision * recall / (4 * precision + recall) if (precision + recall) > 0 else 0

        n = len(y_true)
        nb = tp / n - fp / n * (t / (1 - t)) if t < 1 else 0
        nb = max(0, nb)

        rows.append({
            'Threshold': t,
            'Recall': recall,
            'Specificity': specificity,
            'Miss_Rate': miss_rate,
            'Precision': precision,
            'F2_Score': f2,
            'Net_Benefit': nb,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        })

    df = pd.DataFrame(rows)

    best_f2_idx = df['F2_Score'].idxmax()
    best_nb_idx = df['Net_Benefit'].idxmax()

    print(f"\n【{dataset_name} 阈值敏感性分析】")
    print("-" * 70)
    print(f"  F2分数最优阈值: {df.loc[best_f2_idx, 'Threshold']:.2f} (F2={df.loc[best_f2_idx, 'F2_Score']:.4f})")
    print(f"  净获益最优阈值: {df.loc[best_nb_idx, 'Threshold']:.2f} (NB={df.loc[best_nb_idx, 'Net_Benefit']:.4f})")
    print(f"  当前模型阈值: {current_threshold:.2f}")
    print("-" * 70)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"threshold_sensitivity_{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV已保存: {csv_path}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    ax1.plot(df['Threshold'], df['Recall'], 'b-o', linewidth=2, markersize=6, label='Recall')
    ax1.plot(df['Threshold'], df['Specificity'], 'g-s', linewidth=2, markersize=6, label='Specificity')
    ax1.plot(df['Threshold'], df['Miss_Rate'], 'r-^', linewidth=2, markersize=6, label='Miss Rate')
    ax1.axvline(x=current_threshold, color='purple', linestyle='--', linewidth=2,
                label=f'Current={current_threshold:.2f}')
    ax1.axvline(x=df.loc[best_f2_idx, 'Threshold'], color='orange', linestyle=':', linewidth=2,
                label=f'F2-best={df.loc[best_f2_idx, "Threshold"]:.2f}')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Value')
    ax1.set_title(f'{dataset_name}: Performance vs Threshold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2 = axes[1]
    ax2.plot(df['Threshold'], df['F2_Score'], color='purple', marker='o',
             linestyle='-', linewidth=2, markersize=6, label='F2 Score')
    ax2.plot(df['Threshold'], df['Net_Benefit'], color='orange', marker='s',
             linestyle='-', linewidth=2, markersize=6, label='Net Benefit')
    ax2.axvline(x=current_threshold, color='purple', linestyle='--', linewidth=2,
                label=f'Current={current_threshold:.2f}')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title(f'{dataset_name}: F2 Score & Net Benefit')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"threshold_sensitivity_{dataset_name}.png")
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  图片已保存: {fig_path}")

    return df

threshold_df_pima = threshold_sensitivity_analysis(
    y_true=y_test,
    y_prob=best_prob,
    current_threshold=best_thr,
    dataset_name='Pima',
    output_dir='results'
)

# 确保 analysis_utils.py 在同一目录下
from analysis_utils import (
    threshold_sensitivity_analysis,
    cost_sensitivity_analysis,
    temporal_validation,
    benchmark_inference,
    generate_performance_summary
)

CONFIG = {
    'output_dir': 'results',              # 所有结果输出目录
    'threshold_start': 0.10,              # 阈值分析起始值
    'threshold_end': 0.70,                # 阈值分析结束值
    'threshold_step': 0.05,               # 阈值分析步长
    'benchmark_iterations': 1000,         # 推理测试重复次数
    'benchmark_batch_size': 100,          # 批量测试样本数
}

# ============================================================
# 1. 阈值敏感性分析
# ============================================================
threshold_df = threshold_sensitivity_analysis(
    y_true=y_test,
    y_prob=best_prob,
    current_threshold=best_thr,
    dataset_name='Pima',
    output_dir=CONFIG['output_dir'],
    thresholds=np.arange(
        CONFIG['threshold_start'],
        CONFIG['threshold_end'] + CONFIG['threshold_step'],
        CONFIG['threshold_step']
    )
)

# ============================================================
# 2. 推理性能测试
# ============================================================
benchmark_result = benchmark_inference(
    model=best_model,
    X_test=X_test_sel,
    dataset_name='Pima',
    output_dir=CONFIG['output_dir'],
    n_iterations=CONFIG['benchmark_iterations'],
    batch_size=CONFIG['benchmark_batch_size']
)

# ============================================================
# 3. 性能汇总报告
# ============================================================
summary = generate_performance_summary(
    y_true=y_test,
    y_prob=best_prob,
    y_pred=y_pred_best,
    model_name=best_model_name,
    dataset_name='Pima',
    threshold=best_thr,
    cv_scores=cv_scores,
    cm=cm,
    output_dir=CONFIG['output_dir']
)

print("\n✅ 全部补充分析完成！")
