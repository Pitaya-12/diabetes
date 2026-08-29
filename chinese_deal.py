# ======================
# 中国糖尿病数据集 - 完整建模（与Pima流程一致）
# 用于论文对比分析
# ======================
import pandas as pd
import numpy as np
import warnings
import pickle
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report, roc_curve, auc)
import xgboost as xgb
import lightgbm as lgb

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('C:\\temp', exist_ok=True)
os.environ['TEMP'] = 'C:\\temp'
os.environ['TMP'] = 'C:\\temp'
warnings.filterwarnings('ignore')

print("=" * 80)
print("中国糖尿病数据集 - 糖尿病风险预警模型（与Pima流程一致）")
print("=" * 80)

os.makedirs('figures_china', exist_ok=True)
os.makedirs('models_china', exist_ok=True)
os.makedirs('results_china', exist_ok=True)

# ======================
# 1. 数据读取与清洗
# ======================
print("\n【1/7】数据读取与清洗...")

df = pd.read_csv("diabetes.csv")

# 查看数据基本信息
print(f"原始数据形状: {df.shape}")
print(f"特征列表: {df.columns.tolist()}")

# 检查异常值（4.860753 可能是缺失值填充标记）
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        # 检查是否有异常填充值
        if (df[col] == 4.860753).sum() > 0:
            print(f"  {col}: 发现 {(df[col] == 4.860753).sum()} 个异常值(4.860753)，将替换为NaN")
            df[col] = df[col].replace(4.860753, np.nan)

# 缺失值处理：用中位数填充
print("\n缺失值统计:")
print(df.isnull().sum())
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# 剔除异常值
df = df[df['BMI'] > 10]  # 剔除BMI异常值
df = df[df['SBP'] > 0]   # 剔除收缩压异常值
df = df[df['DBP'] > 0]   # 剔除舒张压异常值

# 查看目标变量分布
print(f"\n清洗后数据形状: {df.shape}")
print(f"患病样本: {df['Diabetes'].sum()}例 ({df['Diabetes'].mean()*100:.1f}%)")
print(f"未患病样本: {(df['Diabetes']==0).sum()}例")
print(f"性别分布: 男={df[df['Gender']==1].shape[0]}人, 女={df[df['Gender']==2].shape[0]}人")

# ======================
# 2. 特征工程（与Pima流程一致）
# ======================
print("\n【2/7】特征工程...")

# 原始特征
X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]
feature_names = X.columns.tolist()
print(f"原始特征数: {len(feature_names)}")

# 创建新特征（与Pima类似）
# 交互特征
X['Glucose_BMI'] = X['FPG'] * X['BMI'] / 100  # 血糖×BMI
X['Age_Gender'] = X['Age'] * X['Gender'] / 10  # 年龄×性别
X['SBP_DBP_Ratio'] = X['SBP'] / (X['DBP'] + 1)  # 血压比值
X['BMI_Age'] = X['BMI'] * X['Age'] / 100  # BMI×年龄

# 比例特征
X['HDL_LDL_Ratio'] = X['HDL'] / (X['LDL'] + 1)  # 好/坏胆固醇比值

# 代谢风险评分
X['Metabolic_Risk'] = (
    (X['FPG'] > 5.6).astype(int) * 2 +
    (X['BMI'] > 28).astype(int) * 2 +
    (X['SBP'] > 130).astype(int) +
    (X['Age'] > 50).astype(int)
)

# 对数变换
X['Chol_Log'] = np.log1p(X['Chol'])
X['Tri_Log'] = np.log1p(X['Tri'])

print(f"特征工程后特征数: {X.shape[1]}")

# ======================
# 3. 数据划分
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集: {X_train.shape[0]}样本 | 测试集: {X_test.shape[0]}样本")

# ======================
# 4. LASSO特征筛选
# ======================
print("\n【3/7】LASSO特征筛选...")

lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso.fit(X_train_scaled, y_train)

coef_df = pd.DataFrame({'feature': X.columns.tolist(), 'coef': lasso.coef_})
selected_features = coef_df[coef_df['coef'].abs() > 0.001]['feature'].tolist()

if len(selected_features) == 0:
    selected_features = X.columns.tolist()

print(f"筛选出 {len(selected_features)} 个核心特征")

# 特征重要性排序
coef_df['abs_coef'] = coef_df['coef'].abs()
coef_df_sorted = coef_df.sort_values('abs_coef', ascending=False)
print("\n特征重要性排序（LASSO系数）:")
for i, row in coef_df_sorted.head(10).iterrows():
    print(f"  {row['feature']:25s}: {row['coef']:.4f}")

# 使用筛选后的特征
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]
scaler_sel = StandardScaler()
X_train_scaled_sel = scaler_sel.fit_transform(X_train_sel)
X_test_scaled_sel = scaler_sel.transform(X_test_sel)

# ======================
# 5. 模型训练（超参数网格已与Pima统一）
# ======================
print("\n【4/7】模型训练...")

models = {}

# 逻辑回归
print("  训练 Logistic Regression...")
lr = LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced')
lr_params = {'C': [0.01, 0.05, 0.1, 0.5, 1, 2]}  # 已与Pima统一
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='roc_auc', n_jobs=1, verbose=0)
lr_grid.fit(X_train_scaled_sel, y_train)
models['LR'] = lr_grid.best_estimator_
print(f"    最优参数: C={lr_grid.best_estimator_.C}")

# 随机森林
print("  训练 Random Forest...")
rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1)
rf_params = {'n_estimators': [100, 200], 'max_depth': [7, 10], 'min_samples_split': [2, 5]}  # 已与Pima统一
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_jobs=1, verbose=0)
rf_grid.fit(X_train_sel, y_train)
models['RF'] = rf_grid.best_estimator_
print(f"    最优参数: n_estimators={rf_grid.best_estimator_.n_estimators}")

# XGBoost
print("  训练 XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42,
                               use_label_encoder=False, scale_pos_weight=scale_pos_weight)
xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}  # 已与Pima一致
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='roc_auc', n_jobs=1, verbose=0)
xgb_grid.fit(X_train_sel, y_train)
models['XGB'] = xgb_grid.best_estimator_
print(f"    最优参数: max_depth={xgb_grid.best_estimator_.max_depth}")

# LightGBM
print("  训练 LightGBM...")
lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
lgb_params = {'n_estimators': [100, 200], 'max_depth': [5, 7], 'learning_rate': [0.05, 0.1]}  # 已与Pima统一
lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=5, scoring='roc_auc', n_jobs=1, verbose=0)
lgb_grid.fit(X_train_sel, y_train)
models['LGB'] = lgb_grid.best_estimator_
print(f"    最优参数: n_estimators={lgb_grid.best_estimator_.n_estimators}")

# SVM
print("  训练 SVM...")
svm = SVC(probability=True, random_state=42, class_weight='balanced')
svm_params = {'C': [0.5, 1, 2], 'gamma': ['scale', 'auto']}  # 已与Pima统一
svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='roc_auc', n_jobs=1, verbose=0)
svm_grid.fit(X_train_scaled_sel, y_train)
models['SVM'] = svm_grid.best_estimator_
print(f"    最优参数: C={svm_grid.best_estimator_.C}")

# ======================
# 6. 集成模型
# ======================
print("\n【5/7】构建集成模型...")

# 软投票
voting_soft = VotingClassifier(estimators=[(k, v) for k, v in models.items()], voting='soft', n_jobs=1)
voting_soft.fit(X_train_scaled_sel, y_train)
models['Voting'] = voting_soft

# Stacking
stacking = StackingClassifier(
    estimators=[('LR', models['LR']), ('RF', models['RF']), ('XGB', models['XGB'])],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5,
    stack_method='predict_proba'
)
stacking.fit(X_train_scaled_sel, y_train)
models['Stacking'] = stacking

# ======================
# 7. 模型评估
# ======================
print("\n【6/7】模型评估...")

def evaluate_model_with_threshold(model, X_test_data, y_test, use_scaled, X_test_orig=None):
    if use_scaled:
        y_prob = model.predict_proba(X_test_data)[:, 1]
    else:
        y_prob = model.predict_proba(X_test_data)[:, 1]

    best_f2 = 0
    best_thr = 0.3
    for thr in np.arange(0.15, 0.6, 0.05):
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
# 8. 交叉验证
# ======================
print("\n  计算交叉验证...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
try:
    cv_scores = cross_val_score(best_model, X_train_scaled_sel, y_train,
                                 cv=cv, scoring='roc_auc', n_jobs=1)
except:
    cv_scores = cross_val_score(best_model, X_train_sel, y_train,
                                 cv=cv, scoring='roc_auc', n_jobs=1)

print(f"  5折交叉验证 AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ======================
# 9. 混淆矩阵
# ======================
cm = confusion_matrix(y_test, y_pred_best)
print(f"\n混淆矩阵（阈值={best_thr:.2f}）:")
print(f"  正确预测未患病: {cm[0,0]} | 误诊(假阳性): {cm[0,1]}")
print(f"  漏诊(假阴性): {cm[1,0]} | 正确预测患病: {cm[1,1]}")

print(f"\n详细分类报告:")
print(classification_report(y_test, y_pred_best, target_names=['未患病', '患病']))

# ======================
# 10. 风险分级
# ======================
print("\n" + "=" * 80)
print("风险分级结果")
print("=" * 80)

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
risk_df = pd.DataFrame({
    'Actual_Outcome': y_test.values,
    'Risk_Probability': best_prob,
    'Risk_Level': risk_levels,
    'Predicted': y_pred_best
})

risk_dist = risk_df['Risk_Level'].value_counts()
print("\n风险人群分布:")
for level, count in risk_dist.items():
    pct = count / len(risk_df) * 100
    subset = risk_df[risk_df['Risk_Level'] == level]
    actual_rate = subset['Actual_Outcome'].mean() if len(subset) > 0 else 0
    print(f"  {level}: {count}例 ({pct:.1f}%) - 实际患病率: {actual_rate:.1%}")

high_risk = risk_df[risk_df['Risk_Level'].isin(['高风险', '极高风险'])]
if len(high_risk) > 0:
    print(f"\n⚠️ 高危人群预警:")
    print(f"   人数: {len(high_risk)}人 ({len(high_risk)/len(risk_df)*100:.1f}%)")
    print(f"   实际患病率: {high_risk['Actual_Outcome'].mean():.1%}")

# ======================
# 11. ROC曲线
# ======================
print("\n  生成 ROC 曲线...")
fpr, tpr, _ = roc_curve(y_test, best_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'{best_model_name} (AUC={roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random Classifier')
plt.xlabel('假阳性率 (1 - 特异度)', fontsize=12)
plt.ylabel('真阳性率 (敏感度)', fontsize=12)
plt.title('ROC曲线 - 中国人群数据', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures_china/roc_curve.png', dpi=300)
plt.close()
print("  ✅ ROC曲线已保存: figures_china/roc_curve.png")

# ======================
# 12. 保存结果
# ======================
print("\n保存模型和结果...")

# 保存模型
joblib.dump(best_model, 'models_china/diabetes_china_model.pkl')
joblib.dump(scaler_sel, 'models_china/scaler_china.pkl')
pickle.dump(selected_features, open('models_china/selected_features_china.pkl', 'wb'))
# ----===== 新增：保存所有模型 & 所有概率 =====
pickle.dump(models, open('models_china/all_models.pkl', 'wb'))            # 注意 models 键是 'LR', 'RF' 等
pickle.dump(prob_dict, open('models_china/all_probs.pkl', 'wb'))          # prob_dict 键与 models 一致
# 保存结果表格
res_df.to_csv("results_china/model_performance_china.csv", encoding="utf-8-sig", index=False)
risk_df.to_csv("results_china/risk_classification_china.csv", encoding="utf-8-sig", index=False)

# 论文对比表格
comparison_df = pd.DataFrame({
    '数据集': ['Pima Indians', '中国人群'],
    '样本量': [768, len(X)],
    '患病率': [f"{34.9}%", f"{y.sum()/len(y)*100:.1f}%"],
    '最佳模型': ['XGBoost', best_model_name],
    'AUC': [0.8263, round(roc_auc_score(y_test, best_prob), 4)],
    '召回率': [0.963, round(recall_score(y_test, y_pred_best), 4)],
    '最优阈值': [0.20, best_thr]
})
comparison_df.to_csv("results_china/model_comparison.csv", encoding="utf-8-sig", index=False)

print("\n✅ 已保存文件:")
print("  📁 figures_china/ → ROC曲线")
print("  📁 models_china/ → 训练好的模型")
print("  📁 results_china/ → 性能表格、风险分级、对比结果")

# ======================
# 13. 最终总结
# ======================
print("\n" + "=" * 80)
print("中国人群数据建模结果汇总")
print("=" * 80)
print(f"""
【数据概况】
  总样本量: {len(X)} 例
  患病率: {y.mean()*100:.1f}%
  性别分布: 男={(df['Gender']==1).sum()}人, 女={(df['Gender']==2).sum()}人

【模型性能】
  最佳模型: {best_model_name}
  AUC: {roc_auc_score(y_test, best_prob):.4f}
  5折交叉验证 AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
  召回率: {recall_score(y_test, y_pred_best):.3f}
  特异度: {cm[0,0]/(cm[0,0]+cm[0,1]):.3f}

【风险分级】
  低风险: {(np.array(risk_levels)=='低风险').sum()}人
  中风险: {(np.array(risk_levels)=='中风险').sum()}人
  高风险: {(np.array(risk_levels)=='高风险').sum()}人
  极高风险: {(np.array(risk_levels)=='极高风险').sum()}人

【与Pima数据对比】
  Pima Indians: AUC=0.8263, 召回率=0.963
  中国人群: AUC={roc_auc_score(y_test, best_prob):.4f}, 召回率={recall_score(y_test, y_pred_best):.3f}
""")

print("\n" + "=" * 80)
print("🎉 中国人群数据建模完成！")
print("=" * 80)


# ======================
# 阈值敏感性分析（修复版）
# ======================
def threshold_sensitivity_analysis(y_true, y_prob, model_name, dataset_name,
                                   thresholds=None, save_dir='results'):
    """
    在0.1~0.7区间每隔0.05取阈值，输出AUC、召回率、特异度、漏诊率、F2分数、净获益
    """
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

        # 净获益 (Net Benefit)
        n = len(y_true)
        nb = tp / n - fp / n * (t / (1 - t)) if t < 1 else 0
        nb = max(0, nb)  # 不低于0

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

    # 找出F2分数最大和净获益最大的阈值
    best_f2_idx = df['F2_Score'].idxmax()
    best_nb_idx = df['Net_Benefit'].idxmax()

    print(f"\n【{dataset_name} 阈值敏感性分析结果】")
    print("-" * 90)
    print(f"  F2分数最优阈值: {df.loc[best_f2_idx, 'Threshold']:.2f} (F2={df.loc[best_f2_idx, 'F2_Score']:.4f})")
    print(f"  净获益最优阈值: {df.loc[best_nb_idx, 'Threshold']:.2f} (NB={df.loc[best_nb_idx, 'Net_Benefit']:.4f})")
    print(f"  当前模型阈值: {best_thr:.2f}")

    # 保存CSV
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/threshold_sensitivity_{dataset_name}.csv", index=False)

    # 绘制敏感性曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Recall、Specificity、Miss_Rate随阈值变化
    ax1 = axes[0]
    ax1.plot(df['Threshold'], df['Recall'], 'b-o', linewidth=2, markersize=6, label='Recall (敏感度)')
    ax1.plot(df['Threshold'], df['Specificity'], 'g-s', linewidth=2, markersize=6, label='Specificity (特异度)')
    ax1.plot(df['Threshold'], df['Miss_Rate'], 'r-^', linewidth=2, markersize=6, label='Miss Rate (漏诊率)')
    ax1.axvline(x=best_thr, color='purple', linestyle='--', linewidth=2,
                label=f'当前阈值={best_thr:.2f}')
    ax1.axvline(x=df.loc[best_f2_idx, 'Threshold'], color='orange', linestyle=':', linewidth=2,
                label=f'F2最优阈值={df.loc[best_f2_idx, "Threshold"]:.2f}')
    ax1.set_xlabel('阈值', fontsize=12)
    ax1.set_ylabel('指标值', fontsize=12)
    ax1.set_title(f'{dataset_name} - 阈值-性能曲线', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # 右图：F2分数和净获益（修复颜色格式）
    ax2 = axes[1]
    # 修复：将 'purple-o' 拆分为 color='purple', marker='o'
    ax2.plot(df['Threshold'], df['F2_Score'], color='purple', marker='o',
             linestyle='-', linewidth=2, markersize=6, label='F2分数')
    ax2.plot(df['Threshold'], df['Net_Benefit'], color='orange', marker='s',
             linestyle='-', linewidth=2, markersize=6, label='净获益')
    ax2.axvline(x=best_thr, color='purple', linestyle='--', linewidth=2,
                label=f'当前阈值={best_thr:.2f}')
    ax2.set_xlabel('阈值', fontsize=12)
    ax2.set_ylabel('分数', fontsize=12)
    ax2.set_title(f'{dataset_name} - F2分数与净获益曲线', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/threshold_sensitivity_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ 阈值敏感性分析已保存: {save_dir}/threshold_sensitivity_{dataset_name}.csv/.png")

    return df

# ======================
# 阈值敏感性分析（调用）
# ======================
print("\n" + "=" * 80)
print("【阈值敏感性分析】")
print("=" * 80)

# 使用全局变量（确保 y_test, best_prob, best_model_name, best_thr 已定义）
threshold_df_china = threshold_sensitivity_analysis(
    y_true=y_test,
    y_prob=best_prob,
    model_name=best_model_name,
    dataset_name='Chinese_Clinical_Cohort',
    save_dir='results_china'  # 或改为您的结果保存路径
)

# ============================================================
# 在 chinese_deal.py 末尾导入并调用
# ============================================================

from analysis_utils import (
    threshold_sensitivity_analysis,
    cost_sensitivity_analysis,
    temporal_validation,
    benchmark_inference,
    generate_performance_summary
)

CONFIG = {
    'output_dir': 'results_china',
    'threshold_start': 0.10,
    'threshold_end': 0.70,
    'threshold_step': 0.05,
    'benchmark_iterations': 1000,
    'benchmark_batch_size': 100,
}

# 1. 阈值敏感性分析
threshold_df_china = threshold_sensitivity_analysis(
    y_true=y_test,
    y_prob=best_prob,
    current_threshold=best_thr,
    dataset_name='Chinese_Clinical_Cohort',
    output_dir=CONFIG['output_dir']
)

# 2. 推理性能测试
benchmark_result_china = benchmark_inference(
    model=best_model,
    X_test=X_test_sel,
    dataset_name='Chinese_Clinical_Cohort',
    output_dir=CONFIG['output_dir'],
    n_iterations=CONFIG['benchmark_iterations']
)

# 3. 性能汇总报告
summary_china = generate_performance_summary(
    y_true=y_test,
    y_prob=best_prob,
    y_pred=y_pred_best,
    model_name=best_model_name,
    dataset_name='Chinese_Clinical_Cohort',
    threshold=best_thr,
    cv_scores=None,
    cm=cm,
    output_dir=CONFIG['output_dir']
)

# 4. 经济敏感性分析（使用您论文中的数据）
cost_df = cost_sensitivity_analysis(
    n_screen=1000,
    prevalence=0.303,
    baseline_miss_rate=0.132,
    model_miss_rate=0.039,
    baseline_fp_rate=0.057,
    model_fp_rate=0.129,
    output_dir=CONFIG['output_dir']
)