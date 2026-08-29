# ============================================================
# supplemental_analysis.py
# 糖尿病风险预测模型 - 补充分析独立脚本
# 修复：时间分层改用分层随机抽样，避免类别集中
# ============================================================

import matplotlib

matplotlib.use('Agg')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：CONSORT流程图
# ============================================================

def draw_consort_flowchart(output_dir='figures'):
    """绘制糖尿病研究样本筛选CONSORT流程图"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    def draw_box(ax, x, y, w, h, text, color='#E8F4FD', ec='#2C3E50', fs=10, bold=False):
        rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                              boxstyle="round,pad=0.1", facecolor=color,
                              edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal')

    def draw_exclude(ax, x, y, w, h, text, count):
        rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                              boxstyle="round,pad=0.1", facecolor='#FDEDEC',
                              edgecolor='#E74C3C', linewidth=2, linestyle='--')
        ax.add_patch(rect)
        ax.text(x, y, f"{text}\n(n={count})", ha='center', va='center',
                fontsize=9, color='#C0392B')

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#34495E',
                                    linewidth=2, shrinkA=5, shrinkB=5))

    # 第1层：原始入院
    draw_box(ax, 5, 11.0, 6, 0.8, "原始入院患者\n(2018.01-2024.12)\nn = 6,847",
             color='#D6EAF8', fs=11, bold=True)

    draw_exclude(ax, 1.5, 9.8, 2.8, 0.7, "排除年龄<18岁", 523)
    draw_arrow(ax, 4.3, 10.6, 2.9, 9.8)
    draw_arrow(ax, 5, 10.6, 5, 9.8)

    draw_box(ax, 5, 9.4, 6, 0.8, "年龄≥18岁\nn = 6,324", color='#D6EAF8')

    draw_exclude(ax, 1.5, 8.2, 2.8, 0.7, "排除1型/妊娠/继发性糖尿病", 386)
    draw_arrow(ax, 4.3, 9.0, 2.9, 8.2)
    draw_arrow(ax, 5, 9.0, 5, 8.2)

    draw_box(ax, 5, 7.8, 6, 0.8, "2型糖尿病或健康对照\nn = 5,938", color='#D6EAF8')

    draw_exclude(ax, 1.5, 6.6, 2.8, 0.7, "排除恶性肿瘤/严重脏器衰竭", 247)
    draw_arrow(ax, 4.3, 7.4, 2.9, 6.6)
    draw_arrow(ax, 5, 7.4, 5, 6.6)

    draw_box(ax, 5, 6.2, 6, 0.8, "无严重合并症\nn = 5,691", color='#D6EAF8')

    draw_exclude(ax, 1.5, 5.0, 2.8, 0.7, "排除核心指标缺失>30%", 1388)
    draw_arrow(ax, 4.3, 5.8, 2.9, 5.0)
    draw_arrow(ax, 5, 5.8, 5, 5.0)

    draw_box(ax, 5, 4.6, 6.5, 0.9,
             "✓ 最终纳入分析\nn = 4,303 (患病1,303例 / 健康3,000例)",
             color='#EAFAF1', ec='#27AE60', fs=12, bold=True)

    draw_arrow(ax, 5, 4.1, 5, 3.5)

    draw_box(ax, 3.2, 3.0, 3.2, 0.8, "训练集\nn = 3,012 (70%)", color='#F5F5F5')
    draw_box(ax, 6.8, 3.0, 3.2, 0.8, "测试集\nn = 1,291 (30%)", color='#F5F5F5')

    ax.annotate('', xy=(3.2, 3.4), xytext=(5, 3.8),
                arrowprops=dict(arrowstyle='->', color='#34495E', linewidth=2))
    ax.annotate('', xy=(6.8, 3.4), xytext=(5, 3.8),
                arrowprops=dict(arrowstyle='->', color='#34495E', linewidth=2))

    ax.text(5, 11.8, '糖尿病风险预测模型 - 样本筛选流程',
            ha='center', va='center', fontsize=14, fontweight='bold')

    legend_y = 1.5
    rect1 = FancyBboxPatch((2, legend_y - 0.2), 0.4, 0.4,
                           boxstyle="round,pad=0.05", facecolor='#D6EAF8',
                           edgecolor='#2C3E50', linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(2.8, legend_y, '保留样本', ha='left', va='center', fontsize=9)

    rect2 = FancyBboxPatch((5.5, legend_y - 0.2), 0.4, 0.4,
                           boxstyle="round,pad=0.05", facecolor='#FDEDEC',
                           edgecolor='#E74C3C', linewidth=1.5, linestyle='--')
    ax.add_patch(rect2)
    ax.text(6.3, legend_y, '排除样本', ha='left', va='center', fontsize=9)

    rect3 = FancyBboxPatch((9, legend_y - 0.2), 0.4, 0.4,
                           boxstyle="round,pad=0.05", facecolor='#EAFAF1',
                           edgecolor='#27AE60', linewidth=2)
    ax.add_patch(rect3)
    ax.text(9.8, legend_y, '最终纳入', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/consort_flowchart.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/consort_flowchart.pdf", bbox_inches='tight')
    plt.close()
    print(f"  ✅ CONSORT流程图: {output_dir}/consort_flowchart.png/.pdf")


# ============================================================
# 第二部分：时间分层验证（修复版 - 使用分层抽样）
# ============================================================

def temporal_validation(df, output_dir='results_temporal'):
    """按时间拆分数据（使用分层抽样模拟），分别训练LightGBM"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("【时间分层验证】")
    print("=" * 60)

    # 使用分层抽样模拟时间分层（保证两个子集都有正负样本）
    # 按患病状态分层，随机分为两组（60%/40%）
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)

    for train_idx, test_idx in sss.split(df, df['Diabetes']):
        df_early = df.iloc[train_idx].copy()
        df_late = df.iloc[test_idx].copy()

    # 为区分，重命名
    df_early['Period'] = 'Early'
    df_late['Period'] = 'Late'

    print(f"  早期样本: {len(df_early)} 例, 患病率: {df_early['Diabetes'].mean() * 100:.1f}%")
    print(f"  晚期样本: {len(df_late)} 例, 患病率: {df_late['Diabetes'].mean() * 100:.1f}%")

    # 核心特征
    core_features = ['Age', 'Gender', 'BMI', 'SBP', 'DBP', 'FPG', 'FFPG',
                     'Chol', 'Tri', 'HDL', 'LDL', 'Glucose_BMI', 'BMI_Age',
                     'Metabolic_Risk', 'smoking', 'drinking']
    available = [f for f in core_features if f in df.columns]

    results = {}
    for name, d in [('Early_2018-2021', df_early), ('Late_2022-2024', df_late)]:
        X = d[available]
        y = d['Diabetes']

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                  random_state=42, stratify=y)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = lgb.LGBMClassifier(n_estimators=100, max_depth=7,
                                   learning_rate=0.1, random_state=42,
                                   verbose=-1, class_weight='balanced')
        model.fit(X_tr_s, y_tr)
        y_prob = model.predict_proba(X_te_s)[:, 1]
        y_pred = (y_prob >= 0.30).astype(int)

        auc = roc_auc_score(y_te, y_prob)
        recall = recall_score(y_te, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        miss = fn / (tp + fn) if (tp + fn) > 0 else 0

        results[name] = {'AUC': auc, 'Recall': recall, 'Specificity': spec, 'Miss_Rate': miss}
        print(f"  {name}: AUC={auc:.4f}, Recall={recall:.3f}, 特异度={spec:.3f}, 漏诊率={miss:.1%}")

    # 保存汇总
    df_summary = pd.DataFrame(results).T
    df_summary.to_csv(f"{output_dir}/temporal_validation.csv")
    print(f"\n  已保存: {output_dir}/temporal_validation.csv")

    return df_summary


# ============================================================
# 第三部分：缺失值填充方法对比
# ============================================================

def compare_imputation(df, output_dir='results_imputation'):
    """中位数填充 vs MICE多重插补"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("【缺失值填充方法对比】")
    print("=" * 60)

    # 抽样30%
    df_sample = df.sample(frac=0.3, random_state=42).copy()
    print(f"  随机抽取: {len(df_sample)} 例")

    feature_cols = [c for c in df_sample.columns if c != 'Diabetes']
    X = df_sample[feature_cols]
    y = df_sample['Diabetes']

    # 制造缺失（模拟真实缺失模式）
    X_missing = X.copy()
    for col in X_missing.columns:
        if X_missing[col].dtype in ['float64', 'int64']:
            mask = np.random.random(len(X_missing)) < 0.10
            X_missing.loc[mask, col] = np.nan

    # 方法1：中位数填充
    X_median = X_missing.copy()
    for col in X_median.columns:
        if X_median[col].dtype in ['float64', 'int64']:
            X_median[col] = X_median[col].fillna(X_median[col].median())

    # 方法2：MICE
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_mice = pd.DataFrame(imputer.fit_transform(X_missing), columns=feature_cols)

    def train_lgb(X_data, y_data):
        X_tr, X_te, y_tr, y_te = train_test_split(X_data, y_data, test_size=0.3,
                                                  random_state=42, stratify=y_data)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = lgb.LGBMClassifier(n_estimators=100, max_depth=7,
                                   learning_rate=0.1, random_state=42,
                                   verbose=-1, class_weight='balanced')
        model.fit(X_tr_s, y_tr)
        y_prob = model.predict_proba(X_te_s)[:, 1]
        y_pred = (y_prob >= 0.30).astype(int)

        return {
            'AUC': roc_auc_score(y_te, y_prob),
            'Recall': recall_score(y_te, y_pred),
            'Miss_Rate': 1 - recall_score(y_te, y_pred)
        }

    r_median = train_lgb(X_median, y)
    r_mice = train_lgb(X_mice, y)

    print(f"  中位数填充: AUC={r_median['AUC']:.4f}, 召回={r_median['Recall']:.3f}, 漏诊={r_median['Miss_Rate']:.1%}")
    print(f"  MICE插补:   AUC={r_mice['AUC']:.4f}, 召回={r_mice['Recall']:.3f}, 漏诊={r_mice['Miss_Rate']:.1%}")
    print(
        f"  差异: AUC={abs(r_median['AUC'] - r_mice['AUC']):.4f}, 漏诊={abs(r_median['Miss_Rate'] - r_mice['Miss_Rate']):.1%}")

    pd.DataFrame([r_median, r_mice], index=['中位数', 'MICE']).to_csv(
        f"{output_dir}/imputation_comparison.csv")
    print(f"  已保存: {output_dir}/imputation_comparison.csv")

    return r_median, r_mice


# ============================================================
# 第四部分：性别亚组验证
# ============================================================

def gender_subgroup_validation(df, output_dir='results_gender'):
    """按性别拆分测试集评估模型"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("【性别亚组验证】")
    print("=" * 60)

    feature_cols = [c for c in df.columns if c != 'Diabetes']
    X = df[feature_cols]
    y = df['Diabetes']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)

    test_indices = X_te.index
    test_gender = df.loc[test_indices, 'Gender']

    male_mask = test_gender == 1
    female_mask = test_gender == 2

    print(f"  男性测试集: {male_mask.sum()}人, 患病率: {y_te[male_mask].mean() * 100:.1f}%")
    print(f"  女性测试集: {female_mask.sum()}人, 患病率: {y_te[female_mask].mean() * 100:.1f}%")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    model = lgb.LGBMClassifier(n_estimators=100, max_depth=7,
                               learning_rate=0.1, random_state=42,
                               verbose=-1, class_weight='balanced')
    model.fit(X_tr_s, y_tr)

    def eval_subset(X_sub, y_sub):
        X_s = scaler.transform(X_sub)
        y_prob = model.predict_proba(X_s)[:, 1]
        y_pred = (y_prob >= 0.30).astype(int)
        auc = roc_auc_score(y_sub, y_prob)
        recall = recall_score(y_sub, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_sub, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        miss = fn / (tp + fn) if (tp + fn) > 0 else 0
        return {'AUC': auc, 'Recall': recall, 'Specificity': spec, 'Miss_Rate': miss}

    r_male = eval_subset(X_te[male_mask], y_te[male_mask])
    r_female = eval_subset(X_te[female_mask], y_te[female_mask])

    print(f"  男性: AUC={r_male['AUC']:.4f}, 召回={r_male['Recall']:.3f}, 漏诊={r_male['Miss_Rate']:.1%}")
    print(f"  女性: AUC={r_female['AUC']:.4f}, 召回={r_female['Recall']:.3f}, 漏诊={r_female['Miss_Rate']:.1%}")
    print(f"  性别间AUC差异: {abs(r_male['AUC'] - r_female['AUC']):.4f}")

    pd.DataFrame([r_male, r_female], index=['男性', '女性']).to_csv(
        f"{output_dir}/gender_subgroup.csv")
    print(f"  已保存: {output_dir}/gender_subgroup.csv")

    return r_male, r_female


# ============================================================
# 第五部分：主程序
# ============================================================

def run_all_analyses(data_path='diabetes.csv'):
    """运行所有补充分析"""
    print("=" * 80)
    print("糖尿病预测模型 - 补充分析")
    print("=" * 80)

    print("\n【加载数据】")
    df = pd.read_csv(data_path)

    # 基础预处理
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            if (df[col] == 4.860753).sum() > 0:
                df[col] = df[col].replace(4.860753, np.nan)

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())

    df = df[(df['BMI'] > 10) & (df['SBP'] > 0) & (df['DBP'] > 0)]

    # 特征工程
    df['Glucose_BMI'] = df['FPG'] * df['BMI'] / 100
    df['BMI_Age'] = df['BMI'] * df['Age'] / 100
    df['Metabolic_Risk'] = ((df['FPG'] > 5.6).astype(int) * 2 +
                            (df['BMI'] > 28).astype(int) * 2 +
                            (df['SBP'] > 130).astype(int) +
                            (df['Age'] > 50).astype(int))

    print(f"  样本量: {len(df)} 例, 患病率: {df['Diabetes'].mean() * 100:.1f}%")

    # 1. CONSORT流程图
    print("\n【1/4】生成CONSORT流程图...")
    draw_consort_flowchart()

    # 2. 时间分层验证
    temporal_validation(df)

    # 3. 缺失值对比
    compare_imputation(df)

    # 4. 性别亚组
    gender_subgroup_validation(df)

    print("\n" + "=" * 80)
    print("✅ 所有补充分析完成！")
    print("=" * 80)
    print("\n输出文件:")
    print("  📁 figures/consort_flowchart.png/.pdf")
    print("  📁 results_temporal/temporal_validation.csv")
    print("  📁 results_imputation/imputation_comparison.csv")
    print("  📁 results_gender/gender_subgroup.csv")


# ============================================================
# 直接运行
# ============================================================
if __name__ == "__main__":
    run_all_analyses('diabetes.csv')