# -*- coding: utf-8 -*-
"""
7_MICE敏感性分析.py — MICE 多重插补敏感性分析（论文 2.2.1 / 附表S2）v2 锚点对齐版
=====================================================================
v1 的问题（2026-11-10 修正，旧版备份在 results_mice/_v1旧口径备份/）：
  1) "主分析"臂不是表1/表2 管线的复跑——硬编码固定超参 XGBoost/Voting + 固定阈值，
     Pima 锚点行 0.8360/88.9%/F2 0.802/漏9 对不上表1 任何一行；
     PUC 锚点行 0.9637/93.4% 与表2（0.965/92.5%）也有偏差。
  2) Pima 主分析最优模型已随改稿换为 Stacking（表1 首行），v1 仍用 XGBoost。

v2 口径：唯一自变量 = 插补方式（中位数 vs MICE）。
  两臂共用同一质控行集（Pima 768 / PUC 4061），各自走完整主分析管线
  （补充实验工具.run_pipeline：LASSO 仅训练集 → 5 模型网格搜索 →
   Voting/Stacking → 与主分析同款 F2 最优阈值规则）。
  验收（硬性）：中位数臂必须逐位复现主分析当次输出——Pima 表1 Stacking 行、
  PUC 表2 Voting 行（运行时从 results_pima/results_china 的模型对比 CSV 读取锚点），
  否则报错退出，禁止把数字写死。
  2026-11-10 口径更新：Pima 中位数臂随主分析改为【非零中位数、只在训练集上算】；
  MICE 臂保持不变（IterativeImputer 全量拟合，与 v1 一致）。
输出：results_mice/mice_sensitivity.csv（增量写入，每臂完成即保存）
运行：python 7_MICE敏感性分析.py（约 15~30 分钟）
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

from 补充实验工具 import run_pipeline, f2_best_threshold, metrics_at

warnings.filterwarnings('ignore')
SEED = 42
os.makedirs('results_mice', exist_ok=True)
OUT_CSV = os.path.join('results_mice', 'mice_sensitivity.csv')

print("=" * 70)
print("MICE 多重插补敏感性分析 v2（锚点对齐版，论文 2.2.1 / 附表S2）")
print("=" * 70)

# ====================== 质控（保留缺失，供两种填补对比；两臂共用同一行集） ======================
def qc_puc_raw():
    """PUC 质控：4303 -> 4061，与 2 号脚本逐行一致，但不做填补（保留缺失）"""
    df = pd.read_csv('diabetes.csv')
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].replace(4.860753, np.nan)
    df = df[(df['BMI'] > 10) & (df['BMI'] < 60) &
            (df['FPG'] > 2.8) & (df['SBP'] > 0) & (df['DBP'] > 0) &
            (df['Age'] >= 18)]
    core = ['FPG', 'BMI', 'Age', 'SBP', 'DBP', 'Chol']
    df = df[df[core].isnull().sum(axis=1) < 3]
    for col in core:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    return df.reset_index(drop=True)


def qc_pima_raw():
    """Pima 质控：读入全部 768 行，0 值留给填补臂处理（与 1 号脚本同口径）"""
    return pd.read_csv('pima_diabetes.csv')


# ====================== 两种填补方式 ======================
def impute_mice(df, round_cols=()):
    """MICE：IterativeImputer 20 次迭代（随机种子 42）；
    填补值截断到观测范围，二值/整数编码列填补后取整。
    插补模型含结局列（多重插补标准做法：结局作为辅助变量进链式方程），
    与 v1 口径一致。"""
    out = df.copy()
    num = [c for c in out.columns if out[c].dtype in ['float64', 'int64']]
    obs_min, obs_max = out[num].min(), out[num].max()
    imp = IterativeImputer(max_iter=20, random_state=SEED)
    out[num] = imp.fit_transform(out[num])
    for c in num:
        out[c] = out[c].clip(obs_min[c], obs_max[c])
    for c in round_cols:
        out[c] = out[c].round()
    return out


def arm_median_puc(raw):
    """中位数填补：与 2 号主分析完全一致（逐列 fillna(该列中位数)）"""
    out = raw.copy()
    for col in out.columns:
        if out[col].dtype in ['float64', 'int64']:
            out[col] = out[col].fillna(out[col].median())
    return out


def arm_mice_puc(raw):
    return impute_mice(raw.copy(),
                       round_cols=('Gender', 'smoking', 'drinking', 'family_history'))


def arm_median_pima(raw):
    """中位数填补 v2 口径（2026-11-10，与 1 号 v2 一致）：
    0 值视作缺失，以【非零中位数】填补，且中位数只在训练集上估计（防泄漏）"""
    out = raw.copy()
    idx_train, _ = train_test_split(out.index, test_size=0.3, random_state=SEED,
                                    stratify=out['Outcome'])
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        med = out.loc[idx_train, col]
        out[col] = out[col].replace(0, med[med != 0].median())
    return out[(out['BloodPressure'] > 0) & (out['BMI'] > 10)]


def arm_mice_pima(raw):
    """MICE 填补：0 值视作缺失后插补"""
    out = raw.copy()
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        out[col] = out[col].replace(0, np.nan)
    out = impute_mice(out, round_cols=('Pregnancies',))
    return out[(out['BloodPressure'] > 0) & (out['BMI'] > 10)]


# ====================== 特征工程（与 1/2 号主分析完全一致） ======================
def add_features_puc(X):
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


def add_features_pima(X):
    X = X.copy()
    X['Glucose_BMI'] = X['Glucose'] * X['BMI'] / 100
    X['Age_Pregnancies'] = X['Age'] * X['Pregnancies'] / 10
    X['Insulin_Glucose_Ratio'] = X['Insulin'] / (X['Glucose'] + 1)
    X['BMI_Age'] = X['BMI'] * X['Age'] / 100
    X['SkinThickness_BMI'] = X['SkinThickness'] / (X['BMI'] + 1)
    X['Metabolic_Risk'] = ((X['Glucose'] > 99).astype(int) * 2 +
                           (X['BMI'] > 30).astype(int) * 2 +
                           (X['BloodPressure'] > 80).astype(int) +
                           (X['Age'] > 40).astype(int))
    X['Insulin_Log'] = np.log1p(X['Insulin'])
    X['DPF_Log'] = np.log1p(X['DiabetesPedigreeFunction'])
    return X


# ====================== 锚点（运行时读主分析结果，硬性验收基准） ======================
# Pima: 表1 Stacking 行（results_pima/table1_model_comparison.csv，1 号脚本输出）
# PUC : 表2 Voting 行（results_china/table2_model_comparison.csv，2 号脚本输出）
# 2026-11-10 改为动态读取：主分析口径更新（如 Pima 填补 v2）后锚点自动跟随，
# 中位数臂必须逐位复现主分析当次输出。
def load_anchor():
    t1 = pd.read_csv('results_pima/table1_model_comparison.csv').set_index('Model')
    t2 = pd.read_csv('results_china/table2_model_comparison.csv').set_index('Model')
    s, v = t1.loc['Stacking'], t2.loc['Voting']
    return {
        ('Pima', '中位数填补(主分析)'): {
            'thr': float(s['Threshold']), 'AUC': float(s['AUC']),
            'Recall': float(s['Recall']), 'F2': float(s['F2']),
            'Missed': int(s['Missed'])},
        ('PUC', '中位数填补(主分析)'): {
            'thr': float(v['Threshold']), 'AUC': float(v['AUC']),
            'Recall': float(v['Recall']), 'F2': float(v['F2']),
            'Missed': int(v['Missed'])},
    }


ANCHOR = load_anchor()
print("锚点来源: results_pima/table1_model_comparison.csv (Stacking 行) / "
      "results_china/table2_model_comparison.csv (Voting 行)")
MAIN_THR = {'Pima': 0.25, 'PUC': 0.35}   # 主分析最优阈值（固定阈值对照用）


def check_anchor(cohort, tag, row):
    """中位数臂必须逐位复现表1/表2；对不上就报错退出"""
    exp = ANCHOR.get((cohort, tag))
    if exp is None:
        return
    ok = (abs(row['阈值'] - exp['thr']) < 1e-9 and
          abs(row['AUC'] - exp['AUC']) < 1e-6 and
          abs(row['召回率'] - exp['Recall']) < 1e-6 and
          abs(row['F2'] - exp['F2']) < 1e-6 and
          row['漏诊'] == exp['Missed'])
    status = 'PASS 与主分析逐位一致' if ok else 'FAIL 与主分析不一致!'
    print(f"  [锚点验收] {cohort}-{tag}: "
          f"thr={row['阈值']:.2f} AUC={row['AUC']:.4f} 召回={row['召回率']:.4f} "
          f"F2={row['F2']:.4f} 漏诊={row['漏诊']}  -> {status}")
    if not ok:
        print(f"    期望: thr={exp['thr']:.2f} AUC={exp['AUC']:.4f} "
              f"召回={exp['Recall']:.4f} F2={exp['F2']:.4f} 漏诊={exp['Missed']}")
        print("    锚点对不上——禁止写死数字，请先修管线。退出。")
        sys.exit(1)


# ====================== 一条评估流水线（全主分析口径） ======================
def run_arm(raw, cohort, tag, impute_fn, add_feat, target, model_key):
    print(f"\n【{cohort}】{tag}（主模型 {model_key}，全管线：LASSO→网格搜索→集成→F2 阈值）")
    df = impute_fn(raw)
    X = add_feat(df.drop(target, axis=1))
    y = df[target].astype(int)
    res = run_pipeline(X, y, split_seed=SEED, cohort=cohort.lower(),
                       want_stacking=True)
    yte = res['y_test'].values
    prob = res['probs'][model_key]

    thr, _ = f2_best_threshold(yte, prob)          # 每臂自身 F2 最优阈值（主分析同款规则）
    m = metrics_at(yte, prob, thr)
    m_fix = metrics_at(yte, prob, MAIN_THR[cohort])  # 主分析固定阈值对照

    row = {'队列': cohort, '填补方式': tag, '主模型': model_key,
           '特征数': len(res['keep']),
           '阈值': round(float(thr), 2),
           'AUC': float(m['AUC']), '召回率': float(m['Recall']),
           'PPV': float(m['PPV']), 'F2': float(m['F2']),
           '漏诊': int(m['FN']), 'TP': int(m['TP']), 'FP': int(m['FP']),
           'FN': int(m['FN']), 'TN': int(m['TN']),
           '固定主阈值': MAIN_THR[cohort],
           '固定阈值_召回率': float(m_fix['Recall']),
           '固定阈值_PPV': float(m_fix['PPV']),
           '固定阈值_F2': float(m_fix['F2']),
           '固定阈值_漏诊': int(m_fix['FN'])}
    print(f"  特征={row['特征数']}, 阈值={row['阈值']:.2f}, AUC={row['AUC']:.4f}, "
          f"召回={row['召回率']:.4f}, PPV={row['PPV']:.4f}, F2={row['F2']:.4f}, "
          f"漏诊={row['漏诊']}")
    check_anchor(cohort, tag, row)
    return row


rows = []
raw_puc = qc_puc_raw()
raw_pima = qc_pima_raw()
print(f"\n质控（两臂共用行集）: PUC {len(raw_puc)} 行 | Pima {len(raw_pima)} 行")

# ====================== PUC 队列（主模型 Voting，表2 口径） ======================
rows.append(run_arm(raw_puc, 'PUC', '中位数填补(主分析)',
                    arm_median_puc, add_features_puc, 'Diabetes', 'Voting'))
pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
rows.append(run_arm(raw_puc, 'PUC', 'MICE(20次迭代)',
                    arm_mice_puc, add_features_puc, 'Diabetes', 'Voting'))
pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

# ====================== Pima 队列（主模型 Stacking，表1 口径） ======================
rows.append(run_arm(raw_pima, 'Pima', '中位数填补(主分析)',
                    arm_median_pima, add_features_pima, 'Outcome', 'Stacking'))
pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
rows.append(run_arm(raw_pima, 'Pima', 'MICE(20次迭代)',
                    arm_mice_pima, add_features_pima, 'Outcome', 'Stacking'))
pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

# ====================== 汇总 + 结论判断 ======================
res = pd.DataFrame(rows)
base = res[res['填补方式'] == '中位数填补(主分析)'].set_index('队列')
res['ΔAUC(vs主分析)'] = [
    round(r['AUC'] - base.loc[r['队列'], 'AUC'], 4) for _, r in res.iterrows()]
res['Δ召回(vs主分析)'] = [
    round(r['召回率'] - base.loc[r['队列'], '召回率'], 4) for _, r in res.iterrows()]
res.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

print("\n===== MICE 敏感性分析汇总（v2 锚点对齐） =====")
print(res.to_string(index=False))
print("\n结论判断（ΔAUC<0.02 且 Δ召回<0.05 为稳定）：")
for cohort in ['PUC', 'Pima']:
    sub = res[res['队列'] == cohort].set_index('填补方式')
    dauc = abs(sub.loc['MICE(20次迭代)', 'AUC'] - sub.loc['中位数填补(主分析)', 'AUC'])
    drec = abs(sub.loc['MICE(20次迭代)', '召回率'] - sub.loc['中位数填补(主分析)', '召回率'])
    ok = dauc < 0.02 and drec < 0.05
    print(f"  {cohort}: AUC 变化 {dauc:.4f}, 召回率变化 {drec:.4f} -> "
          f"{'结论稳定 [OK]' if ok else '存在差异，需如实报告'}")
print(f"\n已保存: {OUT_CSV}")
