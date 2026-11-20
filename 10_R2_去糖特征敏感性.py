# -*- coding: utf-8 -*-
"""
10_R2_去糖特征敏感性.py — R2：剔除血糖派生特征后的敏感性分析
对应外部复核意见 R2：§2.1.3 用 FPG≥7.0 定义结局、§3.1/§3.5 第一大变量是血糖相关特征，
审稿人可能质疑"模型在复现诊断阈值"。本分析剔除全部血糖派生特征后重跑主分析管线。
剔除清单（凡含血糖派生的一律剔除）：
  PUC  : FPG, FFPG, Glucose_BMI(FPG×BMI), Metabolic_Risk(含 I(FPG>5.6))
  Pima : Glucose, Glucose_BMI(Glucose×BMI), Insulin_Glucose_Ratio(含 Glucose),
         Metabolic_Risk(含 I(Glucose>99))
  注：Pima 的 Insulin / Insulin_Log 为胰岛素（非血糖派生），按清单保留。
管线：与主分析完全一致（7:3 分层划分种子42 → LASSO 仅训练集 → 网格搜索 → 集成），
     阈值按主分析同款规则（0.10~0.70 步长 0.05 最大化 F2）在测试集上选取。
输出：results_extra/R2_desugared_sensitivity.csv
本脚本不写死任何结果数字，全部由运行时计算。
"""
import os
import numpy as np
import pandas as pd

from 工具函数 import bootstrap_auc_ci
from 补充实验工具 import (qc_puc, add_features_puc, qc_pima,
                      run_pipeline, f2_best_threshold, metrics_at)

os.makedirs('results_extra', exist_ok=True)

DROP_PUC = ['FPG', 'FFPG', 'Glucose_BMI', 'Metabolic_Risk']
DROP_PIMA = ['Glucose', 'Glucose_BMI', 'Insulin_Glucose_Ratio', 'Metabolic_Risk']


def one_row(tag, cohort, model_name, res, drop_list):
    y = res['y_test'].values
    prob = res['probs'][model_name]
    thr, _ = f2_best_threshold(y, prob)
    met = metrics_at(y, prob, thr)
    auc_pt, auc_lo, auc_hi = bootstrap_auc_ci(y, prob, n_boot=2000, seed=42)
    # 固定 0.35 阈值下的补充指标（供与主分析 Voting 行对照）
    met35 = metrics_at(y, prob, 0.35)
    dropped = [c for c in drop_list]
    row = {'实验': tag, '队列': cohort, '模型': model_name,
           '剔除特征': '+'.join(dropped),
           '候选特征数(剔糖后)': len(res['coef']),
           '保留特征数': len(res['keep']),
           '最终特征清单': ', '.join(res['keep']),
           '阈值(F2最优)': round(float(thr), 2),
           'AUC': round(auc_pt, 4),
           'AUC95%CI': f'{auc_lo:.3f}-{auc_hi:.3f}',
           'Recall': round(met['Recall'], 4), 'Specificity': round(met['Specificity'], 4),
           'PPV': round(met['PPV'], 4), 'F2': round(met['F2'], 4),
           'TP': met['TP'], 'FP': met['FP'], 'FN': met['FN'], 'TN': met['TN'],
           '补:Recall@0.35': round(met35['Recall'], 4),
           '补:Specificity@0.35': round(met35['Specificity'], 4),
           '补:PPV@0.35': round(met35['PPV'], 4),
           '补:F2@0.35': round(met35['F2'], 4),
           '补:TP/FP/FN/TN@0.35': f"{met35['TP']}/{met35['FP']}/{met35['FN']}/{met35['TN']}"}
    print(f"\n[{tag}] {cohort} {model_name}")
    print(f"  剔糖后候选 {row['候选特征数(剔糖后)']} → LASSO 保留 {row['保留特征数']}: {row['最终特征清单']}")
    print(f"  阈值(F2最优)={row['阈值(F2最优)']}  AUC={row['AUC']} ({row['AUC95%CI']})")
    print(f"  Recall={row['Recall']:.3f} Spec={row['Specificity']:.3f} "
          f"PPV={row['PPV']:.3f} F2={row['F2']:.3f}  TP/FP/FN/TN="
          f"{row['TP']}/{row['FP']}/{row['FN']}/{row['TN']}")
    return row


if __name__ == '__main__':
    rows = []

    # ---- PUC：Voting + XGBoost 两行 ----
    print('=' * 70)
    print('PUC 队列（剔除 FPG/FFPG/Glucose_BMI/Metabolic_Risk）')
    df = qc_puc()
    X = add_features_puc(df.drop('Diabetes', axis=1))
    y = df['Diabetes'].astype(int)
    res = run_pipeline(X, y, split_seed=42, cohort='puc',
                       drop_cols=DROP_PUC, want_stacking=False)
    rows.append(one_row('R2-去糖', 'PUC', 'Voting', res, DROP_PUC))
    rows.append(one_row('R2-去糖', 'PUC', 'XGB', res, DROP_PUC))

    # ---- Pima：XGBoost 一行 ----
    print('=' * 70)
    print('Pima 队列（剔除 Glucose/Glucose_BMI/Insulin_Glucose_Ratio/Metabolic_Risk）')
    Xp, yp = qc_pima()
    yp = yp.astype(int)
    res = run_pipeline(Xp, yp, split_seed=42, cohort='pima',
                       drop_cols=DROP_PIMA, want_stacking=False)
    rows.append(one_row('R2-去糖', 'Pima', 'XGB', res, DROP_PIMA))

    out = pd.DataFrame(rows)
    out.to_csv('results_extra/R2_desugared_sensitivity.csv', index=False,
               encoding='utf-8-sig')
    print('\n已保存: results_extra/R2_desugared_sensitivity.csv')
