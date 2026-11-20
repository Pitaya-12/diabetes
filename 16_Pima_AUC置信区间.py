# -*- coding: utf-8 -*-
"""
16_Pima_AUC置信区间.py — Pima 测试集上 XGBoost 与 Stacking 的 AUC Bootstrap 95%CI
用途：若按 (a) 把 Pima 最优模型改为 Stacking，§3.2.3/表3 需要 Stacking 的 AUC 95%CI
     （正文现有 0.774–0.882 是 XGBoost 的）。本脚本按主分析同口径重跑 Pima 主管线
     （7:3 种子42 → LASSO → 网格搜索 → 集成），对两模型各做 2000 次 Bootstrap。
输出：results_extra/R5_pima_auc_ci.csv
"""
import os
import pandas as pd

from 工具函数 import bootstrap_auc_ci
from 补充实验工具 import qc_pima, run_pipeline

os.makedirs('results_extra', exist_ok=True)

if __name__ == '__main__':
    X, y = qc_pima()
    y = y.astype(int)
    res = run_pipeline(X, y, split_seed=42, cohort='pima', want_stacking=True)
    yte = res['y_test'].values

    rows = []
    for name in ['XGB', 'Stacking']:
        pt, lo, hi = bootstrap_auc_ci(yte, res['probs'][name], n_boot=2000, seed=42)
        rows.append({'model': name, 'AUC': round(pt, 4),
                     'CI95_low': round(lo, 4), 'CI95_high': round(hi, 4),
                     'n_test': len(yte), 'n_pos': int(yte.sum())})
        print(f'{name}: AUC {pt:.3f} (95%CI {lo:.3f}-{hi:.3f})')

    pd.DataFrame(rows).to_csv('results_extra/R5_pima_auc_ci.csv', index=False,
                              encoding='utf-8-sig')
    print('已保存: results_extra/R5_pima_auc_ci.csv')
