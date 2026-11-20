# -*- coding: utf-8 -*-
"""
12_R4_配对bootstrap_AUC差值.py — R4：Voting 与 LGB / Stacking 的 AUC 差值配对 Bootstrap 95%CI
对应外部复核意见 R4：用同一测试集（n=1,219）上 2,000 次配对重采样，
计算 Voting−LGB 与 Voting−Stacking 的 AUC 差值 95%CI，
用于"top models 之间差异 CI 包含 0，选择是程序性结果而非优越性证据"的表述。
数据来源：results_china/y_test.npy + results_china/all_probs.pkl
     （2 号主分析脚本保存的真实测试集预测概率；运行时先复核各模型 AUC 与表2 一致）。
输出：results_extra/R4_paired_bootstrap_auc_diff.csv
本脚本不写死任何结果数字，全部由运行时计算。
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.makedirs('results_extra', exist_ok=True)
N_BOOT = 2000
SEED = 42

if __name__ == '__main__':
    y = np.load('results_china/y_test.npy')
    probs = pickle.load(open('results_china/all_probs.pkl', 'rb'))
    print(f'测试集 n={len(y)}，阳性 {int(y.sum())}')

    auc_point = {}
    for k in ['Voting', 'LGB', 'Stacking']:
        auc_point[k] = roc_auc_score(y, np.asarray(probs[k]).ravel())
        print(f'  {k}: AUC = {auc_point[k]:.6f}（与表2 核对）')

    rng = np.random.default_rng(SEED)
    n = len(y)
    diffs = {'Voting-LGB': [], 'Voting-Stacking': []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        a_v = roc_auc_score(y[idx], np.asarray(probs['Voting']).ravel()[idx])
        a_l = roc_auc_score(y[idx], np.asarray(probs['LGB']).ravel()[idx])
        a_s = roc_auc_score(y[idx], np.asarray(probs['Stacking']).ravel()[idx])
        diffs['Voting-LGB'].append(a_v - a_l)
        diffs['Voting-Stacking'].append(a_v - a_s)

    rows = []
    for comp, v in diffs.items():
        arr = np.asarray(v)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        a, b = comp.split('-')
        point = auc_point[a] - auc_point[b]
        rows.append({'comparison': comp,
                     'point_diff_AUC': round(point, 4),
                     'CI95_low': round(float(lo), 4),
                     'CI95_high': round(float(hi), 4),
                     'CI含0': bool(lo <= 0 <= hi),
                     'n_boot': len(arr)})
        print(f"\n{comp}: AUC 差值 {point:+.4f}，95%CI [{lo:+.4f}, {hi:+.4f}]"
              f"（{'包含' if lo <= 0 <= hi else '不包含'} 0）")

    pd.DataFrame(rows).to_csv('results_extra/R4_paired_bootstrap_auc_diff.csv',
                              index=False, encoding='utf-8-sig')
    print('\n已保存: results_extra/R4_paired_bootstrap_auc_diff.csv')
