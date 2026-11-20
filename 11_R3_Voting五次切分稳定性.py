# -*- coding: utf-8 -*-
"""
11_R3_Voting五次切分稳定性.py — R3：主模型 Voting 的五次随机切分稳定性检验
对应外部复核意见 R3：§2.5.3/§3.6 的稳定性检验只重训了 LightGBM，
而主分析选出的主模型是 Voting，需补检验"选出来的那个"。
口径：种子 0~4 各做一次 7:3 分层划分；每次完整重跑主分析管线
     （LASSO 仅训练集 → 5 模型网格搜索 → 软投票 Voting），阈值固定 0.35
     （= 主分析 Voting 最优阈值），报 test AUC 与该阈值下的指标。
输出：results_extra/R3_voting_5split_stability.csv
本脚本不写死任何结果数字，全部由运行时计算。
"""
import os
import numpy as np
import pandas as pd

from 补充实验工具 import qc_puc, add_features_puc, run_pipeline, metrics_at

os.makedirs('results_extra', exist_ok=True)
THRESHOLD = 0.35   # 主分析 Voting 最优阈值，五次切分统一固定


if __name__ == '__main__':
    df = qc_puc()
    X = add_features_puc(df.drop('Diabetes', axis=1))
    y = df['Diabetes'].astype(int)

    rows = []
    for s in [0, 1, 2, 3, 4]:
        res = run_pipeline(X, y, split_seed=s, cohort='puc', want_stacking=False)
        yte = res['y_test'].values
        prob = res['probs']['Voting']
        met = metrics_at(yte, prob, THRESHOLD)
        rows.append({'seed': s,
                     'n_train': res['n_train'], 'n_test': res['n_test'],
                     'LASSO_保留特征数': len(res['keep']),
                     'threshold': THRESHOLD,
                     'test_AUC': round(met['AUC'], 4),
                     'Recall@0.35': round(met['Recall'], 4),
                     'Specificity@0.35': round(met['Specificity'], 4),
                     'PPV@0.35': round(met['PPV'], 4),
                     'F2@0.35': round(met['F2'], 4),
                     'missed@0.35': met['FN']})
        print(f"  seed={s}  test AUC={met['AUC']:.4f}  "
              f"召回@0.35={met['Recall']:.3f}  漏诊={met['FN']}")

    out = pd.DataFrame(rows)
    aucs = out['test_AUC'].values
    recs = out['Recall@0.35'].values
    summary = {
        'seed': ['mean±SD', 'min~max'],
        'n_train': ['', ''], 'n_test': ['', ''],
        'LASSO_保留特征数': ['', ''],
        'threshold': ['', ''],
        'test_AUC': [f'{aucs.mean():.4f} ± {aucs.std(ddof=1):.4f}',
                     f'{aucs.min():.4f} ~ {aucs.max():.4f}'],
        'Recall@0.35': [f'{recs.mean():.4f} ± {recs.std(ddof=1):.4f}',
                        f'{recs.min():.4f} ~ {recs.max():.4f}'],
        'Specificity@0.35': ['', ''], 'PPV@0.35': ['', ''],
        'F2@0.35': ['', ''], 'missed@0.35': ['', ''],
    }
    out = pd.concat([out, pd.DataFrame(summary)], ignore_index=True)
    out.to_csv('results_extra/R3_voting_5split_stability.csv', index=False,
               encoding='utf-8-sig')
    print(f"\nVoting 五次切分 test AUC: {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f} "
          f"(min {aucs.min():.3f} / max {aucs.max():.3f})")
    print('已保存: results_extra/R3_voting_5split_stability.csv')
