# -*- coding: utf-8 -*-
"""
14_Pima_OOF配对差值.py — Pima 队列 OOF 口径下 Stacking vs XGBoost 的配对差值检验
用途：R1 显示 Pima 的 OOF 预设规则选中 Stacking（F2 0.790）而非 XGBoost（0.776），
     两者差距很小；本脚本按 R4 同款口径（2000 次配对重采样）检验该差距是否可区分，
     供"top models 差异 CI 包含 0，选择是程序性结果"表述使用。
口径：Pima 训练集内部 5 折 OOF（折内 LASSO + 折内网格搜索，与 9 号脚本一致），测试集不参与。
输出：results_extra/R4b_pima_oof_paired_bootstrap.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from 补充实验工具 import (qc_pima, lasso_select, train_base_models,
                      build_voting, build_stacking, predict_all,
                      f2_best_threshold, metrics_at, SEED)

os.makedirs('results_extra', exist_ok=True)
N_BOOT = 2000

if __name__ == '__main__':
    X_all, y_all = qc_pima()
    y_all = y_all.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X_all, y_all, test_size=0.3, random_state=SEED, stratify=y_all)
    print(f'Pima 训练集 {len(ytr)} / 测试集 {len(yte)}（测试集不参与）')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(ytr)) for m in ['XGB', 'Stacking']}
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, ytr), 1):
        Xtr_f, ytr_f = Xtr.iloc[tr_idx], ytr.iloc[tr_idx]
        Xva_f = Xtr.iloc[va_idx]
        keep, _ = lasso_select(Xtr_f, ytr_f)
        Xtr_raw, Xva_raw = Xtr_f[keep], Xva_f[keep]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_raw)
        Xva_s = sc.transform(Xva_raw)
        models = train_base_models(Xtr_raw, Xtr_s, ytr_f, 'pima')
        voting = build_voting(models, Xtr_s, ytr_f)
        stacking = build_stacking(models, Xtr_s, ytr_f)
        probs = predict_all(models, voting, stacking, Xva_raw, Xva_s)
        oof['XGB'][va_idx] = probs['XGB']
        oof['Stacking'][va_idx] = probs['Stacking']
        print(f'  fold {fold}/5 完成')

    y = ytr.values
    thr_x, f2_x = f2_best_threshold(y, oof['XGB'])
    thr_s, f2_s = f2_best_threshold(y, oof['Stacking'])
    print(f'OOF 最优阈值: XGB {thr_x:.2f} (F2={f2_x:.4f}) | Stacking {thr_s:.2f} (F2={f2_s:.4f})')

    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    def f2_at(prob, yv, thr):
        """在给定概率与【同步重采样的】标签上算 F2"""
        yp = (prob > thr).astype(int)
        p, r = precision_score(yv, yp), recall_score(yv, yp)
        return 5 * p * r / (4 * p + r) if p + r > 0 else 0.0

    point_auc_s = roc_auc_score(y, oof['Stacking'])
    point_auc_x = roc_auc_score(y, oof['XGB'])
    point_f2_s = f2_at(oof['Stacking'], y, thr_s)
    point_f2_x = f2_at(oof['XGB'], y, thr_x)

    rng = np.random.default_rng(SEED)
    n = len(y)
    d_auc, d_f2 = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        yy = y[idx]
        d_auc.append(roc_auc_score(yy, oof['Stacking'][idx]) -
                     roc_auc_score(yy, oof['XGB'][idx]))
        d_f2.append(f2_at(oof['Stacking'][idx], yy, thr_s) -
                    f2_at(oof['XGB'][idx], yy, thr_x))

    rows = []
    for name, arr, point in [('OOF_AUC Stacking-XGB', d_auc, point_auc_s - point_auc_x),
                             ('OOF_F2 Stacking-XGB', d_f2, point_f2_s - point_f2_x)]:
        arr = np.asarray(arr)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append({'comparison': name, 'point_diff': round(float(point), 4),
                     'CI95_low': round(float(lo), 4), 'CI95_high': round(float(hi), 4),
                     'CI含0': bool(lo <= 0 <= hi), 'n_boot': len(arr)})
        print(f'{name}: {point:+.4f}，95%CI [{lo:+.4f}, {hi:+.4f}]'
              f'（{"包含" if lo <= 0 <= hi else "不包含"} 0）')

    pd.DataFrame(rows).to_csv(
        'results_extra/R4b_pima_oof_paired_bootstrap.csv', index=False, encoding='utf-8-sig')
    print('已保存: results_extra/R4b_pima_oof_paired_bootstrap.csv')
