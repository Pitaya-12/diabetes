# -*- coding: utf-8 -*-
"""
18_Pima_Stacking补充指标.py — Pima Stacking（新最优模型）改稿所需补充指标
若按 (a) 把 Pima 最优模型换成 Stacking，以下数字论文要用，本脚本一次算齐：
  1) 测试集混淆矩阵@0.25 与 Recall/Specificity/PPV/F2/漏诊率（表3/3.2.1 用）
  2) Brier + Bootstrap 95%CI（3.5 节用；另给 XGB 同口径核对）
  3) 推理耗时 + 模型大小（3.7 节若要换成 Stacking 用；同款协议 3 轮×1000 次）
口径与主分析一致（7:3 种子42 → LASSO → 网格搜索 → Stacking）。
输出：results_extra/R7_pima_stacking_metrics.csv
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score

from 工具函数 import bootstrap_auc_ci, brier_with_bootstrap_ci
from 补充实验工具 import qc_pima, run_pipeline

os.makedirs('results_extra', exist_ok=True)
# 阈值动态读主分析表1（2026-11-10：主分析填补口径更新后阈值可能移动，不再写死）
_t1 = pd.read_csv('results_pima/table1_model_comparison.csv').set_index('Model')
THR = float(_t1.loc['Stacking', 'Threshold'])       # Stacking 在表1 的 F2 最优阈值
THR_XGB = float(_t1.loc['XGB', 'Threshold'])        # XGB 在表1 的 F2 最优阈值

if __name__ == '__main__':
    X, y = qc_pima()
    y = y.astype(int)
    res = run_pipeline(X, y, split_seed=42, cohort='pima', want_stacking=True)
    yte = res['y_test'].values

    rows = []
    for name in ['Stacking', 'XGB']:
        prob = res['probs'][name]
        # Stacking/XGB 均用主分析表1 各自的 F2 最优阈值（动态读取）
        thr = THR if name == 'Stacking' else THR_XGB
        yp = (prob > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
        rec = recall_score(yte, yp)
        prec = precision_score(yte, yp)
        spec = tn / (tn + fp)
        f2 = 5 * prec * rec / (4 * prec + rec)
        auc_pt, auc_lo, auc_hi = bootstrap_auc_ci(yte, prob, n_boot=2000, seed=42)
        b = brier_with_bootstrap_ci(yte, prob, n_boot=2000, seed=42)
        miss = fn / (tp + fn)
        rows.append({'model': name, 'threshold': thr,
                     'AUC': round(auc_pt, 4), 'AUC_CI': f'{auc_lo:.3f}-{auc_hi:.3f}',
                     'Recall': round(rec, 4), 'Specificity': round(spec, 4),
                     'PPV': round(prec, 4), 'F2': round(f2, 4),
                     'miss_rate': round(miss, 4),
                     'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
                     'Brier': round(b['brier'], 4),
                     'Brier_CI': f"{b['ci_low']:.4f}-{b['ci_high']:.4f}"})
        print(f"[{name}@{thr}] AUC={auc_pt:.3f}({auc_lo:.3f}-{auc_hi:.3f}) "
              f"Rec={rec:.3f} Spec={spec:.3f} PPV={prec:.3f} F2={f2:.3f} "
              f"TP/FP/FN/TN={tp}/{fp}/{fn}/{tn} Brier={b['brier']:.4f}({b['ci_low']:.4f}-{b['ci_high']:.4f})")

    # 推理耗时（与 8 号脚本同款协议：预热 10 次 + 3 轮×1000 次单样本 + 批量 100，同会话）
    model = res['stacking']
    Xte = res['Xte_s']
    single = Xte[0:1]
    batch = Xte[:100]
    for _ in range(10):
        model.predict_proba(single)
    s_times, b_times = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(1000):
            model.predict_proba(single)
        s_times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        model.predict_proba(batch)
        b_times.append((time.perf_counter() - t0) * 1000)
    size_mb = len(pickle.dumps(model)) / (1024 * 1024)
    print(f"\nStacking 推理耗时: 单样本 {np.mean(s_times):.3f}±{np.std(s_times, ddof=1):.3f} ms | "
          f"批量100 {np.mean(b_times):.3f}±{np.std(b_times, ddof=1):.3f} ms | 模型 {size_mb:.2f} MB")
    print('（3.7 节若换 Stacking 报此数；Pima XGBoost 现有 1.372±0.007 / 1.492±0.049 ms / 0.11 MB 可保留作对照）')

    out = pd.DataFrame(rows)
    out.to_csv('results_extra/R7_pima_stacking_metrics.csv', index=False,
               encoding='utf-8-sig')
    print('\n已保存: results_extra/R7_pima_stacking_metrics.csv')
