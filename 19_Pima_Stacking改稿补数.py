# -*- coding: utf-8 -*-
"""
19_Pima_Stacking改稿补数.py — Pima 最优模型改为 Stacking 后剩余缺口一次补齐
  1) 推理耗时复测（修正 18 号脚本单样本未除次数的计量错误，给出真实单样本/批量耗时）
  2) Stacking 阈值敏感性（0.10~0.70 步长 0.05，F2 与净获益，供 3.3 节替换 Pima 描述）
  3) Stacking 四级风险分级（0.25/0.50/0.75 切点，供表5 Pima 部分）
输出：results_extra/R8_pima_stacking_gapfill.csv（汇总）+ R8_threshold.csv + R8_risk.csv
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score

from 补充实验工具 import qc_pima, run_pipeline, metrics_at, f2_best_threshold, THR_GRID

os.makedirs('results_extra', exist_ok=True)

if __name__ == '__main__':
    X, y = qc_pima()
    y = y.astype(int)
    res = run_pipeline(X, y, split_seed=42, cohort='pima', want_stacking=True)
    yte = res['y_test'].values
    prob = res['probs']['Stacking']

    # ---- 1) 耗时（正确计量：单样本 = 总耗时/次数）----
    model, Xte = res['stacking'], res['Xte_s']
    single, batch = Xte[0:1], Xte[:100]
    for _ in range(10):
        model.predict_proba(single)
    s_ms, b_ms = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(1000):
            model.predict_proba(single)
        s_ms.append((time.perf_counter() - t0) / 1000 * 1000)   # 每次
        t0 = time.perf_counter()
        model.predict_proba(batch)
        b_ms.append((time.perf_counter() - t0) * 1000)          # 每批100
    size_mb = len(pickle.dumps(model)) / (1024 * 1024)
    print(f'Stacking: 单样本 {np.mean(s_ms):.3f}±{np.std(s_ms, ddof=1):.3f} ms | '
          f'批量100 {np.mean(b_ms):.3f}±{np.std(b_ms, ddof=1):.3f} ms | {size_mb:.2f} MB')

    # ---- 2) 阈值敏感性（同 3.3 节口径）----
    rows = []
    for thr in THR_GRID:
        m = metrics_at(yte, prob, thr)
        n = len(yte)
        tp, fp = m['TP'], m['FP']
        nb = max(0, tp / n - fp / n * (thr / (1 - thr))) if thr < 1 else 0
        miss = m['FN'] / (m['TP'] + m['FN'])
        rows.append({'Threshold': thr, 'Recall': round(m['Recall'], 4),
                     'Specificity': round(m['Specificity'], 4),
                     'Miss_Rate': round(miss, 4), 'PPV': round(m['PPV'], 4),
                     'F2': round(m['F2'], 4), 'Net_Benefit': round(nb, 4)})
    tdf = pd.DataFrame(rows)
    tdf.to_csv('results_extra/R8_threshold.csv', index=False, encoding='utf-8-sig')
    best = tdf.loc[tdf['F2'].idxmax()]
    print(f"F2 最优阈值 {best['Threshold']:.2f} (F2={best['F2']:.4f})；"
          f"净获益最优 {tdf.loc[tdf['Net_Benefit'].idxmax(), 'Threshold']:.2f} "
          f"(NB={tdf['Net_Benefit'].max():.3f})")

    # ---- 3) 四级风险分级（0.25/0.50/0.75）----
    def lvl(p):
        return '低风险' if p < 0.25 else '中风险' if p < 0.5 else '高风险' if p < 0.75 else '极高风险'
    rrows = []
    for name, lo, hi in [('低风险', -1, 0.25), ('中风险', 0.25, 0.5),
                         ('高风险', 0.5, 0.75), ('极高风险', 0.75, 1.01)]:
        mask = (prob > lo) & (prob <= hi) if lo >= 0 else (prob <= hi)
        n_lv = int(mask.sum())
        prev = float(yte[mask].mean()) if n_lv else 0.0
        rrows.append({'风险等级': name, '人数': n_lv,
                      '占比': f'{100 * n_lv / len(yte):.1f}%',
                      '实际患病率': f'{100 * prev:.1f}%'})
    rdf = pd.DataFrame(rrows)
    rdf.to_csv('results_extra/R8_risk.csv', index=False, encoding='utf-8-sig')
    print(rdf.to_string(index=False))

    pd.DataFrame([{'model': 'Stacking', 'single_ms': round(float(np.mean(s_ms)), 3),
                   'batch100_ms': round(float(np.mean(b_ms)), 3),
                   'model_mb': round(size_mb, 2),
                   'F2_thr': float(best['Threshold']),
                   'NB_best_thr': float(tdf.loc[tdf['Net_Benefit'].idxmax(), 'Threshold'])}]) \
        .to_csv('results_extra/R8_pima_stacking_gapfill.csv', index=False, encoding='utf-8-sig')
    print('已保存: results_extra/R8_*.csv')
