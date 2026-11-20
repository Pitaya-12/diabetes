# -*- coding: utf-8 -*-
"""
9_R1_OOF七模型对比.py — R1：5 折 OOF 口径的七模型对比（PUC + Pima 各 7 行，共 14 行）
对应外部复核意见 R1：替换"最优模型选择"所用的测试集数字。
口径（关键）：
  · 先按主分析同款 7:3 分层划分（种子 42），OOF 全部在【训练集内部】生成，测试集完全不参与；
  · 训练集内 5 折分层 CV；每个折内【完整重做】LASSO 特征筛选 + 5 模型网格搜索 + Voting/Stacking，
    即特征筛选与调参严格位于 5 折 CV 循环内部（折内），再预测留出折；
  · 汇总 5 折留出预测得 OOF 概率；OOF 最优阈值在 OOF 概率上按主分析同款规则
    （0.10~0.70 步长 0.05 最大化 F2）选取；Recall/Specificity/PPV/F2 均为该阈值下取值。
输出：results_extra/R1_oof_model_comparison.csv（含表注行）+ R1_oof_fold_features.csv（折内 LASSO 稳定性）
本脚本不写死任何结果数字，全部由运行时计算。
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from 补充实验工具 import (qc_puc, add_features_puc, qc_pima,
                      lasso_select, train_base_models, build_voting,
                      build_stacking, predict_all, f2_best_threshold,
                      metrics_at, SEED)

os.makedirs('results_extra', exist_ok=True)
MODEL_ORDER = ['LR', 'SVM', 'RF', 'XGB', 'LGB', 'Voting', 'Stacking']
NAME_MAP = {'LR': 'LR', 'SVM': 'SVM', 'RF': 'RF', 'XGB': 'XGBoost',
            'LGB': 'LightGBM', 'Voting': 'Voting', 'Stacking': 'Stacking'}


def oof_seven_models(X_all, y_all, cohort):
    """训练集内 5 折 OOF：折内 LASSO + 折内网格搜索 + 折内集成，预测留出折。"""
    Xtr, Xte, ytr, yte = train_test_split(
        X_all, y_all, test_size=0.3, random_state=SEED, stratify=y_all)
    print(f"[{cohort}] 训练集 {len(ytr)} / 测试集 {len(yte)}（测试集不参与任何拟合与调参）")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(ytr)) for m in MODEL_ORDER}
    fold_feature_records = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, ytr), 1):
        Xtr_f, ytr_f = Xtr.iloc[tr_idx], ytr.iloc[tr_idx]
        Xva_f = Xtr.iloc[va_idx]

        # 折内 LASSO（仅用折内训练部分）
        keep, coef = lasso_select(Xtr_f, ytr_f)
        for f in Xtr.columns:
            fold_feature_records.append({'cohort': cohort, 'fold': fold,
                                         'feature': f, 'selected': f in keep})
        Xtr_raw, Xva_raw = Xtr_f[keep], Xva_f[keep]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_raw)
        Xva_s = sc.transform(Xva_raw)

        # 折内网格搜索（网格与主分析一致）+ 折内集成
        models = train_base_models(Xtr_raw, Xtr_s, ytr_f, cohort)
        voting = build_voting(models, Xtr_s, ytr_f)
        stacking = build_stacking(models, Xtr_s, ytr_f)
        probs = predict_all(models, voting, stacking, Xva_raw, Xva_s)
        for m in MODEL_ORDER:
            oof[m][va_idx] = probs[m]
        print(f"  [{cohort}] fold {fold}/5 完成（折内 LASSO 保留 {len(keep)} 特征）")

    # 汇总 OOF 指标：OOF 最优阈值（F2 规则）+ 该阈值下指标
    rows = []
    for m in MODEL_ORDER:
        prob = oof[m]
        thr, _ = f2_best_threshold(ytr.values, prob)
        met = metrics_at(ytr.values, prob, thr)
        rows.append({'cohort': cohort, 'model': NAME_MAP[m],
                     'OOF_AUC': met['AUC'], 'OOF_best_thr': met['Threshold'],
                     'OOF_Recall': met['Recall'], 'OOF_Specificity': met['Specificity'],
                     'OOF_PPV': met['PPV'], 'OOF_F2': met['F2'],
                     'TP': met['TP'], 'FP': met['FP'], 'FN': met['FN'], 'TN': met['TN']})

    # 预设规则（§2.4）在 OOF 口径下的选择结果：召回率>=0.92 中 F2 最高
    df_rows = pd.DataFrame(rows)
    cand = df_rows[df_rows['OOF_Recall'] >= 0.92]
    sel = (cand if len(cand) else df_rows).sort_values('OOF_F2', ascending=False).iloc[0]
    print(f"  [{cohort}] OOF 口径预设规则选出：{sel['model']} "
          f"(OOF F2={sel['OOF_F2']:.4f}, OOF 召回={sel['OOF_Recall']:.4f})")
    return df_rows, pd.DataFrame(fold_feature_records)


if __name__ == '__main__':
    all_rows, all_feats = [], []

    # ---- PUC 队列 ----
    df_puc = qc_puc()
    X_puc = add_features_puc(df_puc.drop('Diabetes', axis=1))
    y_puc = df_puc['Diabetes'].astype(int)
    r, f = oof_seven_models(X_puc, y_puc, 'PUC')
    all_rows.append(r); all_feats.append(f)

    # ---- Pima 队列 ----
    X_pima, y_pima = qc_pima()
    y_pima = y_pima.astype(int)
    r, f = oof_seven_models(X_pima, y_pima, 'Pima')
    all_rows.append(r); all_feats.append(f)

    out = pd.concat(all_rows, ignore_index=True)
    out = out.round({'OOF_AUC': 4, 'OOF_best_thr': 2, 'OOF_Recall': 4,
                     'OOF_Specificity': 4, 'OOF_PPV': 4, 'OOF_F2': 4})
    out.to_csv('results_extra/R1_oof_model_comparison.csv', index=False,
               encoding='utf-8-sig')
    pd.concat(all_feats, ignore_index=True).to_csv(
        'results_extra/R1_oof_fold_features.csv', index=False, encoding='utf-8-sig')

    print('\n===== R1 五折 OOF 七模型对比（OOF 预测在训练集内部生成、不含测试集）=====')
    show = out.copy()
    show['OOF_AUC'] = show['OOF_AUC'].map('{:.3f}'.format)
    show['OOF_best_thr'] = show['OOF_best_thr'].map('{:.2f}'.format)
    for c in ['OOF_Recall', 'OOF_Specificity', 'OOF_PPV']:
        show[c] = show[c].map('{:.1%}'.format)
    show['OOF_F2'] = show['OOF_F2'].map('{:.3f}'.format)
    print(show[['cohort', 'model', 'OOF_AUC', 'OOF_best_thr', 'OOF_Recall',
                'OOF_Specificity', 'OOF_PPV', 'OOF_F2']].to_string(index=False))
    print('\n已保存: results_extra/R1_oof_model_comparison.csv, R1_oof_fold_features.csv')
