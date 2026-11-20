# -*- coding: utf-8 -*-
"""
25_OOF选模_全规则.py — 5 折 OOF 选模（完整预设规则），全程不读写测试集
对应论文 §2.4.1（Out-of-Fold Model Selection）的全规则实现：
  · 先按主分析同款 7:3 分层划分（种子 42）定位训练行 —— 测试行只用于定位划分，
    随即丢弃，不进入任何拟合、预测、选模、阈值步骤（全程不读写测试集）；
  · 训练集内 5 折分层 CV（shuffle=True, 种子 42）；每个折内【完整重做】
    LASSO 特征筛选 + 5 模型网格搜索（网格与 1_Pima建模.py / 2_中国数据建模.py
    完全一致，cohort 区分 LGB 网格）+ Voting（5 成员，与主分析一致）/Stacking
    构建，再预测留出折；
  · 汇总 5 折留出预测得 OOF 概率；每模型的运行阈值取 OOF 汇总值
    （0.10~0.70 步长 0.05 最大化 F2，prob > thr，与主分析同规则）；
  · 选模按预设全规则（§2.4.1）：recall>=0.92 → 最高 F2 → 最少漏诊 → 最高特异度。

与 9_R1_OOF七模型对比.py 的差异（详见 核对报告_1120.md）：
  (1) 9 号把 'Pima'/'PUC' 传入 train_base_models（内部判断 'pima'/'puc'），
      导致 Pima 的 LGB 误用 PUC 网格（多出 learning_rate 维度），Pima 的
      LGB/Voting OOF 数值失真；本脚本传入小写键，修正该 bug；
  (2) 9 号选模只实现"recall>=0.92 → 最高 F2"两级，缺"最少漏诊 → 最高特异度"
      两级平局决胜；本脚本实现全部四级；
  (3) 9 号无选模过程追溯；本脚本输出每级过滤名单与排序依据。

输出（全部由运行时计算，不写死任何结果数字）：
  results_extra/OOF_select_1120/oof_metrics_{cohort}.csv       7 模型 OOF 指标+混淆矩阵
  results_extra/OOF_select_1120/oof_selection_{cohort}.csv     选中模型+四级规则追溯
  results_extra/OOF_select_1120/oof_fold_features_{cohort}.csv 折内 LASSO 特征稳定性
  logs/25_OOF选模_run_1120.txt                                  运行日志（可回溯）

运行方式：在本文件夹（工作目录 = 本文件夹）执行
    python 25_OOF选模_全规则.py
"""
import io
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from 补充实验工具 import (qc_puc, add_features_puc, qc_pima,
                      lasso_select, train_base_models, build_voting,
                      build_stacking, predict_all, f2_best_threshold,
                      metrics_at, SEED, THR_GRID, USE_SCALED)

OUT_DIR = os.path.join('results_extra', 'OOF_select_1120')
LOG_DIR = 'logs'
MODEL_ORDER = ['LR', 'SVM', 'RF', 'XGB', 'LGB', 'Voting', 'Stacking']
NAME_MAP = {'LR': 'LR', 'SVM': 'SVM', 'RF': 'RF', 'XGB': 'XGBoost',
            'LGB': 'LightGBM', 'Voting': 'Voting', 'Stacking': 'Stacking'}
RECALL_GATE = 0.92   # §2.4.1 第 (1) 级，预设固定，不得改动


class Tee(object):
    """同时写控制台与日志文件"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, '25_OOF选模_run_1120.txt')
    f = io.open(path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, f)
    return path


def log_env():
    import sklearn, xgboost, lightgbm
    print('时间: %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    print('脚本: 25_OOF选模_全规则.py（OOF 全规则选模，不读写测试集）')
    print('Python %s | sklearn %s | numpy %s | pandas %s | xgboost %s | lightgbm %s'
          % (sys.version.split()[0], sklearn.__version__, np.__version__,
             pd.__version__, xgboost.__version__, lightgbm.__version__))
    print('SEED = %d | 阈值网格 %s | 选模规则: recall>=%.2f → 最高F2 → 最少漏诊 → 最高特异度'
          % (SEED, np.array2string(THR_GRID, precision=2), RECALL_GATE))
    print('Voting 口径: 5 成员（LR/RF/XGB/LGB/SVM，与主分析 1/2 号脚本一致）')
    print('-' * 70)


def select_by_rule(df):
    """§2.4.1 全规则：recall>=0.92 → 最高 F2 → 最少漏诊 → 最高特异度。
    df: 含 model, OOF_Recall, OOF_F2, FN, OOF_Specificity 列（每模型一行）。
    返回 (选中行 dict, 追溯文本列表)。规则本身固定，任何情况下不得改动。"""
    trace = []
    cand = df[df['OOF_Recall'] >= RECALL_GATE].copy()
    if len(cand):
        trace.append('第1级 recall>=%.2f：保留 %d 个模型 [%s]'
                     % (RECALL_GATE, len(cand), ', '.join(cand['model'])))
    else:
        trace.append('第1级 recall>=%.2f：无模型达标——按规则此时无选中模型；'
                     '此处退化为全池并已显式警告（非静默处理）' % RECALL_GATE)
        cand = df.copy()
        trace.append('  [警告] 无模型达到召回率门槛，后续在全池中进行')

    cand = cand.sort_values(['OOF_F2', 'FN', 'OOF_Specificity'],
                            ascending=[False, True, False]).reset_index(drop=True)
    top_f2 = cand.loc[0, 'OOF_F2']
    s2 = cand[cand['OOF_F2'] == top_f2]
    if len(s2) == 1:
        gap = top_f2 - (cand.loc[1, 'OOF_F2'] if len(cand) > 1 else float('nan'))
        trace.append('第2级 最高F2：胜出 %s（OOF F2=%.4f，与次名差 %.4f）'
                     % (s2.loc[0, 'model'], top_f2, gap))
    else:
        trace.append('第2级 最高F2：并列 %s（OOF F2=%.4f），进入第3级'
                     % (', '.join(s2['model']), top_f2))

    s3 = s2[s2['FN'] == s2['FN'].min()]
    if len(s2) > 1:
        if len(s3) == 1:
            trace.append('第3级 最少漏诊：胜出 %s（FN=%d）'
                         % (s3.loc[0, 'model'], int(s3.loc[0, 'FN'])))
        else:
            trace.append('第3级 最少漏诊：并列 %s（FN=%d），进入第4级'
                         % (', '.join(s3['model']), int(s3.loc[0, 'FN'])))

    s4 = s3[s3['OOF_Specificity'] == s3['OOF_Specificity'].max()]
    if len(s3) > 1:
        if len(s4) == 1:
            trace.append('第4级 最高特异度：胜出 %s（Spec=%.4f）'
                         % (s4.loc[0, 'model'], s4.loc[0, 'OOF_Specificity']))
        else:
            trace.append('第4级 最高特异度：仍并列 %s，取排序首位 %s'
                         % (', '.join(s4['model']), s4.loc[0, 'model']))

    sel = s4.sort_values(['OOF_F2', 'FN', 'OOF_Specificity'],
                         ascending=[False, True, False]).iloc[0]
    trace.append('最终选中：%s（OOF F2=%.4f, OOF Recall=%.4f, OOF 阈值=%.2f）'
                 % (sel['model'], sel['OOF_F2'], sel['OOF_Recall'],
                    sel['OOF_best_thr']))
    return sel, trace


def oof_full_rule(X_all, y_all, cohort):
    """训练集内 5 折 OOF：折内 LASSO + 折内网格搜索 + 折内集成，预测留出折。
    测试行在划分后立即丢弃，全程不读写。"""
    n = len(y_all)
    idx = np.arange(n)
    idx_tr, _idx_te = train_test_split(idx, test_size=0.3, random_state=SEED,
                                       stratify=y_all)
    # 测试行（_idx_te）在此之后不再被引用：不拟合、不预测、不选模、不评估
    Xtr_all = X_all.iloc[idx_tr]
    ytr_all = y_all.iloc[idx_tr].reset_index(drop=True)
    print('[%s] 训练集 %d 例（测试集 %d 例已丢弃，不进入任何后续步骤）'
          % (cohort, len(ytr_all), len(_idx_te)))
    assert len(ytr_all) == (537 if cohort == 'Pima' else 2842), '训练集行数与主分析不符'

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(ytr_all)) for m in MODEL_ORDER}
    fold_records = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr_all, ytr_all), 1):
        Xtr_f, ytr_f = Xtr_all.iloc[tr_idx], ytr_all.iloc[tr_idx]
        Xva_f = Xtr_all.iloc[va_idx]

        # 折内 LASSO（仅用折内训练部分估计）
        keep, coef = lasso_select(Xtr_f, ytr_f)
        for feat in Xtr_all.columns:
            fold_records.append({'cohort': cohort, 'fold': fold,
                                 'feature': feat, 'selected': feat in keep})
        Xtr_raw, Xva_raw = Xtr_f[keep], Xva_f[keep]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_raw)
        Xva_s = sc.transform(Xva_raw)

        # 折内网格搜索（网格与主分析一致）+ 折内集成构建
        # 修正 9 号 bug：cohort 必须传 'pima'/'puc'（小写），否则 LGB 网格错配
        models = train_base_models(Xtr_raw, Xtr_s, ytr_f, cohort.lower())
        voting = build_voting(models, Xtr_s, ytr_f)
        stacking = build_stacking(models, Xtr_s, ytr_f)
        probs = predict_all(models, voting, stacking, Xva_raw, Xva_s)
        for m in MODEL_ORDER:
            oof[m][va_idx] = probs[m]
        print('  [%s] fold %d/5 完成（折内 LASSO 保留 %d/%d 特征）'
              % (cohort, fold, len(keep), Xtr_all.shape[1]))

    # 汇总 OOF：每模型阈值取 OOF 汇总值（F2 规则），并给完整混淆矩阵
    rows = []
    for m in MODEL_ORDER:
        prob = oof[m]
        thr, _ = f2_best_threshold(ytr_all.values, prob)
        met = metrics_at(ytr_all.values, prob, thr)
        rows.append({'cohort': cohort, 'model': NAME_MAP[m],
                     'OOF_AUC': met['AUC'], 'OOF_best_thr': met['Threshold'],
                     'OOF_Recall': met['Recall'], 'OOF_Specificity': met['Specificity'],
                     'OOF_PPV': met['PPV'], 'OOF_F2': met['F2'],
                     'TP': met['TP'], 'FP': met['FP'],
                     'FN': met['FN'], 'TN': met['TN']})
    df_rows = pd.DataFrame(rows)

    sel, trace = select_by_rule(df_rows)
    for line in trace:
        print('  ' + line)

    out_rows = df_rows.round({'OOF_AUC': 4, 'OOF_best_thr': 2, 'OOF_Recall': 4,
                              'OOF_Specificity': 4, 'OOF_PPV': 4, 'OOF_F2': 4})
    out_rows.to_csv(os.path.join(OUT_DIR, 'oof_metrics_%s.csv' % cohort),
                    index=False, encoding='utf-8-sig')
    pd.DataFrame([{'cohort': cohort, 'step': i + 1, 'detail': t}
                  for i, t in enumerate(trace)]
                 + [{'cohort': cohort, 'step': 0,
                     'detail': 'selected_model=%s; oof_threshold=%.2f; '
                               'rule=recall>=%.2f→maxF2→minFN→maxSpec'
                               % (sel['model'], sel['OOF_best_thr'], RECALL_GATE)}]
                 ).to_csv(os.path.join(OUT_DIR, 'oof_selection_%s.csv' % cohort),
                          index=False, encoding='utf-8-sig')
    pd.DataFrame(fold_records).to_csv(
        os.path.join(OUT_DIR, 'oof_fold_features_%s.csv' % cohort),
        index=False, encoding='utf-8-sig')
    return df_rows, sel, trace


if __name__ == '__main__':
    log_path = setup_log()
    os.makedirs(OUT_DIR, exist_ok=True)
    log_env()

    # ---- PUC 队列（质控与 2_中国数据建模.py 完全一致；划分后丢弃测试行）----
    print('===== PUC 队列 =====')
    df_puc = qc_puc()
    X_puc = add_features_puc(df_puc.drop('Diabetes', axis=1))
    y_puc = df_puc['Diabetes'].astype(int)
    oof_full_rule(X_puc, y_puc, 'PUC')

    # ---- Pima 队列（清洗与 1_Pima建模.py 完全一致）----
    print('===== Pima 队列 =====')
    X_pima, y_pima = qc_pima()
    y_pima = y_pima.astype(int)
    oof_full_rule(X_pima, y_pima, 'Pima')

    print('-' * 70)
    print('完成。输出目录: %s' % OUT_DIR)
    print('运行日志: %s' % log_path)
    print('本脚本全程未读写测试集：划分产生的测试行仅用于确定训练行索引，随即丢弃。')
