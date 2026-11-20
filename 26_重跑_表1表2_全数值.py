# -*- coding: utf-8 -*-
"""
26_重跑_表1表2_全数值.py — 固定种子 42、同一 7:3 分层划分重跑 Pima 与 PUC，
输出论文表1/表2 全部数值 + 每模型混淆矩阵 + AUC 95%CI（2000 次 bootstrap）。
口径约束（与 1_Pima建模.py / 2_中国数据建模.py 完全一致，不改）：
  · 超参搜索空间不变（LGB 网格按队列区分：Pima 无 learning_rate，PUC 有）；
  · 评价函数不变（F2，阈值网格 0.10~0.70 步长 0.05，prob > thr）；
  · 数据划分不变（7:3 分层，种子 42）；Voting 5 成员不变。
选模：沿用 25_OOF选模_全规则.py 的 OOF 全规则结果（阈值取 OOF 汇总值），
      测试集只做一次性报告性评估，不参与任何选模与调参。
输出（全部由运行时计算）：
  results_rerun_1120/table1_pima_full.csv     表1 全数值（OOF 列+测试列+混淆矩阵+AUC CI）
  results_rerun_1120/table2_puc_full.csv      表2 全数值（同上）
  results_rerun_1120/metrics_at_oof_thr_{cohort}.csv  OOF 阈值下的测试指标（协议口径）
  results_rerun_1120/selection_summary.csv    OOF 全规则选模结果（来自 25 号输出）
  logs/26_重跑_run_1120.txt                    运行日志（版本/种子/时间戳，可回溯）
运行：python 26_重跑_表1表2_全数值.py   （需先运行 25 号脚本）
"""
import io
import os
import sys
import time

import numpy as np
import pandas as pd

from 工具函数 import bootstrap_auc_ci
from 补充实验工具 import (qc_puc, add_features_puc, qc_pima, run_pipeline,
                      f2_best_threshold, metrics_at, SEED, THR_GRID)

OUT_DIR = 'results_rerun_1120'
OOF_DIR = os.path.join('results_extra', 'OOF_select_1120')
LOG_DIR = 'logs'
MODEL_ORDER = ['LR', 'SVM', 'RF', 'XGB', 'LGB', 'Voting', 'Stacking']
NAME_MAP = {'LR': 'LR', 'SVM': 'SVM', 'RF': 'RF', 'XGB': 'XGBoost',
            'LGB': 'LightGBM', 'Voting': 'Voting', 'Stacking': 'Stacking'}
N_BOOT = 2000   # 论文 §2.5.1：2000 次 bootstrap，不得改动


class Tee(object):
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
    path = os.path.join(LOG_DIR, '26_重跑_run_1120.txt')
    f = io.open(path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, f)
    return path


def eval_cohort(cohort, X_all, y_all, oof_df, sel_row):
    """完整主分析管线重跑 + 测试集一次性报告性评估。"""
    print('=' * 70)
    print('[%s] 重跑主分析管线（种子 %d，7:3 分层）' % (cohort, SEED))
    res = run_pipeline(X_all, y_all, split_seed=SEED,
                       cohort='pima' if cohort == 'Pima' else 'puc',
                       want_stacking=True)
    yte = res['y_test'].values
    print('[%s] 训练 %d / 测试 %d（阳性 %d）'
          % (cohort, res['n_train'], res['n_test'], int(yte.sum())))

    rows = []
    for m in MODEL_ORDER:
        prob = res['probs'][m]
        # 测试集 F2 最优阈值（表1/表2 的"乐观描述性"口径，论文 §2.5.1 注）
        thr, _ = f2_best_threshold(yte, prob)
        met = metrics_at(yte, prob, thr)
        auc_pt, lo, hi = bootstrap_auc_ci(yte, prob, n_boot=N_BOOT, seed=SEED)
        oof_row = oof_df[oof_df['model'] == NAME_MAP[m]].iloc[0]
        rows.append({'Model': NAME_MAP[m],
                     'OOF_F2': oof_row['OOF_F2'],
                     'OOF_threshold': oof_row['OOF_best_thr'],
                     'OOF_Recall': oof_row['OOF_Recall'],
                     'OOF_Specificity': oof_row['OOF_Specificity'],
                     'Test_threshold': met['Threshold'],
                     'AUC': auc_pt, 'AUC_CI_low': lo, 'AUC_CI_high': hi,
                     'Recall': met['Recall'], 'Specificity': met['Specificity'],
                     'PPV': met['PPV'], 'Test_F2': met['F2'], 'Missed': met['FN'],
                     'TP': met['TP'], 'FP': met['FP'],
                     'FN': met['FN'], 'TN': met['TN']})
        print('  %-9s AUC=%.4f (%.4f-%.4f) thr=%.2f R=%.3f P=%.3f F2=%.3f 漏诊=%d CM=%d/%d/%d/%d'
              % (NAME_MAP[m], auc_pt, lo, hi, met['Threshold'], met['Recall'],
                 met['PPV'], met['F2'], met['FN'],
                 met['TP'], met['FP'], met['FN'], met['TN']))

    df = pd.DataFrame(rows)

    # 协议口径补充表：在 OOF 汇总阈值下的测试集指标（§2.4.1 "阈值取 OOF 汇总值"）
    oof_thr_rows = []
    for m in MODEL_ORDER:
        prob = res['probs'][m]
        oof_row = oof_df[oof_df['model'] == NAME_MAP[m]].iloc[0]
        met = metrics_at(yte, prob, oof_row['OOF_best_thr'])
        oof_thr_rows.append({'Model': NAME_MAP[m],
                             'OOF_threshold': oof_row['OOF_best_thr'],
                             'AUC': met['AUC'], 'Recall': met['Recall'],
                             'Specificity': met['Specificity'], 'PPV': met['PPV'],
                             'F2': met['F2'], 'Missed': met['FN'],
                             'TP': met['TP'], 'FP': met['FP'],
                             'FN': met['FN'], 'TN': met['TN']})
    pd.DataFrame(oof_thr_rows).round(4).to_csv(
        os.path.join(OUT_DIR, 'metrics_at_oof_thr_%s.csv' % cohort),
        index=False, encoding='utf-8-sig')

    # 标记 OOF 全规则选中的模型
    df['OOF_selected'] = df['Model'] == sel_row['model']
    print('[%s] OOF 全规则选中：%s（OOF 阈值 %.2f）'
          % (cohort, sel_row['model'], sel_row['OOF_best_thr']))
    return df


if __name__ == '__main__':
    log_path = setup_log()
    os.makedirs(OUT_DIR, exist_ok=True)

    import sklearn, xgboost, lightgbm
    print('时间: %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    print('脚本: 26_重跑_表1表2_全数值.py')
    print('Python %s | sklearn %s | numpy %s | pandas %s | xgboost %s | lightgbm %s'
          % (sys.version.split()[0], sklearn.__version__, np.__version__,
             pd.__version__, xgboost.__version__, lightgbm.__version__))
    print('SEED = %d | 划分 7:3 分层 | bootstrap n=%d seed=%d | 阈值网格 %s'
          % (SEED, N_BOOT, SEED, np.array2string(THR_GRID, precision=2)))
    print('口径：超参搜索空间/评价函数/数据划分均与主分析一致；Voting 5 成员')
    print('-' * 70)

    # 读取 25 号 OOF 全规则输出（缺失则报错，不静默降级）
    sel_rows = []
    for cohort in ['PUC', 'Pima']:
        p = os.path.join(OOF_DIR, 'oof_metrics_%s.csv' % cohort)
        if not os.path.exists(p):
            raise SystemExit('缺少 %s，请先运行 25_OOF选模_全规则.py' % p)
    oof_puc = pd.read_csv(os.path.join(OOF_DIR, 'oof_metrics_PUC.csv'))
    oof_pima = pd.read_csv(os.path.join(OOF_DIR, 'oof_metrics_Pima.csv'))
    sel_puc = pd.read_csv(os.path.join(OOF_DIR, 'oof_selection_PUC.csv'))
    sel_pima = pd.read_csv(os.path.join(OOF_DIR, 'oof_selection_Pima.csv'))

    def sel_of(sel_df):
        line = sel_df[sel_df['step'] == 0]['detail'].iloc[0]
        name = line.split('selected_model=')[1].split(';')[0]
        thr = float(line.split('oof_threshold=')[1].split(';')[0])
        return {'model': name, 'OOF_best_thr': thr}

    s_puc, s_pima = sel_of(sel_puc), sel_of(sel_pima)
    sel_rows = [{'cohort': 'PUC', **s_puc}, {'cohort': 'Pima', **s_pima}]
    pd.DataFrame(sel_rows).to_csv(os.path.join(OUT_DIR, 'selection_summary.csv'),
                                  index=False, encoding='utf-8-sig')

    # ---- Pima（表1）----
    X_pima, y_pima = qc_pima()
    y_pima = y_pima.astype(int)
    t1 = eval_cohort('Pima', X_pima, y_pima, oof_pima, s_pima)
    t1 = t1.sort_values('OOF_F2', ascending=False)   # 论文表1 注：按 OOF F2 降序
    t1.round(4).to_csv(os.path.join(OUT_DIR, 'table1_pima_full.csv'),
                       index=False, encoding='utf-8-sig')

    # ---- PUC（表2）----
    df_puc = qc_puc()
    X_puc = add_features_puc(df_puc.drop('Diabetes', axis=1))
    y_puc = df_puc['Diabetes'].astype(int)
    t2 = eval_cohort('PUC', X_puc, y_puc, oof_puc, s_puc)
    t2 = t2.sort_values('Test_F2', ascending=False)   # 表2 按测试 F2 降序
    t2.round(4).to_csv(os.path.join(OUT_DIR, 'table2_puc_full.csv'),
                       index=False, encoding='utf-8-sig')

    print('-' * 70)
    print('表1（Pima，按 OOF F2 降序）:')
    print(t1[['Model', 'OOF_F2', 'OOF_threshold', 'Test_threshold', 'AUC',
              'AUC_CI_low', 'AUC_CI_high', 'Recall', 'PPV', 'Test_F2',
              'Missed', 'OOF_selected']].to_string(index=False))
    print('表2（PUC，按测试 F2 降序）:')
    print(t2[['Model', 'AUC', 'AUC_CI_low', 'AUC_CI_high', 'Recall', 'PPV',
              'Test_F2', 'Test_threshold', 'Missed', 'OOF_selected']].to_string(index=False))
    print('完成。输出目录: %s | 运行日志: %s' % (OUT_DIR, log_path))
