# -*- coding: utf-8 -*-
"""
23_口径v2重跑对照写回.py — Pima 填补口径 v2（非零中位数、只在训练集上算）链式重跑后，
把【论文各处旧数字 → 新数字】对照表追加进《PEERJ论文实验结果汇总.docx》"二十二"节。
旧数字读 _备份_旧口径_1110/，新数字读当前 results_*，全部动态，不写死。
链条：1号(表1) → 18/19/9号(R7/R8/R1-OOF) → 7号(表S2) → 4号(3.2.4) → 6号(图)。
"""
import os
import pandas as pd
import docx
from docx.shared import Pt

DOC_PATH = r'D:\竞赛\小论文\PEERJ论文实验结果汇总.docx'
OLD = '_备份_旧口径_1110'
NEW = '.'


def rd(path):
    return pd.read_csv(path)


t1o = rd(os.path.join(OLD, 'results_pima', 'table1_model_comparison.csv')).set_index('Model')
t1n = rd(os.path.join(NEW, 'results_pima', 'table1_model_comparison.csv')).set_index('Model')
r1o = rd(os.path.join(OLD, 'R1_oof_model_comparison.csv'))
r1n = rd(os.path.join(NEW, 'results_extra', 'R1_oof_model_comparison.csv'))
r7o = rd(os.path.join(OLD, 'R7_pima_stacking_metrics.csv')).set_index('model')
r7n = rd(os.path.join(NEW, 'results_extra', 'R7_pima_stacking_metrics.csv')).set_index('model')
r8o = rd(os.path.join(OLD, 'R8_pima_stacking_gapfill.csv')).iloc[0]
r8n = rd(os.path.join(NEW, 'results_extra', 'R8_pima_stacking_gapfill.csv')).iloc[0]
risko = rd(os.path.join(OLD, 'R8_risk.csv'))
riskn = rd(os.path.join(NEW, 'results_extra', 'R8_risk.csv'))
tro = rd(os.path.join(OLD, 'results_transfer', 'transportability_results.csv')).set_index('experiment')
trn = rd(os.path.join(NEW, 'results_transfer', 'transportability_results.csv')).set_index('experiment')
mo = rd(os.path.join(OLD, 'mice_sensitivity_v2.csv')).set_index(['队列', '填补方式'])
mn = rd(os.path.join(NEW, 'results_mice', 'mice_sensitivity.csv')).set_index(['队列', '填补方式'])
r8to = rd(os.path.join(OLD, 'R8_threshold.csv'))
r8tn = rd(os.path.join(NEW, 'results_extra', 'R8_threshold.csv'))


def thr_str(df):
    f2 = df.loc[df['F2'].idxmax()]
    nb = df.loc[df['Net_Benefit'].idxmax()]
    return f"F2 最优 {f2['Threshold']:.2f}（F2 {f2['F2']:.3f}）/ 净获益最优 {nb['Threshold']:.2f}（NB {nb['Net_Benefit']:.3f}）"


def t1_str(t, m):
    r = t.loc[m]
    return (f"thr {r['Threshold']:.2f} / AUC {r['AUC']:.3f} / 召回 {r['Recall'] * 100:.1f}% / "
            f"PPV {r['PPV'] * 100:.1f}% / F2 {r['F2']:.3f} / 漏 {int(r['Missed'])}")


def r1_str(df, m):
    r = df[(df['cohort'] == 'Pima') & (df['model'] == m)].iloc[0]
    return (f"OOF F2 {r['OOF_F2']:.3f} / 阈值 {r['OOF_best_thr']:.2f} / "
            f"召回 {r['OOF_Recall'] * 100:.1f}%")


def r7_str(t, m):
    r = t.loc[m]
    return (f"{r['AUC']:.3f} ({r['AUC_CI']}) / Spec {r['Specificity'] * 100:.1f}% / "
            f"PPV {r['PPV'] * 100:.1f}% / F2 {r['F2']:.3f} / 漏诊率 {r['miss_rate'] * 100:.1f}% / "
            f"Brier {r['Brier']:.4f}")


def risk_str(df):
    return '，'.join(f"{r['风险等级']} {int(r['人数'])}人({r['占比']}, 患病率{r['实际患病率']})"
                     for _, r in df.iterrows())


def tr_str(df, key):
    r = df.loc[[i for i in df.index if i.startswith(key)][0]]
    return f"AUC {r['AUC']:.3f} / 召回 {r['Recall'] * 100:.1f}% / Brier {r['Brier']:.3f}"


def mice_delta(df):
    b = df.loc[('Pima', '中位数填补(主分析)')]
    m = df.loc[('Pima', 'MICE(20次迭代)')]
    return f"ΔAUC {abs(m['AUC'] - b['AUC']):.4f} / Δ召回 {abs(m['召回率'] - b['召回率']) * 100:.1f}pp"


# ====================== 组装文档 ======================
doc = docx.Document(DOC_PATH)


def h1(text):
    doc.add_heading(text, level=1)


def p(text, bold=False):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold
    run.font.size = Pt(10.5)
    return par


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    try:
        t.style = 'Table Grid'
    except Exception:
        pass
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = str(h)
        for par in cell.paragraphs:
            for run in par.runs:
                run.bold = True
                run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            for par in cells[i].paragraphs:
                for run in par.runs:
                    run.font.size = Pt(9)
    doc.add_paragraph()
    return t


h1('二十二、Pima 填补口径 v2 重跑对照（2026-11-10，链式重跑）')
p('口径变化：Pima 0 值视作缺失，填补由"含 0 的全量中位数"（Insulin 填 30.5 等明显失真）改为'
  '"非零中位数、且只在训练集（7:3 分层、种子 42）上估计后应用全量（防泄漏）"。'
  '重跑链：1号(表1) → 18/19号(R7/R8) + 9号(R1 OOF) → 7号(表S2) → 4号(3.2.4 迁移) → 6号(图)。'
  'PUC 侧全部不动。旧口径结果备份于 项目\\_备份_旧口径_1110\\。')
p('一句话结论：表1 七行全变（Stacking 0.844→0.842、漏 4→3；XGB 0.831→0.837、漏 3→2）；'
  'OOF 预设规则仍选中 Stacking（OOF F2 0.790→0.784，仍高于 XGB 0.780），改稿叙事成立；'
  '表S2 稳定性结论仍成立（Pima Δ召回 3.7pp→4.9pp，贴 5pp 边界）；'
  '迁移 A 方向改善明显（0.582→0.678），B/A0 完全不变。', bold=True)

# ---------------- 表1 七行 ----------------
p('（1）表1 七行 旧→新（1 号脚本，Paper Table 1）：')
rows = [[m, t1_str(t1o, m), t1_str(t1n, m)] for m in t1o.index]
table(['模型', '旧口径', 'v2 新口径'], rows)

# ---------------- R1 OOF ----------------
p('（2）表1 OOF 列 / 预设规则（9 号 R1，Pima 七行）——OOF 阈值与 F2 的归属数字：')
rows = [[m, r1_str(r1o, m), r1_str(r1n, m)]
        for m in ['Stacking', 'XGBoost', 'LR', 'RF', 'Voting', 'LightGBM', 'SVM']]
table(['模型', '旧口径', 'v2 新口径'], rows)
p('预设规则（召回≥0.92 中 F2 最高）：v2 仍选中 Stacking（0.784，召回 97.3%），XGBoost 0.780——'
  '与 19.7 改稿清单"程序性选择"叙事一致，差值更小（0.014→0.004）。', bold=True)

# ---------------- 论文位置映射 ----------------
p('（3）论文各处 旧→新 映射（"跑完影响论文哪里"逐项）：')
rows = [
    ['摘要 Pima 三个数（AUC/95%CI/召回）',
     f"{r7o.loc['Stacking', 'AUC']:.3f} ({r7o.loc['Stacking', 'AUC_CI']}) / {r7o.loc['Stacking', 'Recall'] * 100:.1f}%",
     f"{r7n.loc['Stacking', 'AUC']:.3f} ({r7n.loc['Stacking', 'AUC_CI']}) / {r7n.loc['Stacking', 'Recall'] * 100:.1f}%", '18号 R7'],
    ['表3 Pima 列（Stacking 六项）', r7_str(r7o, 'Stacking'), r7_str(r7n, 'Stacking'), '18号 R7'],
    ['§3.2.1 XGB 对照（AUC/F2/召回）',
     f"{t1o.loc['XGB', 'AUC']:.3f} / {t1o.loc['XGB', 'F2']:.3f} / {t1o.loc['XGB', 'Recall'] * 100:.1f}%",
     f"{t1n.loc['XGB', 'AUC']:.3f} / {t1n.loc['XGB', 'F2']:.3f} / {t1n.loc['XGB', 'Recall'] * 100:.1f}%", '1号 表1'],
    ['§3.3 Stacking 阈值（F2 最优/净获益最优）', thr_str(r8to), thr_str(r8tn), '19号 R8_threshold'],
    ['§3.4/表5 Pima 风险梯度', risk_str(risko), risk_str(riskn), '19号 R8_risk'],
    ['§3.5 Pima Brier（Stacking）',
     f"{r7o.loc['Stacking', 'Brier']:.4f} ({r7o.loc['Stacking', 'Brier_CI']})",
     f"{r7n.loc['Stacking', 'Brier']:.4f} ({r7n.loc['Stacking', 'Brier_CI']})", '18号 R7'],
    ['§3.7 Stacking 耗时/大小',
     f"单样本 {r8o['single_ms']:.2f} ms / 批量100 {r8o['batch100_ms']:.2f} ms / {r8o['model_mb']:.2f} MB",
     f"单样本 {r8n['single_ms']:.2f} ms / 批量100 {r8n['batch100_ms']:.2f} ms / {r8n['model_mb']:.2f} MB", '19号 R8'],
    ['§2.2.1 / 表S2 Pima 稳定性（MICE vs 主分析）', mice_delta(mo), mice_delta(mn), '7号（表S2 已写回十三节）'],
    ['§3.2.4 A 行（Pima→PUC 测试）', tr_str(tro, 'A:'), tr_str(trn, 'A:'), '4号'],
    ['§3.2.4 A-full 行', tr_str(tro, 'A-full'), tr_str(trn, 'A-full'), '4号'],
    ['§3.2.4 A0 对照', tr_str(tro, 'A0'), tr_str(trn, 'A0'), '4号（不变）'],
    ['§3.2.4 B 行（PUC→Pima 测试）', tr_str(tro, 'B:'), tr_str(trn, 'B:'), '4号（不变）'],
    ['§3.2.4 B0 对照', tr_str(tro, 'B0'), tr_str(trn, 'B0'), '4号'],
    ['§3.2.4 A 相对 A0 的 AUC 损失',
     f"{tro.loc[[i for i in tro.index if i.startswith('A0')][0], 'AUC'] - tro.loc[[i for i in tro.index if i.startswith('A:')][0], 'AUC']:.3f}",
     f"{trn.loc[[i for i in trn.index if i.startswith('A0')][0], 'AUC'] - trn.loc[[i for i in trn.index if i.startswith('A:')][0], 'AUC']:.3f}", '4号'],
]
table(['论文位置', '旧数字', 'v2 新数字', '来源'], rows)

# ---------------- 说明与次级影响 ----------------
p('（4）不变项与机制说明：', bold=True)
p('· 4 号的 Pima 填补原本就是"0→NaN→训练集观测中位数"（目标口径），其数字变化全部来自 '
  'XGB_PIMA 超参跟随新表1（200/3/0.1 → 100/3/0.05）；B（PUC→Pima）与 A0 的输入全链未动，数字逐位不变，'
  '并非重跑遗漏。PUC 侧（表2、图1-5、R1 PUC 七行、MICE PUC 臂）全部不变。')
p('· LASSO 保留特征 8→6；XGB 网格赢家 200/3/0.1→100/3/0.05（results_pima/grid_best_params.csv，'
  '18 号阈值已改为动态读表1）。Stacking 模型体积 2.72→1.48 MB（基模型 XGB 变小）。'
  'figures_pima/（Fig S1–S5）已随 1 号重生成；注意 SHAP 图（S4/S5）仍按 1 号的"F2 最优=XGB"绘制，'
  '若论文按 Stacking 叙事需另换模型重画（19.7 遗留事项）。')
p('（5）次级受影响、本次未重跑（如需同步请安排）：', bold=True)
p('· 14 号 R4b 配对差值（19.5 节）：其文字引用 "OOF F2 0.790 vs 0.776、CI [−0.001, +0.030]"，'
  '新基准为 0.784 vs 0.780，需重跑 14 号刷新 CI 与解读；')
p('· 本汇总文档十九节（19.1 R1 表与"五折 5/8/8/5/7"文字、19.5、19.7 改稿清单各处旧数字）'
  '与 table 5（表1 旧副本）、table 25（R1 旧副本）仍是旧口径数字，属历史记录，投稿前需按本节对照统一；')
p('· 8 号（推理耗时复测）仍是旧口径填补重训的载体（只影响耗时数字，可选重跑）；'
  'Diabetes1110.docx 正文（摘要/表1/表3/3.2.1/3.3/3.4/3.5/3.7/4.1/5）按本节映射逐处替换。')

doc.save(DOC_PATH)
print('已写入:', DOC_PATH)
print('新增章节：二十二（口径 v2 重跑对照）')
for tb in doc.tables[-3:]:
    print('---')
    for r in tb.rows:
        print(' | '.join(c.text for c in r.cells)[:150])
