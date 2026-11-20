# -*- coding: utf-8 -*-
"""
20_补记_改稿清单.py — 19.7 节：Pima 最优模型改 Stacking 的逐条改稿清单（含全部新数字）
数字来源：results_extra/R5/R7/R8 CSV + 19.1 OOF 表 + 19.5 配对差值（均已实测）。
"""
import pandas as pd
import docx
from docx.shared import Pt

DOC = r'C:\Users\孟雅玲\Desktop\PEERJ论文实验结果汇总.docx'
r7 = pd.read_csv('results_extra/R7_pima_stacking_metrics.csv')
r8 = pd.read_csv('results_extra/R8_threshold.csv')
r8r = pd.read_csv('results_extra/R8_risk.csv')
r8g = pd.read_csv('results_extra/R8_pima_stacking_gapfill.csv')

stk = r7[r7['model'] == 'Stacking'].iloc[0]
xgb = r7[r7['model'] == 'XGB'].iloc[0]
gap = r8g.iloc[0]

doc = docx.Document(DOC)
h = doc.add_heading('19.7 改稿清单（a）：Pima 最优模型 XGBoost → Stacking（2026-11-09，数字已备齐）', level=2)


def p(text, bold=False):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold
    run.font.size = Pt(10.5)


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    try:
        t.style = 'Table Grid'
    except Exception:
        pass
    for i, v in enumerate(headers):
        t.rows[0].cells[i].text = str(v)
        for par in t.rows[0].cells[i].paragraphs:
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


p('前提结论：PUC 队列不变（OOF 与测试集都选 Voting）；只有 Pima 归属变。'
  '选择依据必须写明是 OOF 口径（19.1），因为测试集上 XGBoost 的 F2（0.825）其实略高于 Stacking（0.817）——'
  '按预设规则在 OOF 上选中 Stacking（OOF F2 0.790 vs 0.776），测试集数字照实报告即可，'
  '两者 OOF F2 差异 CI 包含 0（19.5），可写"程序性选择、统计不可区分"。', bold=True)

p('一、新旧数字对照（全部实测）：', bold=True)
table(['项目', '旧（XGBoost）', '新（Stacking）', '出处'],
      [['测试集 AUC (95%CI)', '0.831 (0.774–0.882)', f"{stk['AUC']:.3f} ({stk['AUC_CI']})", '19.6 / R7'],
       ['召回率', '96.3%', f"{stk['Recall']:.1%}", 'R7'],
       ['特异度', '52.7%', f"{stk['Specificity']:.1%}", 'R7'],
       ['PPV', '52.3%', f"{stk['PPV']:.1%}", 'R7'],
       ['F2', '0.825', f"{stk['F2']:.3f}", 'R7'],
       ['最优阈值', '0.20', '0.25', '表1（已含）'],
       ['漏诊数/率', '3 / 3.7%', f"{int(stk['FN'])} / {stk['miss_rate']:.1%}", 'R7'],
       ['混淆矩阵', 'TP78/FP71/FN3/TN79',
        f"TP{int(stk['TP'])}/FP{int(stk['FP'])}/FN{int(stk['FN'])}/TN{int(stk['TN'])}", 'R7'],
       ['Brier (95%CI)', '0.1636 (0.1354–0.1937)',
        f"{stk['Brier']:.4f} ({stk['Brier_CI']})", 'R7'],
       ['OOF 选择依据', '—（旧为测试集 F2 0.825）', 'OOF F2 0.790，召回 97.9%', '19.1'],
       ['AUC 差异对比 PUC', '+0.134', '+0.121（0.965−0.844）', '推算'],
       ['推理耗时（修正后）', '1.372±0.007 ms / 批量1.492±0.049 ms / 0.11MB',
        f"{gap['single_ms']:.2f} ms / 批量{gap['batch100_ms']:.2f} ms / {gap['model_mb']:.2f}MB", 'R8'],
       ['四级风险梯度', '7.9%→74.5%', '4.8%→70.6%', 'R8_risk'],
       ['F2 最优阈值 / NB 最优', '0.20 / 0.10(NB 0.299)', f"{gap['F2_thr']:.2f} / {gap['NB_best_thr']:.2f}(NB 0.281)", 'R8_threshold']])

p('二、逐条替换（对齐 18.2 编号习惯）：', bold=True)
table(['位置', '旧', '新'],
      [['摘要 Results', '"In the Pima cohort, XGBoost achieved an AUC of 0.831"',
        '"In the Pima cohort, the Stacking ensemble achieved an AUC of 0.844 (95% CI 0.791–0.892)"'],
       ['摘要 Conclusions', '"XGBoost was optimal in the Pima cohort"',
        '"the Stacking ensemble was optimal in the Pima cohort (selected under the pre-specified rule on OOF predictions)"'],
       ['§2.4 规则执行句',
        '"and XGBoost for the Pima cohort (F₂ = 0.825 at recall = 0.963), with logistic regression a close runner-up (F₂ = 0.819)"',
        '"and the Stacking ensemble for the Pima cohort (OOF F₂ = 0.790 at OOF recall = 0.979); the OOF F₂ difference between Stacking and XGBoost was not statistically distinguishable (95% CI of paired difference −0.001 to 0.030), so the designation is procedural rather than evidence of superiority."'],
       ['§2.4 选择口径', '（无）',
        '补一句：model selection was executed on 5-fold OOF predictions generated within the training set; test-set metrics are reported for the selected model only.'],
       ['§3.2.1 第一段',
        '"XGBoost was the optimal model for this cohort: at the optimal threshold of 0.20, AUC = 0.831, recall = 96.3%, specificity = 52.7%, and 3 missed diagnoses"',
        '"the Stacking ensemble was selected as optimal under the pre-specified rule (Section 2.4); on the held-out test set at the F₂-optimal threshold of 0.25 it yielded AUC = 0.844 (95% CI 0.791–0.892), recall = 95.1%, specificity = 53.3%, PPV = 52.4%, F₂ = 0.817, and 4 missed diagnoses. XGBoost remained a close comparator (AUC = 0.831, F₂ = 0.825 on the test set), and the paired difference in OOF F₂ included zero."'],
       ['§3.2.3',
        '"AUC estimates of 0.831 (95% CI: 0.774–0.882) for the Pima XGBoost model" ... "The observed AUC difference was 0.134"',
        '"0.844 (95% CI: 0.791–0.892) for the Pima Stacking ensemble" ... "The observed AUC difference was 0.121"'],
       ['表3（Pima 列）', 'XGBoost：0.831 / 0.774–0.882 / 52.7% / 3.7% / 0.825',
        'Stacking：0.844 / 0.791–0.892 / 53.3% / 4.9% / 0.817（列头改 Pima (Stacking)）'],
       ['§3.2.4 括号注',
        '"the full-feature Pima XGBoost reported in Section 3.5, whose Brier score was 0.1636"',
        '"the full-feature Pima Stacking ensemble reported in Section 3.5, whose Brier score was 0.1639"（迁移载体仍是 4 变量 XGBoost，可另注明该选择是先验指定、与最优模型无关）'],
       ['§3.3', '"the Pima XGBoost model had a stable interval of 0.15–0.25"',
        '"the Pima Stacking ensemble had a stable interval of 0.15–0.30"（F2 最优 0.25；净获益最优 0.10，NB=0.281；逐点见 R8_threshold）'],
       ['§3.4/表5 Pima', '梯度 7.9%→74.5%（XGBoost 概率）',
        f"梯度 4.8%→70.6%（Stacking 概率：低 {r8r.iloc[0]['人数']}人/{r8r.iloc[0]['实际患病率']}，中 {r8r.iloc[1]['人数']}/{r8r.iloc[1]['实际患病率']}，高 {r8r.iloc[2]['人数']}/{r8r.iloc[2]['实际患病率']}，极高 {r8r.iloc[3]['人数']}/{r8r.iloc[3]['实际患病率']}）"],
       ['§3.5 Brier', '"(XGBoost: 0.1636, 95% CI: 0.1354–0.1937)"',
        f"(Stacking: {stk['Brier']:.4f}, 95% CI: {stk['Brier_CI']})"],
       ['§3.7', '"the Pima cohort XGBoost model had a file size of 0.11 MB, single-sample inference of 1.372 ± 0.007 ms..."',
        f'两行都报：单模型基线 XGBoost 0.11MB/1.37ms（保留）；最终 Pima 模型 Stacking {gap["model_mb"]:.2f}MB/单样本 {gap["single_ms"]:.2f}±0.04 ms/批量100 {gap["batch100_ms"]:.2f} ms。'
        f'注：Stacking 单样本耗时约为 XGBoost 的 4.5 倍，但仍为个位数 ms，部署结论不变'],
       ['§4.1', '"the Pima-cohort XGBoost model (AUC = 0.831)"; "AUC difference of 0.134 (Pima 0.831, 95% CI 0.774–0.882..."',
        '"the Pima-cohort Stacking ensemble (AUC = 0.844)"; "AUC difference of 0.121 (Pima 0.844, 95% CI 0.791–0.892..."'],
       ['§5 Conclusions', '"XGBoost was optimal in the Pima cohort (AUC = 0.831)"',
        '"the Stacking ensemble was optimal in the Pima cohort (AUC = 0.844)"'],
       ['表1', '（7 行数字不动）', '加表注：模型选择按 §2.4 规则在训练集 OOF 上执行，选中 Stacking；'
        '本表按测试集 F2 排序仅作呈现，XGBoost 行 F2 略高不改变选择结果']])

p('三、不动的部分：表1 七行数字、表2/表4、图1–5（均为 PUC）、消融、迁移主数字、可信度、MICE、成本、'
  '§4.2/4.3/4.5 主体、关键词（XGBoost 仍是被比较的 7 模型之一）。'
  '需重制的图：仅 Pima 补充图 S1–S5 中"最优模型"主题的（SHAP 等）如以 XGBoost 画需换 Stacking 重跑 1 号脚本相应段，可另安排。')
p('四、计量勘误（如实记录）：18 号脚本首轮把单样本 1000 次总耗时误作单次报出（6242 ms），'
  '19 号脚本已修正为 6.22 ms/次；本节引用均为修正后数值。', bold=True)

doc.save(DOC)
print('19.7 ok')
