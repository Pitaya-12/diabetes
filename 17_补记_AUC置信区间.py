# -*- coding: utf-8 -*-
"""17_补记_AUC置信区间.py — 把 16 号脚本的 Pima AUC 95%CI 写入汇总 docx（19.6 节）"""
import pandas as pd
import docx
from docx.shared import Pt

DOC = r'C:\Users\孟雅玲\Desktop\PEERJ论文实验结果汇总.docx'
r5 = pd.read_csv('results_extra/R5_pima_auc_ci.csv')
doc = docx.Document(DOC)

doc.add_heading('19.6 补记：Pima 测试集 AUC 95%CI（表3 备用，2026-11-09）', level=2)
p = doc.add_paragraph()
r = p.add_run('若按 (a) 将 Pima 最优模型改为 Stacking，表3/§3.2.3 的 AUC 95%CI 需换成 Stacking 的。'
              '按主分析同口径（7:3 种子42 → LASSO → 网格搜索 → 集成，2,000 次 Bootstrap）实测：')
r.font.size = Pt(10.5)

t = doc.add_table(rows=1, cols=5)
try:
    t.style = 'Table Grid'
except Exception:
    pass
for i, v in enumerate(['模型', 'AUC', '95%CI', '测试 n（阳性）', '备注']):
    t.rows[0].cells[i].text = v
    for par in t.rows[0].cells[i].paragraphs:
        for run in par.runs:
            run.bold = True
            run.font.size = Pt(9)
notes = {'XGB': '与正文 0.774–0.882 完全一致，口径验证通过',
         'Stacking': '改稿 (a) 时表3 Pima 列用此值'}
for _, row in r5.iterrows():
    cells = t.add_row().cells
    vals = [row['model'], f"{row['AUC']:.3f}",
            f"{row['CI95_low']:.3f}-{row['CI95_high']:.3f}",
            f"{int(row['n_test'])} ({int(row['n_pos'])})",
            notes.get(row['model'], '')]
    for i, v in enumerate(vals):
        cells[i].text = str(v)
        for par in cells[i].paragraphs:
            for run in par.runs:
                run.font.size = Pt(9)
doc.add_paragraph()
doc.save(DOC)
print('19.6 ok:', DOC)
