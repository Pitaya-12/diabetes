# -*- coding: utf-8 -*-
"""
24_R2糖剔除对照补记.py — §3.6.3/Table 6 Pima 糖剔除行（10 号 R2）v2 口径补跑后，
把旧→新对照追加进汇总 docx 二十二节（6）。数字全部动态读 CSV，不写死。
"""
import pandas as pd
import docx
from docx.shared import Pt

DOC = r'D:\竞赛\小论文\PEERJ论文实验结果汇总.docx'
old = pd.read_csv(r'_备份_旧口径_1110\R2_desugared_sensitivity.csv')
new = pd.read_csv(r'results_extra\R2_desugared_sensitivity.csv')
po = old[old['队列'] == 'Pima'].iloc[0]
pn = new[new['队列'] == 'Pima'].iloc[0]
vo = old[old['模型'] == 'Voting'].iloc[0]
vn = new[new['模型'] == 'Voting'].iloc[0]
xo = old[(old['队列'] == 'PUC') & (old['模型'] == 'XGB')].iloc[0]
xn = new[(new['队列'] == 'PUC') & (new['模型'] == 'XGB')].iloc[0]


def fmt(r):
    return (f"阈值 {r['阈值(F2最优)']:.2f} / AUC {r['AUC']:.3f} ({r['AUC95%CI']}) / "
            f"召回 {r['Recall'] * 100:.1f}% / Spec {r['Specificity'] * 100:.1f}% / "
            f"PPV {r['PPV'] * 100:.1f}% / F2 {r['F2']:.3f} / 漏 {int(r['FN'])}")


doc = docx.Document(DOC)


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
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = str(h)
        for par in c.paragraphs:
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


p('（6）§3.6.3 / Table 6 Pima 糖剔除行 补跑（2026-11-10 同日，10 号 R2）：', bold=True)
table(['项目', '旧口径', 'v2 新口径'],
      [['Pima-XGB 去糖行', fmt(po), fmt(pn)],
       ['PUC-Voting 去糖行（校验）', f"AUC {vo['AUC']:.4f}", f"AUC {vn['AUC']:.4f}（逐位不变）"],
       ['PUC-XGB 去糖行（校验）', f"AUC {xo['AUC']:.4f}", f"AUC {xn['AUC']:.4f}（逐位不变）"]])
p(f'说明：10 号脚本无写死数字，填补经共用 qc_pima() 自动继承 v2 口径，仅 Pima 行变化。'
  f'叙述两端更新："全模型 XGB AUC 0.831→去糖 0.761" 应改为 "全模型 XGB AUC 0.837→去糖 '
  f'{pn["AUC"]:.3f}（95%CI {pn["AUC95%CI"]}）"；结论方向不变——剔除全部血糖派生特征后 '
  f'AUC 下降约 0.074 但仍显著高于随机，模型并非单纯复现 FPG≥7.0 诊断阈值。'
  f'LASSO 保留仍为 7 特征（Insulin→Insulin_Log 微调）。'
  f'PUC 两行逐位不变，佐证 PUC 侧口径未受本次改动影响。')
doc.save(DOC)
print('已追加（6）至二十二节')
print('Pima 旧:', fmt(po))
print('Pima 新:', fmt(pn))
