# -*- coding: utf-8 -*-
"""
15_补记_Pima配对差值.py — 把 R4b（Pima OOF 口径 Stacking vs XGBoost 配对差值）补记进汇总 docx（19.5 节）
"""
import pandas as pd
import docx
from docx.shared import Pt

DOC_PATH = r'C:\Users\孟雅玲\Desktop\PEERJ论文实验结果汇总_19.5补记待合并.docx'
r4b = pd.read_csv('results_extra/R4b_pima_oof_paired_bootstrap.csv')

doc = docx.Document(DOC_PATH)

# 找到"二十、"标题，在其前插入 19.5 —— python-docx 不支持直接插队，
# 改为在文末追加"补记"（编号 19.5，注明接 19.1）
h = doc.add_heading('19.5 补记：Pima 队列 OOF 口径 Stacking vs XGBoost 配对差值（2026-11-09）', level=2)
p = doc.add_paragraph()
r = p.add_run('背景：R1（19.1）显示 Pima 的 OOF 预设规则选中 Stacking（OOF F2 0.790）而非 XGBoost（0.776），'
              '两者差距很小。本检验按 R4 同款口径（Pima 训练集内 5 折 OOF，2000 次配对重采样，种子 42）'
              '检验该差距是否可区分，用于评估"最优模型改名"的实际影响。脚本 14_Pima_OOF配对差值.py。')
r.font.size = Pt(10.5)

t = doc.add_table(rows=1, cols=6)
try:
    t.style = 'Table Grid'
except Exception:
    pass
hdr = ['对比', '点估计', '95%CI', 'CI 是否含 0', '重采样次数', '含义']
for i, v in enumerate(hdr):
    t.rows[0].cells[i].text = v
    for par in t.rows[0].cells[i].paragraphs:
        for run in par.runs:
            run.bold = True
            run.font.size = Pt(9)
meanings = ['Stacking 的 OOF AUC 显著高于 XGBoost（AUC 非选择指标）',
            '选择指标（F2）上两者统计不可区分']
for (_, row), meaning in zip(r4b.iterrows(), meanings):
    cells = t.add_row().cells
    vals = [row['comparison'], f"{row['point_diff']:+.4f}",
            f"[{row['CI95_low']:+.4f}, {row['CI95_high']:+.4f}]",
            '是' if row['CI含0'] else '否', str(int(row['n_boot'])), meaning]
    for i, v in enumerate(vals):
        cells[i].text = str(v)
        for par in cells[i].paragraphs:
            for run in par.runs:
                run.font.size = Pt(9)
doc.add_paragraph()

p = doc.add_paragraph()
r = p.add_run('解读：OOF_F2 差值 95%CI [−0.001, +0.030] 包含 0——按选择规则的指标（F2）看，'
              'Stacking 与 XGBoost 统计上不可区分，"最优"归属是程序性结果而非优越性证据'
              '（与 19.4 的 PUC Voting/LGB/Stacking 结论同款表述）；OOF_AUC 差值 +0.0165，'
              '95%CI [+0.005, +0.028] 不含 0，但 AUC 不是 §2.4 预设规则的排序指标，不改变上述结论。')
r.font.size = Pt(10.5)

p = doc.add_paragraph()
r = p.add_run('对改稿的影响：若按 (a) 如实改，Pima"最优模型"换为 Stacking，但表1 七行测试集数字一行不用换，'
              '只需替换归属表述与表3 Pima 列（Stacking 测试集 AUC 0.844 / 召回 95.1% / F2 0.817 / 漏诊 4，'
              '头条数字反而变好）；配合本节 CI 结果可补一句"Stacking 与 XGBoost 的 OOF F2 差异 CI 包含 0"，'
              '使叙事与 19.4 完全同构。若审稿人追问为何 OOF 换了名，一句话即可答：'
              '选择规则本就应在验证口径上执行，OOF 才是它的合法执行场所。')
r.font.size = Pt(10.5)

doc.save(DOC_PATH)
print('已补记 19.5 到', DOC_PATH)
