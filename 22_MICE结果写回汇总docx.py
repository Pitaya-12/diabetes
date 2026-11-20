# -*- coding: utf-8 -*-
"""
22_MICE结果写回汇总docx.py — 把 v2 重跑的 MICE 敏感性结果写回
《D:\\竞赛\\小论文\\PEERJ论文实验结果汇总.docx》十三节：
  1) 替换十三节 MICE 表（4 行）为 results_mice/mice_sensitivity.csv 新数字
     （"主分析"两行逐位复现表1 Stacking / 表2 Voting，锚点已由 7 号脚本验收）；
  2) 重写表后结论段（含 v1 错误勘误、ΔAUC/Δ召回 实测值、结论判断）；
  3) 在"三、不动的部分"段落追加更新注记（MICE 已重跑，不再属于不动部分）；
  4) 全文扫描 v1 旧数字残留并列出位置。
所有数字从 CSV 读取，不写死。运行前请先跑 7_MICE敏感性分析.py。
"""
import os
import pandas as pd
import docx
from docx.shared import Pt

DOC_PATH = r'D:\竞赛\小论文\PEERJ论文实验结果汇总.docx'
CSV_PATH = os.path.join('results_mice', 'mice_sensitivity.csv')

# ====================== 读取并整理结果 ======================
res = pd.read_csv(CSV_PATH)
order_cohort = ['PUC', 'Pima']
order_arm = ['中位数填补(主分析)', 'MICE(20次迭代)']
res['_c'] = res['队列'].map({c: i for i, c in enumerate(order_cohort)})
res['_a'] = res['填补方式'].map({a: i for i, a in enumerate(order_arm)})
res = res.sort_values(['_c', '_a']).reset_index(drop=True)
assert len(res) == 4, f'期望 4 行结果，实际 {len(res)} 行'

base = res[res['填补方式'] == '中位数填补(主分析)'].set_index('队列')
mice = res[res['填补方式'] == 'MICE(20次迭代)'].set_index('队列')


def fmt_row(r):
    return (f"{r['队列']}-{r['填补方式']}, 阈值{r['阈值']:.2f}", str(int(r['特征数'])),
            f"{r['AUC']:.4f}", f"{r['召回率'] * 100:.1f}%", f"{r['PPV'] * 100:.1f}%",
            f"{r['F2']:.3f}", str(int(r['漏诊'])))


table_rows = [fmt_row(r) for _, r in res.iterrows()]

# 结论判断（ΔAUC<0.02 且 Δ召回<0.05 为稳定）
verdicts = {}
for c in order_cohort:
    dauc = abs(mice.loc[c, 'AUC'] - base.loc[c, 'AUC'])
    drec = abs(mice.loc[c, '召回率'] - base.loc[c, '召回率'])
    verdicts[c] = (dauc, drec, dauc < 0.02 and drec < 0.05)
all_ok = all(v[2] for v in verdicts.values())


def verdict_txt(c):
    dauc, drec, ok = verdicts[c]
    edge = '，接近 5pp 边界' if (ok and drec >= 0.04) else ''
    return (f"{c} 队 AUC 变化 {dauc:.4f}、召回率变化 {drec:.4f}"
            f"（{'<0.02/<0.05，稳定' + edge if ok else '超出判据，需如实报告'}）")


puc_fix = base.loc['PUC']
pima_fix = base.loc['Pima']
puc_m = mice.loc['PUC']
pima_m = mice.loc['Pima']

conclusion = (
    '结论【2026-11-10 v2 重跑，勘误 v1】：v1 本表 Pima"主分析"行'
    '（AUC 0.8360 / 召回 88.9% / F2 0.802 / 漏 9）系旧口径产物——固定超参 XGBoost'
    '（200/3/0.1）+ 固定阈值 0.20，与表1 任何一行都对不上；PUC"主分析"行（0.9637 / 93.4%）'
    '与表2（0.965 / 92.5%）亦有偏差。v2 已改为与主分析完全同口径的全管线'
    '（质控→特征工程→7:3 分层划分(种子42)→LASSO 仅训练集→5 模型网格搜索→集成→F2 最优阈值），'
    '两臂共用同一质控行集，唯一自变量为插补方式（中位数 vs MICE 20 次迭代）。'
    f'中位数填补臂已逐位复现主分析：Pima 与表1 Stacking 行一致（AUC {pima_fix["AUC"]:.4f} / '
    f'召回 {pima_fix["召回率"] * 100:.1f}% / F2 {pima_fix["F2"]:.3f} / 漏 {int(pima_fix["漏诊"])} / '
    f'阈值 {pima_fix["阈值"]:.2f}），PUC 与表2 Voting 行一致（AUC {puc_fix["AUC"]:.4f} / '
    f'召回 {puc_fix["召回率"] * 100:.1f}% / F2 {puc_fix["F2"]:.3f} / 漏 {int(puc_fix["漏诊"])} / '
    f'阈值 {puc_fix["阈值"]:.2f}）。MICE 臂对比：{verdict_txt("PUC")}；{verdict_txt("Pima")}。'
    + ('主结论稳定——论文 2.2.1 的声称成立。' if all_ok else
       '结论并非全部稳定——论文 2.2.1 需软化为"总体一致、个别指标有差异"并如实报告上列变化。')
    + f'（补充：固定主分析阈值下 MICE 臂 PUC 召回 {puc_m["固定阈值_召回率"] * 100:.1f}% / '
      f'漏 {int(puc_m["固定阈值_漏诊"])}，Pima 召回 {pima_m["固定阈值_召回率"] * 100:.1f}% / '
      f'漏 {int(pima_m["固定阈值_漏诊"])}，见 results_mice/mice_sensitivity.csv"固定阈值_"列。）'
)

# ====================== 写入 docx ======================
doc = docx.Document(DOC_PATH)


def set_para_text(p, text, size=10.5, bold=False):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)


def set_cell(cell, text, size=9, bold=False):
    cell.text = text
    for par in cell.paragraphs:
        for run in par.runs:
            run.bold = bold
            run.font.size = Pt(size)


# 1) 十三节 MICE 表：按内容定位（首列含'中位数填补'的表）
target_tbl = None
for tb in doc.tables:
    first_col = [r.cells[0].text for r in tb.rows]
    if any('中位数填补' in t for t in first_col) and any('MICE' in t for t in first_col):
        target_tbl = tb
        break
assert target_tbl is not None, '未找到十三节 MICE 表'
assert len(target_tbl.rows) == 5 and len(target_tbl.columns) == 7, \
    f'MICE 表形状异常: {len(target_tbl.rows)}x{len(target_tbl.columns)}'

headers = ['分析', '特征数', 'AUC', '召回率', 'PPV', 'F2', '漏诊']
for j, h in enumerate(headers):
    set_cell(target_tbl.rows[0].cells[j], h, bold=True)
for i, row_vals in enumerate(table_rows):
    for j, v in enumerate(row_vals):
        set_cell(target_tbl.rows[i + 1].cells[j], v)
print('十三节 MICE 表已替换:')
for row_vals in table_rows:
    print('  ', ' | '.join(row_vals))

# 2) 结论段：定位十三节结论段（兼容首版"结论：MICE"与 v2 版"结论【…】"）
done_concl = False
for p in doc.paragraphs:
    t = p.text.strip()
    if t.startswith('结论') and ('MICE' in t) and ('本表' in t or '插补' in t):
        set_para_text(p, conclusion)
        done_concl = True
        break
assert done_concl, '未找到十三节结论段'
print('\n结论段已重写')

# 3) "三、不动的部分"段落追加更新注记（幂等：已有注记则跳过）
for p in doc.paragraphs:
    if '三、不动的部分' in p.text and 'MICE' in p.text and '不再属于' not in p.text:
        run = p.add_run('【2026-11-10 更新：MICE 一项已按 v2 重跑修正（见十三节勘误），不再属于"不动"部分。】')
        run.font.size = Pt(10.5)
        print('已追加"不动的部分"更新注记')
        break

doc.save(DOC_PATH)
print(f'\n已保存: {DOC_PATH}')

# ====================== 4) 全文扫描 v1 旧数字残留 ======================
STALE = ['0.8360', '0.8269', '0.9637', '0.9636', '0.802', '0.8186',
         '93.4%', '90.8%', '88.9%', '91.4%', '57.6%', '57.8%',
         '0.888', '0.874', 'AUC 变化 <0.001', '漏 9', '漏9']
doc2 = docx.Document(DOC_PATH)
print('\n===== v1 旧数字残留扫描 =====')
hits = 0
for i, p in enumerate(doc2.paragraphs):
    for s in STALE:
        if s in p.text:
            hits += 1
            print(f'[para {i}] 命中"{s}":', p.text[:120])
            break
for ti, tb in enumerate(doc2.tables):
    txt = '\n'.join(' | '.join(c.text for c in r.cells) for r in tb.rows)
    for s in STALE:
        if s in txt:
            hits += 1
            print(f'[table {ti}] 命中"{s}"（表首行: '
                  f'{" | ".join(c.text for c in tb.rows[0].cells)[:60]}）')
            break
if hits == 0:
    print('无残留')
else:
    print(f'\n共 {hits} 处命中——其中阈值敏感性表中的 88.9%/91.4% 等为 XGB 阈值扫描的合法数字，'
          '无需修改；其余请人工核对。')

# ====================== 复核：重新提取十三节 ======================
print('\n===== 复核：保存后的十三节 =====')
for tb in doc2.tables:
    first_col = [r.cells[0].text for r in tb.rows]
    if any('中位数填补' in t for t in first_col):
        for r in tb.rows:
            print('  ', ' | '.join(c.text for c in r.cells))
for p in doc2.paragraphs:
    if p.text.strip().startswith('结论【'):
        print('\n结论段:', p.text)
