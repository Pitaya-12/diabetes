# -*- coding: utf-8 -*-
"""
21_生成_TRIPOD清单1110.py — 以 TRIPOD-AI_Checklist.docx 为底稿，生成与
Diabetes1110_PeerJ.docx 正文逐项对齐的 TRIPOD-AI_Checklist_1110.docx
（同目录 D:\\竞赛\\小论文\\）。
必改三处：D4（EPV 32.3/11.7）、J1（随机分层 7:3 切分）、H1（去 Zenodo）。
附带对齐：标题、章节号漂移（2.2.x/3.7/3.8）、Chinese→PUC、阈值口径(I5)、
补充材料编号(O4/O5/O6)等，全部逐条列在修改说明 txt 中。
"""
import docx

SRC = r'D:\竞赛\小论文\TRIPOD-AI_Checklist.docx'
DST = r'D:\竞赛\小论文\TRIPOD-AI_Checklist_1110.docx'

NEW_TITLE = ('Cross-cohort evaluation of machine-learning models for prevalent '
             'diabetes identification: XGBoost, voting and stacking ensembles, '
             'and SHAP-based interpretability across two public datasets')

# (旧文本片段, 新整格文本) —— 按段落/单元格全文替换
REPL = [
    # —— 必改 1：D4 EPV ——
    ('Methods 2.5.1 — post hoc EPV analysis (Chinese EPV ≈ 20.4, Pima EPV ≈ 10.1)',
     'Methods 2.5.1 — post hoc EPV analysis (PUC development set EPV ≈ 32.3, '
     'Pima development set EPV ≈ 11.7)'),
    # —— 必改 2：J1 验证类型 ——
    ('Methods 2.5.3 — internal stability test (index-based partition); cross-cohort transfer experiment',
     'Methods 2.5.3 / Results 3.6.1 — internal stability test (five independent '
     '7:3 stratified random splits); cross-cohort transfer experiment (Results 3.2.4)'),
    # —— 必改 3：H1 去 Zenodo（按"已上传 Pima 文件"的最终状态写）——
    ('Data Availability — GitHub repository and Zenodo archive',
     'Data Availability — GitHub release v1.0.0 (analysis scripts, model-derived '
     'outputs, and the Pima tabular file archived as a release asset)'),
    # —— 标题 ——
    ('Manuscript Title: Cross-cohort evaluation of machine-learning models for '
     'diabetes risk screening: XGBoost, Voting ensembles, and SHAP-based '
     'interpretability across two public datasets',
     'Manuscript Title: ' + NEW_TITLE),
    ('Title (page 1) — "Cross-cohort evaluation of machine-learning models for diabetes risk screening"',
     'Title (page 1) — "Cross-cohort evaluation of machine-learning models for prevalent '
     'diabetes identification: XGBoost, voting and stacking ensembles, and SHAP-based '
     'interpretability across two public datasets"'),
    # —— 日期 ——
    ('Date: October 2026', 'Date: November 2026'),
    # —— 引文措辞对齐 ——
    ('Introduction, paragraph 3 — "methodological comparison and systematic evaluation"',
     'Introduction, paragraph 3 — "a methodological framework for cross-cohort '
     'comparison of diabetes prediction models"'),
    # —— 章节号漂移 / 术语（Chinese→PUC）——
    ('Methods 2.1.1 (Pima), 2.1.2 (Chinese)',
     'Methods 2.1.1 (Pima), 2.1.2 (PUC)'),
    ('Methods 2.2.3 — stratified random sampling, fixed random seed',
     'Methods 2.2.4 — stratified random sampling, fixed random seed'),
    ('Methods 2.2.3 — Pima: 768 (537 training, 231 test); Chinese: 4,061 (2,842 training, 1,219 test)',
     'Methods 2.2.4 — Pima: 768 (537 training, 231 test); PUC: 4,061 (2,842 training, 1,219 test)'),
    ('Methods 2.1.3 — ADA guidelines and Chinese Guidelines for T2DM '
     '(FPG ≥ 7.0, 2h-PPG ≥ 11.1, HbA1c ≥ 6.5%, or documented diagnosis)',
     'Methods 2.1.3 — ADA guidelines and Chinese Guidelines for T2DM '
     '(FPG ≥ 7.0 mmol/L, 2h-PPG ≥ 11.1 mmol/L, or HbA1c ≥ 6.5%)'),
    ('Methods 2.1.2 (17 clinical indicators), 2.2.2 (feature engineering to 25 variables)',
     'Methods 2.1.2 (17 candidate predictors), 2.2.3 (feature engineering to 25 candidates)'),
    ('Methods 2.2.2 — interaction terms, lipid ratios, log-transformed variables, metabolic risk score',
     'Methods 2.2.2 (categorical coding and scaling) and 2.2.3 — interaction terms, '
     'lipid ratios, log-transformed variables, metabolic risk score'),
    ('Methods 2.3–2.4 — LASSO feature screening, grid search + 5-fold CV',
     'Methods 2.3–2.4.1 — LASSO feature screening; grid search + five-fold CV; '
     'out-of-fold model selection (2.4.1)'),
    ('Methods 2.4 — grid search with 5-fold CV; brief search spaces predefined',
     'Methods 2.4 — grid search with five-fold CV; search spaces defined in the archived scripts'),
    ('Data Availability — analysis scripts and trained model files publicly available',
     'Data Availability — analysis scripts and model-derived outputs at GitHub release v1.0.0'),
    # —— I5 阈值口径（正文 1110 已改为双口径）——
    ('Methods 2.5.1 — threshold maximising F₂ on validation folds within training-set CV',
     'Methods 2.4.1 and 2.5.1 — F₂-maximising threshold scanned 0.10–0.70 on the '
     'held-out test set (optimistic operating point); protocol-faithful threshold '
     'derived from pooled out-of-fold predictions (2.4.1)'),
    ('Methods 2.4 — all tuning within training-set CV only; 2.2.3 — deduplication before split',
     'Methods 2.4/2.4.1 — tuning and model selection within the training set '
     '(out-of-fold protocol); 2.2.4 — deduplication before split'),
    # —— J3 敏感性分析补 3.6.2/3.6.3 ——
    ('Methods 2.2.1 — MICE imputation; Results 3.6 — repeated-value filtering sensitivity',
     'Methods 2.2.1 — MICE imputation (Supplemental Table S2); Results 3.6.2 — '
     'repeated-value filtering; Results 3.6.3 — glucose-derived predictor removal'),
    # —— L2/L3 章节号（成本已移至 3.8，耗时为 3.7）——
    ('Methods/Results 3.7.1 — scenario simulation with hypothetical parameters',
     'Methods/Results 3.8 — template-only cost workflow with hypothetical parameters '
     '(Supplemental Table S4)'),
    ('Methods/Results 3.7.2 — CPU inference benchmarking',
     'Methods 2.5.3 / Results 3.7 — CPU inference benchmarking'),
    # —— M2/M5 ——
    ('Discussion 4.5 — six subsections of limitations',
     'Discussion 4.5 — nine subsections of limitations'),
    ('Conclusions — "cannot be directly applied for clinical diabetes screening '
     'before multicenter prospective validation"',
     'Conclusions — "neither model should be applied to clinical screening before '
     'multicentre prospective validation"'),
    # —— N1：正文 1110 为"无需伦理审批"，清单原写"委员会豁免"，对齐 ——
    ('Human Ethics — exemption granted by institutional ethics committee',
     'Human Ethics — ethical approval not required for secondary analysis of '
     'publicly released de-identified data'),
    # —— N2（同 H1 口径）——
    ('Data Availability — GitHub + Zenodo; raw data not redistributed due to license',
     'Data Availability — GitHub release v1.0.0; Pima tabular file archived as a '
     'release asset; Kaggle raw records not redistributed due to license'),
]

# 表 5（F 区）F4 行：原文被拆成 c2/c3/c4 三格，行级合并
F4_NEW = ('Methods 2.3 — L1-regularised LASSO; penalty selected by five-fold CV; '
          '|coefficient| > 0.001 threshold (screening repeated within folds, Section 2.4.1)')

# 表 14（O 区）行级修改：列 2 = Status
O_EDITS = {
    4: 'Supplemental Table S1',          # O4 重复取值表：S2 → S1（正文 2.1.2）
    5: 'Supplemental Table S4',          # O5 成本敏感性：S1 → S4（正文 3.8）
    6: 'GitHub release v1.0.0',          # O6 去 Zenodo
}

# 补行（正文 1110 引用但 O 区缺失的材料）
O_ADD = [
    ('Sensitivity note on highly repeated discrete values', 'Supplemental Note S1'),
    ('Out-of-fold performance of all seven models', 'Supplemental Table S3'),
]

# 表 14（O 区）行级修改：列 2 = Status
O_EDITS = {
    4: 'Supplemental Table S1',          # O4 重复取值表：S2 → S1（正文 2.1.2）
    5: 'Supplemental Table S4',          # O5 成本敏感性：S1 → S4（正文 3.8）
    6: 'GitHub release v1.0.0',          # O6 去 Zenodo
}


def set_para_text(p, new):
    if p.runs:
        p.runs[0].text = new
        for r in p.runs[1:]:
            r.text = ''
    else:
        p.add_run(new)


def main():
    doc = docx.Document(SRC)
    hits = {old: 0 for old, _ in REPL}

    def apply_repl(paras):
        for p in paras:
            t = p.text
            for old, new in REPL:
                if old in t:
                    set_para_text(p, new)
                    hits[old] += 1

    apply_repl(doc.paragraphs)
    for tb in doc.tables:
        for row in tb.rows:
            for cell in row.cells:
                apply_repl(cell.paragraphs)

    # F4 行：三格合并到 c2，c3/c4 清空
    row = doc.tables[5].rows[4]
    set_para_text(row.cells[2].paragraphs[0], F4_NEW)
    for p in row.cells[2].paragraphs[1:]:
        set_para_text(p, '')
    for ci in (3, 4):
        for p in row.cells[ci].paragraphs:
            set_para_text(p, '')
    print('F4 行已合并 ->', F4_NEW[:50], '...')

    # O 区行级替换（Status 列）
    tb = doc.tables[14]
    for ri, new in O_EDITS.items():
        cell = tb.rows[ri].cells[2]
        set_para_text(cell.paragraphs[0], new)
        for p in cell.paragraphs[1:]:
            set_para_text(p, '')
        print('O 行 %d Status -> %s' % (ri, new))

    # O 区补行
    for desc, status in O_ADD:
        cells = tb.add_row().cells
        set_para_text(cells[0].paragraphs[0], 'O%d' % (len(tb.rows) - 1))
        set_para_text(cells[1].paragraphs[0], desc)
        set_para_text(cells[2].paragraphs[0], status)
        print('O 补行 -> %s | %s' % (desc, status))

    doc.save(DST)

    print('--- 替换命中情况（应全部 >=1）---')
    for old, _ in REPL:
        n = hits[old]
        flag = 'OK' if n else '**未命中**'
        print('%s x%d  <- %s' % (flag, n, old[:60]))
    print('已保存:', DST)


if __name__ == '__main__':
    main()
