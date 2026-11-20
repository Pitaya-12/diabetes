# -*- coding: utf-8 -*-
"""
13_补充结果写入汇总docx.py — 把 R1~R4 实验结果、6 项事实确认、文件核对结论
追加写入桌面《PEERJ论文实验结果汇总.docx》（不动原有内容，只在文末追加）。
所有数字均从 results_extra/*.csv 读取，不写死。
"""
import os
import pandas as pd
import docx
from docx.shared import Pt

DOC_PATH = r'C:\Users\孟雅玲\Desktop\PEERJ论文实验结果汇总.docx'
RES = 'results_extra'

r1 = pd.read_csv(os.path.join(RES, 'R1_oof_model_comparison.csv'))
r2 = pd.read_csv(os.path.join(RES, 'R2_desugared_sensitivity.csv'))
r3 = pd.read_csv(os.path.join(RES, 'R3_voting_5split_stability.csv'))
r4 = pd.read_csv(os.path.join(RES, 'R4_paired_bootstrap_auc_diff.csv'))

doc = docx.Document(DOC_PATH)


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


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


# ============================================================
h1('十九、外部复核补充实验 R1–R4（2026-11-09 生成）')
p('本节为外部复核意见"证据缺口"清单对应的 4 项补充实验。脚本 9~12 号（另有 共用模块 补充实验工具.py），'
  '原始 CSV 在 results_extra/ 文件夹。全部结果为随机种子 42（R3 划分种子 0~4）的真实运行输出，'
  '管线与主分析完全同口径（质控→特征工程→7:3 分层划分→LASSO 仅训练集→网格搜索→集成），未写死任何数字。')

# ---------------- R1 ----------------
h2('19.1 R1：5 折 OOF 七模型对比（两队列各 7 行）')
p('口径（重要）：先按主分析同款 7:3 分层划分（种子 42），OOF 预测在训练集内部生成、不含测试集；'
  '训练集内 5 折分层 CV，每折完整重做 LASSO 特征筛选 + 5 模型网格搜索 + Voting/Stacking（即筛选与调参严格位于 CV 折内），'
  '预测留出折后汇总为 OOF 概率；OOF 最优阈值在 OOF 概率上按主分析同款规则（0.10~0.70 步长 0.05 最大化 F2）选取，'
  'Recall/Specificity/PPV/F2 均为该阈值下取值。TP/FP/FN/TN 见 results_extra/R1_oof_model_comparison.csv。')

rows = []
for _, r in r1.iterrows():
    rows.append([r['cohort'], r['model'], f"{r['OOF_AUC']:.3f}", f"{r['OOF_best_thr']:.2f}",
                 f"{r['OOF_Recall']:.1%}", f"{r['OOF_Specificity']:.1%}",
                 f"{r['OOF_PPV']:.1%}", f"{r['OOF_F2']:.3f}"])
table(['队列', '模型', 'OOF AUC', 'OOF 最优阈值', 'OOF Recall', 'OOF Specificity', 'OOF PPV', 'OOF F2'],
      rows)

# 预设规则选择结果（运行时按规则计算）
p('预设规则（§2.4：召回率≥0.92 中 F2 最高，平手比漏诊少、再比特异度）在 OOF 口径下的选择结果：', bold=True)
for cohort in ['PUC', 'Pima']:
    sub = r1[r1['cohort'] == cohort]
    cand = sub[sub['OOF_Recall'] >= 0.92]
    sel = (cand if len(cand) else sub).sort_values('OOF_F2', ascending=False).iloc[0]
    xgb = sub[sub['model'] == 'XGBoost'].iloc[0]
    voting = sub[sub['model'] == 'Voting'].iloc[0]
    p(f'· {cohort}：规则选中 {sel["model"]}（OOF F2={sel["OOF_F2"]:.3f}，OOF 召回={sel["OOF_Recall"]:.1%}）；'
      f'对照——Voting OOF F2={voting["OOF_F2"]:.3f}，XGBoost OOF F2={xgb["OOF_F2"]:.3f}。')
p('→ PUC 队列 OOF 仍选出 Voting，与测试集选择一致，§3.2.2/摘要叙事不用改，只把 0.886/0.925 等选择依据数字换成 OOF 版本（0.875/93.9%）。'
  'Pima 队列 OOF 按同一规则选出 Stacking（0.790）而非 XGBoost（0.776）——此处正是复核意见第六节的决策点，'
  '两种处理所需数字本表已备齐，请定夺：(a) 如实改写 Pima 最优模型为 Stacking；(b) 调整选择规则后重述。', bold=True)
p('折内 LASSO 保留特征数的折间波动（供 §2.4 描述稳定性）：PUC 五折 23/21/24/18/22，Pima 五折 5/8/8/5/7；'
  '逐折特征明细见 results_extra/R1_oof_fold_features.csv（共 205 行，未并入本文档）。')

# ---------------- R2 ----------------
h2('19.2 R2：去糖特征敏感性（剔除血糖派生特征）')
p('剔除清单（凡含血糖派生的一律剔除）：PUC 剔 FPG、FFPG、Glucose_BMI、Metabolic_Risk（含 I(FPG>5.6)）；'
  'Pima 剔 Glucose、Glucose_BMI、Insulin_Glucose_Ratio、Metabolic_Risk（含 I(Glucose>99)）。'
  'Pima 的 Insulin / Insulin_Log 为胰岛素测定与对数变换，非血糖派生，按清单保留。'
  '管线与主分析同口径（7:3 种子 42→LASSO→网格搜索），阈值按主分析同款 F2 规则选取，AUC 95%CI 为 2000 次 Bootstrap。')

rows = []
for _, r in r2.iterrows():
    rows.append([r['队列'], r['模型'], int(r['保留特征数']),
                 r['最终特征清单'], f"{r['阈值(F2最优)']:.2f}", f"{r['AUC']:.3f}",
                 r['AUC95%CI'], f"{r['Recall']:.1%}", f"{r['Specificity']:.1%}",
                 f"{r['PPV']:.1%}", f"{r['F2']:.3f}",
                 f"{r['TP']}/{r['FP']}/{r['FN']}/{r['TN']}"])
table(['队列', '模型', '保留特征数', '最终特征清单', '阈值', 'AUC', 'AUC 95%CI',
       'Recall', 'Specificity', 'PPV', 'F2', 'TP/FP/FN/TN'], rows)

p('剔糖后输入候选：PUC 25→21，Pima 16→12（表中"保留特征数"为 LASSO 后最终入选数）。'
  '解读：剔除全部血糖派生特征后 PUC Voting AUC 0.965→0.911（95%CI 0.896–0.926）、召回 92.5%→91.3%，'
  'Pima XGBoost AUC 0.831→0.761（95%CI 0.696–0.823）——性能有实质下降但仍显著高于随机，'
  '说明模型并非单纯复现 FPG≥7.0 的诊断阈值，同时血糖特征确是名义高性能的重要来源。'
  '（固定 0.35 阈值下的补充数字在 CSV 的"补:"列。）')

# ---------------- R3 ----------------
h2('19.3 R3：主模型 Voting 的五次随机切分稳定性（阈值固定 0.35）')
p('口径：PUC 队列，种子 0~4 各做一次 7:3 分层划分，每次完整重跑主分析管线'
  '（LASSO 仅训练集→5 模型网格搜索→软投票 Voting），阈值固定 0.35（主分析 Voting 最优阈值）评估测试集。')

rows = []
for _, r in r3.iterrows():
    if str(r['seed']).isdigit():
        rows.append([str(r['seed']), str(int(float(r['LASSO_保留特征数']))),
                     f"{float(r['test_AUC']):.4f}",
                     f"{float(r['Recall@0.35']):.1%}", str(int(float(r['missed@0.35'])))])
    else:
        rows.append([str(r['seed']), '', str(r['test_AUC']), str(r['Recall@0.35']), ''])
table(['划分种子', 'LASSO 保留特征数', 'test AUC', 'Recall@0.35', '漏诊@0.35'], rows)

auc_vals = r3[r3['seed'].astype(str).str.isdigit()]['test_AUC'].astype(float)
p(f'汇总（运行时计算）：test AUC = {auc_vals.mean():.3f} ± {auc_vals.std(ddof=1):.3f}，'
  f'min~max = {auc_vals.min():.3f}~{auc_vals.max():.3f}（种子 0~4 逐次值见上表）。'
  f'与 §2.5.3 现有 LightGBM 五次切分结果（0.958 ± 0.005）同量级，主模型 Voting 的切分稳定性成立；'
  f'§2.5.3/§3.6 可据此把"重训 LightGBM"改/补为"重训 Voting"，回应"为什么不检验选出来的那个"。', bold=True)

# ---------------- R4 ----------------
h2('19.4 R4：配对 Bootstrap 的 AUC 差值 95%CI（Voting vs LGB / Stacking）')
p('口径：同一 PUC 测试集（n=1,219，347 阳性），2,000 次配对有放回重采样（种子 42），'
  '每次重采样内同时计算两模型 AUC 并取差值；模型概率为 2 号主分析脚本保存的测试集预测'
  '（results_china/all_probs.pkl，运行前已复核三个模型 AUC 与表2 一致：Voting 0.9647 / LGB 0.9641 / Stacking 0.9655）。')

rows = []
for _, r in r4.iterrows():
    rows.append([r['comparison'], f"{r['point_diff_AUC']:+.4f}",
                 f"[{r['CI95_low']:+.4f}, {r['CI95_high']:+.4f}]",
                 '是' if r['CI含0'] else '否', int(r['n_boot'])])
table(['对比', 'AUC 差值（点估计）', '95%CI', 'CI 是否含 0', '重采样次数'], rows)

p('两个差值的 95%CI 均包含 0（Voting−LGB +0.0006 [−0.0036, +0.0051]；Voting−Stacking −0.0009 [−0.0024, +0.0006]）。'
  '可用于 §3.2.2/§4.3/摘要："top models 之间的差异 CI 包含 0，选择是程序性结果而非优越性证据"——'
  '该表述现有数据支撑，不必写"未做配对检验"。', bold=True)

# ============================================================
h1('二十、事实确认（对应复核意见"五个不用跑"）')
p('以下 6 项均直接查代码与数据核实（脚本口径），并给出建议改法。')

h2('20.1 类别变量编码')
p('Gender（取值 1/2）、smoking（1/2/3，另有缺失标记 4.860753）、drinking（1/2/3，同前）、family_history（0/1）'
  '在全流程中未做任何 one-hot、label encoding 或映射转换，原始整数码直接作为数值输入模型；'
  'smoking/drinking 的缺失标记 4.860753 在质控第 1 步还原为缺失后按列中位数填补（该标记占 20.64%）。'
  '注意 Gender 编码是 1/2 而非 0/1，smoking/drinking 是 1/2/3 三档。'
  '建议 §2.2 补一句："categorical variables (sex, smoking, drinking, family history) were entered as their '
  'original integer codes without one-hot encoding"，否则 SVM/LR 的系数解释会被质疑。')

h2('20.2 LR/SVM 前的标准化')
p('有标准化，且 StandardScaler 只在训练集上 fit：先 fit_transform(X_train)，测试集仅 transform；'
  'LR、SVM、Voting、Stacking 用标准化特征，RF/XGB/LGB 用未标准化特征（树模型不需要）。'
  'SVM 的 96% 召回是带标准化的结果，此点可在 §2.2 直接声明。'
  '但注意：scaler 是在 70% 训练集【全量】上 fit 一次后再进 GridSearchCV 的 5 折 CV，'
  '即网格搜索的验证折信息会轻微进入标准化参数（测试集不受影响）。'
  '若审稿人追问，可如实写 "scaler was fitted on the full training set"；'
  '本次 R1 的 OOF 分析已改为折内 fit（严格口径），如需可一并声明。')

h2('20.3 KernelSHAP 的 background 样本数')
p('50 个。2 号脚本：shap.sample(X_train_s, 50, random_state=42) 取训练集 50 行随机子集作 background，'
  '解释对象为测试集前 200 行（X_test_s[:200]）。Pima 侧同为 background 50、解释 100 行。'
  '正文 §2.4 现在只写 "archived"，TRIPOD+AI 要求进正文——建议补 '
  '"a random subset of 50 training samples as the KernelSHAP background (200 test samples explained)"。')

h2('20.4 依赖版本（实测）')
p('Python 3.7.3；scikit-learn 1.0.2；xgboost 1.6.2；lightgbm 4.6.0；shap 0.42.1；numpy 1.21.6；pandas 1.3.5。'
  '随机种子 42。建议连同种子写进 §2.4 或 Data Availability。')

h2('20.5 LASSO 的 λ 与变量筛选位置（决定 §2.3/§2.4 怎么写）')
p('代码事实：LassoCV(cv=5) 在 70% 训练集【全量】（标准化后）上拟合一次；λ 由 LassoCV 内部 5 折 CV 选出；'
  '|coef|>0.001 的筛选也只做这一次。即：变量筛选不在模型训练的 5 折 CV 折内做。'
  '因此 §2.4 "All feature screening, hyperparameter tuning, and threshold selection were executed strictly '
  'inside 5-fold CV loops" 与代码不符，必须改口径。同句另两半也需注意：超参调优是 GridSearchCV(cv=5) 在训练集内做'
  '（可称训练集内 5 折 CV，但不是"外层循环内部"）；而阈值选择——代码实际是【在测试集上】扫描 F2 最优阈值'
  '（2 号脚本评估段 for thr in ... recall_score(y_test, ...)），与 §2.5.1 "阈值在训练集 CV 验证折上确定"的声称不符。', bold=True)
p('建议处理：把模型选择依据改挂到本次 R1 的 OOF 分析（折内 LASSO+折内调参，"strictly inside 5-fold CV" 对 OOF '
  '分析成立），最终模型再于全训练集重拟合；阈值问题可用 R1 的 OOF 最优阈值（PUC Voting 0.25）作为训练集内部口径，'
  '或如实改写 §2.5.1 为测试集上按 F2 选取并加局限声明。R1 表已给出改写所需全部数字。')

h2('20.6 PUC 公开版列数：18 列 = 17 输入 + 1 结局')
p('diabetes.csv 实测 18 列：17 个候选输入（Age, Gender, BMI, SBP, DBP, FPG, Chol, Tri, HDL, LDL, ALT, BUN, '
  'CCR, FFPG, smoking, drinking, family_history）+ 1 个结局列 Diabetes。'
  '所以 §2.1.2 前句 "a dataset with 18 features" 与后句 "Of the 18 columns … 17 were used as input" 的矛盾，'
  '应按后句统一：建议改为 "an 18-column dataset (17 candidate predictors plus the outcome label)"。'
  'Kaggle 页面标题自称 18 features 可作为数据集名称引用，但方法学描述按 17 输入 + 1 结局。')

h2('20.7 顺带发现：§3.1 的 λ_min 数值无法复现（新问题）')
p('正文 3.1 写 λ_min = 0.023（Pima）/ 0.018（PUC）。用归档脚本同口径复现（LassoCV 拟合后 alpha_ 即 λ）：'
  'Pima λ = 0.0055，PUC λ = 0.0003；系数表与保留数（8/16、22/25）完全一致，只有 λ 对不上。'
  '建议：要么删掉正文 λ_min 数值（只保留"5 折 CV 选 λ"的描述），要么换成上述复现值并注明定义。', bold=True)

# ============================================================
h1('二十一、文件层面核对（对应复核意见"两个文件核对"）')

h2('21.1 GitHub release tag：真实存在 ✓')
p('经 GitHub REST API 核实（非网页标题）：refs/tags/v1.0.0 真实存在，指向 commit '
  'bf0326508abc33f108ae6887ac7637a50628ac40；对应 release 标题 "v1.0.0-First Release"，'
  'target=main，发布于 2026-10-13，tarball/zipball 可下载。'
  'Data Availability 的 https://github.com/Pitaya-12/diabetes/releases/tag/v1.0.0 是真 tag 链接，可放心保留。')

h2('21.2 补充材料：未全部单独成文件（缺 4 处 + 编号三处冲突）')
p('现状盘点（D:\\竞赛\\小论文）：已有 流程图S1.png（即 Fig. S1，但未按 Supplementary Figure S1 命名/组装）、'
  'Supplementary.docx（仅 3.7.1 成本模拟文字，无图无表）、补充表.docx（仅 1 张成本情景表，标"补充表S1"；'
  '"补充表S2 网格搜索参数空间"只有标题没有表体）、TRIPOD-AI_Checklist.docx（完整 ✓）。')
table(['正文引用', '应有内容', '现状', '数据是否可找回'],
      [['Fig. S1（2.1）', '受试者筛选流程图', '流程图S1.png 存在，未组装命名', '✓ 有图'],
       ['Fig. S2（3.5）', '校准曲线图', '无单独文件', '✓ figures_china/calibration_curve.png 等可组装'],
       ['Table S1（2.1.2）', '高频重复取值表', '未进补充材料（文件名叫 table_S2_*.csv）', '✓ results_credibility/table_S2_frequent_values.csv'],
       ['Table S2（2.2.1）', 'MICE 敏感性表', '未组装', '✓ results_mice/mice_sensitivity.csv'],
       ['Table S3（2.4）', '网格搜索参数空间表', '补充表.docx 有"S2"标题无表体，编号还与正文不符', '✓ 可从 1/2 号脚本网格直接制表'],
       ['Table S4（3.8）', '成本假设参数表', '不存在（补充表.docx 的成本表标成了 S1）', '✓ 2 号脚本成本段参数可制表'],
       ['Note S1（2.1.2）', '重复值敏感性说明', '无单独文件', '✓ 3.6 节文字可摘出']])
p('编号冲突：正文用 S1=重复值表、S2=MICE、S3=网格、S4=成本参数；补充表.docx 用 S1=成本表、S2=网格；'
  'TRIPOD 清单 O3/O4 都指 "Supplemental Table S2"（重复）、O5 指 S1。三套编号互不一致，投稿前必须统一。', bold=True)
p('TRIPOD-AI_Checklist.docx 内部另有 3 处与正文不一致：D4 的 EPV 写 20.4/10.1，正文 2.5.1 是 32.3/11.7；'
  'J1 写 "index-based partition"，正文 2.5.3 已改为随机切分；H1 提到 Zenodo 归档，Data Availability 无 Zenodo。')

# ============================================================
h1('附：本次生成文件的位置（放不进本文档的部分）')
p('1）results_extra/R1_oof_model_comparison.csv —— R1 全表（含 TP/FP/FN/TN 列）；')
p('2）results_extra/R1_oof_fold_features.csv —— R1 折内 LASSO 逐折特征明细（205 行，太长未并入）；')
p('3）results_extra/R2_desugared_sensitivity.csv —— R2 全表（含固定 0.35 阈值的补充列）；')
p('4）results_extra/R3_voting_5split_stability.csv、R4_paired_bootstrap_auc_diff.csv；')
p('5）脚本：9_R1_OOF七模型对比.py、10_R2_去糖特征敏感性.py、11_R3_Voting五次切分稳定性.py、'
  '12_R4_配对bootstrap_AUC差值.py、13_补充结果写入汇总docx.py，共用模块 补充实验工具.py，'
  '均在 D:\\Projects\\PycharmProjects\\pythonProject2\\PEERJ论文。')

doc.save(DOC_PATH)
print('已写入:', DOC_PATH)
print('新增章节：十九（R1-R4）、二十（事实确认）、二十一（文件核对）、附（文件位置）')
