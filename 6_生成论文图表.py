# -*- coding: utf-8 -*-
"""
6_生成论文图表.py — 一键生成论文终稿图（论文 图1~图5）
  图1 ROC / 图2 消融 / 图3 阈值曲线 / 图4 DCA / 图5 SHAP
运行前请先运行 2_中国数据建模.py 和 3_消融实验.py
输出：figures_paper/（含 metrics_check.csv，供核对正文数字）
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'figures_paper'
OPT_THRESHOLD = 0.35     # Voting 最优阈值（论文 表2）
os.makedirs(OUTPUT_DIR, exist_ok=True)
METRICS = {}

# ---------------- 读取 2/3 号脚本的输出 ----------------
if not os.path.exists('results_china/y_test.npy') or not os.path.exists('results_china/all_probs.pkl'):
    sys.exit('请先运行 2_中国数据建模.py（需要 results_china/ 下的 y_test.npy 和 all_probs.pkl）')

y = np.load('results_china/y_test.npy')
probs = pickle.load(open('results_china/all_probs.pkl', 'rb'))
name = next((k for k in probs if 'vot' in k.lower()), list(probs)[0])
p = np.asarray(probs[name]).ravel()
print(f"模型: {name}，测试集 n={len(y)}")


def metrics_at(yy, pp, thr):
    pred = (pp >= thr).astype(int)
    tp = int(((pred == 1) & (yy == 1)).sum())
    tn = int(((pred == 0) & (yy == 0)).sum())
    fp = int(((pred == 1) & (yy == 0)).sum())
    fn = int(((pred == 0) & (yy == 1)).sum())
    return {'recall': tp / max(tp + fn, 1), 'specificity': tn / max(tn + fp, 1),
            'miss_rate': fn / max(tp + fn, 1), 'ppv': tp / max(tp + fp, 1),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


# ---------------- 图1 ROC ----------------
fpr, tpr, _ = roc_curve(y, p)
auc_value = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'Voting ensemble (AUC = {auc_value:.3f})')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random classifier (AUC = 0.500)')
ax.set_xlabel('False positive rate (1 - specificity)')
ax.set_ylabel('True positive rate (sensitivity)')
ax.set_title('ROC curve - provenance-unverified public clinical cohort (PUC)')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Figure1_ROC_PUC.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  -> {OUTPUT_DIR}/Figure1_ROC_PUC.png')

# ---------------- 图2 消融 ----------------
abl_path = 'results_ablation/ablation_results.csv'
if os.path.exists(abl_path):
    df = pd.read_csv(abl_path)
    # 按 D(基线), B(无特征工程), C(无LASSO), A(完整) 排列
    order = [3, 1, 2, 0]
    labels = ['D\nbaseline', 'B\nno FE', 'C\nno LASSO', 'A\nfull']
    aucs = [df.loc[i, 'auc'] for i in order]
    recs = [df.loc[i, 'recall'] for i in order]
    miss = [df.loc[i, 'miss_rate'] for i in order]
else:
    sys.exit('请先运行 3_消融实验.py（需要 results_ablation/ablation_results.csv）')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(4)
w = 0.35
b1 = axes[0].bar(x - w / 2, aucs, w, label='AUC', color='#3498db', edgecolor='white')
b2 = axes[0].bar(x + w / 2, recs, w, label='Recall', color='#2ecc71', edgecolor='white')
axes[0].set_ylabel('Score')
axes[0].set_title('Ablation study: AUC vs recall (threshold = 0.35)')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
axes[0].legend(loc='lower right'); axes[0].set_ylim(0.80, 0.98)
axes[0].grid(True, alpha=0.2, axis='y')
for bar, v in list(zip(b1, aucs)) + list(zip(b2, recs)):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{v:.3f}', ha='center', fontsize=9)
b3 = axes[1].bar(x, miss, color=['#e74c3c', '#e67e22', '#f39c12', '#27ae60'], edgecolor='white')
axes[1].set_ylabel('Miss rate (%)')
axes[1].set_title('Ablation study: miss rate comparison (threshold = 0.35)')
axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
axes[1].set_ylim(0, 16); axes[1].grid(True, alpha=0.2, axis='y')
for bar, v in zip(b3, miss):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Figure2_Ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  -> {OUTPUT_DIR}/Figure2_Ablation.png')

# ---------------- 图3 阈值-性能曲线 ----------------
ths = np.arange(0.10, 0.701, 0.05)
rec, spe, mis = [], [], []
for t in ths:
    m = metrics_at(y, p, t)
    rec.append(m['recall']); spe.append(m['specificity']); mis.append(m['miss_rate'])
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(ths, rec, 'b-o', linewidth=2.5, markersize=8, label='Recall (sensitivity)')
ax.plot(ths, spe, 'g-s', linewidth=2.5, markersize=8, label='Specificity')
ax.plot(ths, mis, 'r-^', linewidth=2.5, markersize=8, label='Miss rate')
ax.axvline(x=OPT_THRESHOLD, color='purple', linestyle='--', linewidth=2.5,
           label=f'Optimal threshold = {OPT_THRESHOLD:.2f}')
ax.set_xlabel('Threshold'); ax.set_ylabel('Metric value')
ax.set_title('Threshold-performance curve - PUC cohort')
ax.legend(loc='center right'); ax.grid(True, alpha=0.3)
ax.set_xlim([0.05, 0.75]); ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Figure3_Threshold_PUC.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'  -> {OUTPUT_DIR}/Figure3_Threshold_PUC.png')

# ---------------- 图4 DCA ----------------
ths = np.arange(0.01, 0.501, 0.02)
n = len(y)
prev = float(y.mean())
nb_model = []
for t in ths:
    m = metrics_at(y, p, t)
    nb_model.append(m['tp'] / n - (m['fp'] / n) * (t / (1 - t)))
nb_all = prev - (1 - prev) * (ths / (1 - ths))
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(ths, nb_model, 'b-', linewidth=2.5, label='Voting ensemble')
ax.plot(ths, nb_all, 'g--', linewidth=2, label='Treat all')
ax.plot(ths, np.zeros_like(ths), 'r--', linewidth=2, label='Treat none')
ax.axvline(x=OPT_THRESHOLD, color='purple', linestyle=':', linewidth=2,
           label=f'Optimal threshold = {OPT_THRESHOLD:.2f}')
ax.set_xlabel('Risk threshold'); ax.set_ylabel('Net benefit')
ax.set_title('Decision curve analysis - PUC cohort')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9, frameon=False)
ax.grid(True, alpha=0.3); ax.set_xlim([0, 0.52])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Figure4_DCA_PUC.png', dpi=300, bbox_inches='tight')
plt.close()
i_max = int(np.argmax(nb_model))
METRICS['dca_max_nb'] = round(float(nb_model[i_max]), 3)
print(f'  -> {OUTPUT_DIR}/Figure4_DCA_PUC.png')

# ---------------- 图5 SHAP ----------------
shap_path = 'results_china/shap_voting.csv'
if os.path.exists(shap_path):
    df = pd.read_csv(shap_path).sort_values('mean_abs_shap', ascending=False).head(10)
    vals = df['mean_abs_shap'].astype(float).values
    labels = df['feature'].tolist()
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(vals))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(vals)))
    ax.barh(y_pos, vals, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('SHAP feature importance - PUC cohort (Voting ensemble)')
    ax.invert_yaxis(); ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure5_SHAP_PUC.png', dpi=300, bbox_inches='tight')
    plt.close()
    METRICS['shap_top3'] = ' > '.join(labels[:3])
    print(f'  -> {OUTPUT_DIR}/Figure5_SHAP_PUC.png')
else:
    print('  [跳过图5] 未找到 results_china/shap_voting.csv（SHAP 计算失败？）')

# ---------------- 指标核对表 ----------------
m = metrics_at(y, p, OPT_THRESHOLD)
METRICS.update(n=len(y), model=name, auc=round(float(auc_value), 3),
               recall=round(m['recall'], 3), specificity=round(m['specificity'], 3),
               ppv=round(m['ppv'], 3), missed_rate=round(m['miss_rate'], 3),
               tp=m['tp'], fp=m['fp'], fn=m['fn'], tn=m['tn'], prevalence=round(prev, 3))
pd.Series(METRICS).to_csv(f'{OUTPUT_DIR}/metrics_check.csv')
print(f"\n=== 由真实预测重算的指标（请与论文正文核对）===")
print(f"AUC={auc_value:.3f}  召回率={m['recall']:.3f}  特异度={m['specificity']:.3f}  "
      f"PPV={m['ppv']:.3f}  漏诊率={m['miss_rate']:.1%}  (tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})")
print(f"已保存: {OUTPUT_DIR}/metrics_check.csv")
