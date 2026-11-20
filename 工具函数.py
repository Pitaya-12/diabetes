# -*- coding: utf-8 -*-
"""
工具函数.py — 公共分析函数（被 1~6 号脚本调用，不用单独运行）
包含：Bootstrap 置信区间、Brier 分数、校准曲线、阈值敏感性分析、推理耗时测试
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                             confusion_matrix, recall_score, precision_score)
from sklearn.calibration import calibration_curve


# ============================================================
# 1. AUC 的 Bootstrap 95% 置信区间（论文 3.2.3）
# ============================================================
def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=42):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    auc_point = roc_auc_score(y_true, y_prob)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return auc_point, float(lo), float(hi)


# ============================================================
# 2. Brier 分数 + Bootstrap 95% 置信区间（论文 3.5 校准评估）
# ============================================================
def brier_with_bootstrap_ci(y_true, y_prob, n_boot=2000, seed=42):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    brier_point = brier_score_loss(y_true, y_prob)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_scores.append(brier_score_loss(y_true[idx], y_prob[idx]))
    ci_low, ci_high = np.percentile(boot_scores, [2.5, 97.5])
    return {'brier': float(brier_point),
            'ci_low': float(ci_low), 'ci_high': float(ci_high)}


# ============================================================
# 3. 校准曲线（单模型，图上标注 Brier）
# ============================================================
def plot_calibration_curve(y_true, y_prob, dataset_name,
                           n_bins=10, save_path=None):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='quantile')
    res = brier_with_bootstrap_ci(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(mean_pred, frac_pos, 's-', color='#1f77b4', linewidth=2,
            markersize=8, label=f'{dataset_name} model')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration Curve - {dataset_name}')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    ax.text(0.98, 0.02,
            f"Brier = {res['brier']:.4f}\n95% CI [{res['ci_low']:.4f}, {res['ci_high']:.4f}]",
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.close()
    return res


# ============================================================
# 4. 多模型校准曲线对比
# ============================================================
def plot_multi_model_calibration(prob_dict, y_true, dataset_name,
                                 save_path=None, n_bins=10):
    y_true = np.asarray(y_true).ravel()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')

    rows = []
    for name, prob in prob_dict.items():
        prob = np.asarray(prob).ravel()
        frac, mp = calibration_curve(y_true, prob, n_bins=n_bins, strategy='quantile')
        b = brier_score_loss(y_true, prob)
        ax.plot(mp, frac, 'o-', linewidth=1.8, markersize=6,
                label=f'{name} (Brier={b:.3f})')
        rows.append({'Model': name, 'Brier': b})

    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Calibration Curves - {dataset_name}')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  已保存: {save_path}")
    plt.close()
    return pd.DataFrame(rows)


# ============================================================
# 5. 阈值敏感性分析（论文 3.3）
# ============================================================
def threshold_sensitivity_analysis(y_true, y_prob, current_threshold,
                                   dataset_name, output_dir,
                                   thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.71, 0.05)

    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        miss = fn / (tp + fn) if (tp + fn) > 0 else 0
        f2 = 5 * prec * recall / (4 * prec + recall) if (prec + recall) > 0 else 0
        n = len(y_true)
        nb = max(0, tp / n - fp / n * (t / (1 - t))) if t < 1 else 0
        rows.append({'Threshold': t, 'Recall': recall, 'Specificity': spec,
                     'Miss_Rate': miss, 'Precision': prec, 'F2_Score': f2,
                     'Net_Benefit': nb})

    df = pd.DataFrame(rows)
    best_f2_idx = df['F2_Score'].idxmax()
    best_nb_idx = df['Net_Benefit'].idxmax()
    print(f"\n【{dataset_name} 阈值敏感性】")
    print(f"  F2最优阈值: {df.loc[best_f2_idx, 'Threshold']:.2f} "
          f"(F2={df.loc[best_f2_idx, 'F2_Score']:.4f})")
    print(f"  净获益最优阈值: {df.loc[best_nb_idx, 'Threshold']:.2f} "
          f"(NB={df.loc[best_nb_idx, 'Net_Benefit']:.4f})")
    print(f"  当前阈值: {current_threshold:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"threshold_sensitivity_{dataset_name}.csv"),
              index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.plot(df['Threshold'], df['Recall'], 'b-o', linewidth=2, label='Recall')
    ax1.plot(df['Threshold'], df['Specificity'], 'g-s', linewidth=2, label='Specificity')
    ax1.plot(df['Threshold'], df['Miss_Rate'], 'r-^', linewidth=2, label='Miss Rate')
    ax1.axvline(x=current_threshold, color='purple', linestyle='--', linewidth=2,
                label=f'Current={current_threshold:.2f}')
    ax1.set_xlabel('Threshold'); ax1.set_ylabel('Value')
    ax1.set_title(f'{dataset_name}: Performance vs Threshold')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 1.05)

    ax2 = axes[1]
    ax2.plot(df['Threshold'], df['F2_Score'], 'purple', marker='o', linewidth=2, label='F2 Score')
    ax2.plot(df['Threshold'], df['Net_Benefit'], 'orange', marker='s', linewidth=2, label='Net Benefit')
    ax2.axvline(x=current_threshold, color='purple', linestyle='--', linewidth=2,
                label=f'Current={current_threshold:.2f}')
    ax2.set_xlabel('Threshold'); ax2.set_ylabel('Score')
    ax2.set_title(f'{dataset_name}: F2 Score & Net Benefit')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"threshold_sensitivity_{dataset_name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {fig_path}")
    return df


# ============================================================
# 6. 推理耗时测试（论文 3.7 计算效率）
# ============================================================
def benchmark_inference(model, X_test, dataset_name,
                        output_dir=None, n_iterations=1000, batch_size=100):
    single = X_test.iloc[0:1] if hasattr(X_test, 'iloc') else X_test[0:1]
    batch = X_test.iloc[:min(batch_size, len(X_test))] if hasattr(X_test, 'iloc') \
        else X_test[:min(batch_size, len(X_test))]

    for _ in range(10):  # 预热
        model.predict_proba(single)

    start = time.perf_counter()
    for _ in range(n_iterations):
        model.predict_proba(single)
    single_ms = (time.perf_counter() - start) / n_iterations * 1000

    start = time.perf_counter()
    model.predict_proba(batch)
    batch_ms = (time.perf_counter() - start) * 1000

    model_mb = len(pickle.dumps(model)) / (1024 * 1024)

    result = {'dataset': dataset_name,
              'single_inference_ms': round(single_ms, 3),
              'batch_inference_ms': round(batch_ms, 3),
              'batch_size': min(batch_size, len(X_test)),
              'model_size_mb': round(model_mb, 2)}
    print(f"\n【{dataset_name} 推理耗时】单样本 {single_ms:.3f} ms | "
          f"批量({result['batch_size']}) {batch_ms:.3f} ms | 模型 {model_mb:.2f} MB")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame([result]).to_csv(
            os.path.join(output_dir, f"benchmark_{dataset_name}.csv"), index=False)
    return result
