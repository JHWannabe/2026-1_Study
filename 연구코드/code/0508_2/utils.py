"""공통 평가 유틸리티 — 4개 모델 공유"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve,
)
from sklearn.calibration import calibration_curve
from scipy import stats


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'AUC'  : roc_auc_score(y_true, y_prob),
        'AUPRC': average_precision_score(y_true, y_prob),
        'Brier': brier_score_loss(y_true, y_prob),
        'Acc'  : float((y_pred == y_true).mean()),
    }


def save_eval_plots(y_true: np.ndarray, y_prob: np.ndarray, tag: str, subdir: str):
    os.makedirs(subdir, exist_ok=True)
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.38)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    ax = fig.add_subplot(gs[0])
    ax.plot(fpr, tpr, lw=2, label=f'AUC={auc_val:.3f}')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set(xlabel='FPR', ylabel='TPR', title=f'ROC Curve ({tag})')
    ax.legend()

    # Calibration plot
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    # calibration slope/intercept via linear regression
    slope, intercept, _, _, _ = stats.linregress(prob_pred, prob_true)
    ax = fig.add_subplot(gs[1])
    ax.plot(prob_pred, prob_true, 's-', lw=2, label='Model')
    x_line = np.array([0, 1])
    ax.plot(x_line, slope * x_line + intercept, 'r--', lw=1.5,
            label=f'slope={slope:.2f}, int={intercept:.2f}')
    ax.plot([0,1],[0,1],'k:',lw=1,label='Perfect')
    ax.set(xlabel='Mean predicted prob', ylabel='Fraction positive',
           title=f'Calibration ({tag})')
    ax.legend(fontsize=8)

    # Confusion matrix
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    ax = fig.add_subplot(gs[2])
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='black', fontsize=12)
    ax.set(xticks=[0,1], xticklabels=['Normal','Low'],
           yticks=[0,1], yticklabels=['Normal','Low'],
           xlabel='Predicted', ylabel='Actual',
           title=f'Confusion Matrix ({tag})')

    path = os.path.join(subdir, f'{tag}_eval.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> plot saved: {path}')

    # calibration stats 출력
    print(f'  Calibration slope={slope:.3f}  intercept={intercept:.3f}')


def save_report(cv_metrics: list, test_m: dict, cr_text: str,
                model_name: str, description: str,
                target: str, subdir: str):
    os.makedirs(subdir, exist_ok=True)
    keys = ['AUC', 'AUPRC', 'Brier', 'Acc']
    lines = [
        f'# {model_name} Report -- {target}',
        '',
        f'**{description}**',
        '',
        '## 5-Fold CV (Train 80%)',
        '',
        '| Fold | AUC | AUPRC | Brier | Acc |',
        '|---|---|---|---|---|',
    ]
    for i, m in enumerate(cv_metrics, 1):
        lines.append(f'| {i} | {m["AUC"]:.4f} | {m["AUPRC"]:.4f} '
                     f'| {m["Brier"]:.4f} | {m["Acc"]:.4f} |')
    means = {k: np.mean([m[k] for m in cv_metrics]) for k in keys}
    stds  = {k: np.std( [m[k] for m in cv_metrics]) for k in keys}
    lines += [
        f'| Mean | {means["AUC"]:.4f} | {means["AUPRC"]:.4f} '
        f'| {means["Brier"]:.4f} | {means["Acc"]:.4f} |',
        f'| Std  | {stds["AUC"]:.4f}  | {stds["AUPRC"]:.4f}  '
        f'| {stds["Brier"]:.4f}  | {stds["Acc"]:.4f}  |',
        '',
        '## Test Set (20%)',
        '',
        '| AUC | AUPRC | Brier | Acc |',
        '|---|---|---|---|',
        f'| **{test_m["AUC"]:.4f}** | **{test_m["AUPRC"]:.4f}** '
        f'| **{test_m["Brier"]:.4f}** | **{test_m["Acc"]:.4f}** |',
        '',
        '## Classification Report',
        '',
        '```',
        cr_text.strip(),
        '```',
    ]
    path = os.path.join(subdir, f'{model_name.replace(" ","_")}_Report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  -> report saved: {path}')
