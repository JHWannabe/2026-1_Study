"""
0520 시각화 유틸리티.

0515 visualize.py의 모든 함수를 포함하며, 아래를 추가한다:
  plot_regression_scatter   — SMI 예측 vs 실제 산점도
  plot_calibration_compare  — 캘리브레이션 전/후 비교 reliability diagram
  plot_ensemble_roc         — 앙상블 3종 ROC 비교
  plot_sex_roc_compare      — 성별 분리 모델 vs 통합 모델 ROC 비교
  save_regression_report_md — M6 결과 markdown
  save_calibration_report_md— Calibration 결과 markdown
  save_ensemble_report_md   — M7 앙상블 결과 markdown
"""
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, f1_score,
    average_precision_score, brier_score_loss,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from config import N_FOLDS, EPOCHS, RESULTS_DIR_M4, RESULTS_DIR_M7
from calibration import compute_ece

FOLD_COLORS = plt.get_cmap("tab10")(np.linspace(0, 0.45, N_FOLDS))

# 전역 저장 디렉토리
_out_dir: str = RESULTS_DIR_M4


def _save(fname, fig, out_dir=None):
    d = out_dir or _out_dir
    fig.tight_layout()
    fig.savefig(f"{d}/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 0515 공유 헬퍼 (재사용) ───────────────────────────────────

def plot_calibration(entries, out_path, n_bins=10):
    """Reliability diagram + PR 커브 (0515 원본과 동일 시그니처)."""
    fig, (ax_cal, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Calibration & Precision-Recall Curves — Test Set",
                 fontsize=13, fontweight="bold")
    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    for label, y_true, y_prob, color in entries:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        brier = brier_score_loss(y_true, y_prob)
        ax_cal.plot(mean_pred, frac_pos, marker="o", color=color, linewidth=2,
                    label=f"{label}  (Brier={brier:.4f})")
    ax_cal.set_xlabel("Mean Predicted Probability"); ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.set_title("Calibration Plot"); ax_cal.legend(fontsize=9)
    for label, y_true, y_prob, color in entries:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax_pr.plot(rec, prec, color=color, linewidth=2, label=f"{label}  (AUPRC={ap:.4f})")
    baseline = entries[-1][1].mean()
    ax_pr.axhline(baseline, color="gray", linestyle="--", alpha=0.5,
                  label=f"Baseline (prevalence={baseline:.3f})")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve"); ax_pr.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_data_distribution(X_cv, y_cv, sex_cv, X_te, y_te, sex_te, out_dir=None):
    """데이터 분포 (클래스 비율·Age·BMI)."""
    d = out_dir or _out_dir
    cls_colors = {"Normal": "steelblue", "Sarco": "tomato"}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
    fig.suptitle("Dataset Distribution — Train (CV) vs Test", fontsize=13, fontweight="bold")
    for row, (X, y, sex, name) in enumerate([
        (X_cv, y_cv, sex_cv, "Train (CV)"), (X_te, y_te, sex_te, "Test"),
    ]):
        n_split = len(y)
        ax = axes[row, 0]
        sex_list = [s for s in ["M", "F"] if (sex == s).any()]
        x_pos = np.arange(len(sex_list)); width = 0.3
        for i, (cls, col) in enumerate(cls_colors.items()):
            props = [(y[sex == s] == i).sum() / (sex == s).sum() for s in sex_list]
            ax.bar(x_pos + (i - 0.5) * width, props, width, label=cls, color=col, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{'Male' if s=='M' else 'Female'}\n(n={(sex==s).sum()})" for s in sex_list])
        ax.set_xlim(-0.7, len(sex_list) - 0.3); ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(f"{name} — Class by Sex"); ax.legend()
        for ax2, feat_col, feat_name in [(axes[row, 1], 0, "Age"), (axes[row, 2], 2, "BMI")]:
            shared_bins = np.linspace(X[:, feat_col].min(), X[:, feat_col].max(), 16)
            for i, (cls, col) in enumerate(cls_colors.items()):
                vals = X[y == i, feat_col]; n_cls = max(len(vals), 1)
                ax2.hist(vals, bins=shared_bins, weights=np.ones(n_cls)/n_cls,
                         alpha=0.6, color=col, label=f"{cls} (n={len(vals)})")
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1%}"))
            ax2.set_title(f"{name} — {feat_name}"); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{d}/data_distribution.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ── 0520 새 시각화 함수 ────────────────────────────────────────

def plot_regression_scatter(pred_smi, true_smi, sex_te, out_dir, model_label="SMI Regression"):
    """
    SMI 예측 vs 실제 산점도.
    대각선 = 완벽한 예측, 음영 = ±5 cm²/m² 오차 범위.
    성별 색상 구분.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{model_label} — Predicted vs Actual SMI", fontsize=13, fontweight="bold")

    sex_color = {"M": "steelblue", "F": "tomato"}
    sex_label = {"M": "Male", "F": "Female"}
    all_vals  = np.concatenate([pred_smi, true_smi])
    lo, hi    = all_vals.min() - 2, all_vals.max() + 2

    for ax, title, show_thresh in [(axes[0], "All", True), (axes[1], "By Sex", False)]:
        for s, col in sex_color.items():
            mask = sex_te == s
            if not mask.any(): continue
            ax.scatter(true_smi[mask], pred_smi[mask], c=col, alpha=0.5, s=20,
                       label=sex_label[s])
        # perfect prediction line
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="Perfect prediction")
        ax.fill_between([lo, hi], [lo - 5, hi - 5], [lo + 5, hi + 5],
                        color="gray", alpha=0.1, label="±5 cm²/m²")
        if show_thresh:
            ax.axhline(40.96, color="steelblue", linestyle=":", alpha=0.6, label="M threshold")
            ax.axhline(30.6,  color="tomato",    linestyle=":", alpha=0.6, label="F threshold")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual SMI (cm²/m²)"); ax.set_ylabel("Predicted SMI (cm²/m²)")
        ax.set_title(title); ax.legend(fontsize=8); ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/regression_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_compare(y_true, probs_before, probs_after,
                              label_before, label_after, out_dir, n_bins=10):
    """
    캘리브레이션 전/후 Reliability Diagram + Brier Score 비교.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Temperature Scaling Calibration", fontsize=13, fontweight="bold")

    # Reliability diagrams
    for ax, probs, label, color in [
        (axes[0], probs_before, label_before, "tomato"),
        (axes[1], probs_after,  label_after,  "seagreen"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
        ece = compute_ece(y_true, probs)
        brier = brier_score_loss(y_true, probs)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        ax.plot(mean_pred, frac_pos, "o-", color=color, linewidth=2,
                label=f"ECE={ece:.4f}\nBrier={brier:.4f}")
        ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.05, color="gray")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Confidence"); ax.set_ylabel("Fraction Positive")
        ax.set_title(f"{label}"); ax.legend(fontsize=9)

    # ECE 비교 bar
    ax3 = axes[2]
    ece_before = compute_ece(y_true, probs_before)
    ece_after  = compute_ece(y_true, probs_after)
    brier_before = brier_score_loss(y_true, probs_before)
    brier_after  = brier_score_loss(y_true, probs_after)
    categories = ["ECE", "Brier Score"]
    x = np.arange(len(categories))
    w = 0.3
    ax3.bar(x - w/2, [ece_before, brier_before], w, color="tomato",  alpha=0.7, label=label_before)
    ax3.bar(x + w/2, [ece_after,  brier_after],  w, color="seagreen",alpha=0.7, label=label_after)
    ax3.set_xticks(x); ax3.set_xticklabels(categories)
    ax3.set_ylabel("Score (lower is better)")
    ax3.set_title("Before vs After Calibration"); ax3.legend()
    for bar in ax3.patches:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/calibration_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_roc(y_te, results_dict, individual_probs, individual_labels,
                      out_dir):
    """
    앙상블 3종 + 개별 모델 ROC 비교.
    individual_probs: {label: probs_array}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Ensemble vs Individual Models — Test ROC", fontsize=13, fontweight="bold")

    linestyles = {"M1 LR": "--", "M2 CrossAttn": ":"}
    for label, probs in individual_probs.items():
        fpr, tpr, _ = roc_curve(y_te, probs)
        auc = roc_auc_score(y_te, probs)
        ls  = linestyles.get(label, "-.")
        ax.plot(fpr, tpr, linestyle=ls, linewidth=1.5, alpha=0.7,
                label=f"{label} (AUC={auc:.3f})")

    colors = {"avg": "steelblue", "weighted": "seagreen", "stacking": "tomato"}
    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(y_te, res["probs"])
        ax.plot(fpr, tpr, color=colors.get(name, "gray"), linewidth=2.5,
                label=f"Ensemble ({name}) AUC={res['auc']:.3f}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/ensemble_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sex_roc_compare(y_te_m, prob_combined_m, prob_sex_m,
                          y_te_f, prob_combined_f, prob_sex_f,
                          out_dir):
    """성별 분리 모델 vs 통합 모델 ROC 커브 비교."""
    fig, (ax_m, ax_f) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Sex-Stratified vs Combined Model — ROC Curves", fontsize=13, fontweight="bold")

    for ax, y_te, prob_comb, prob_sex, title in [
        (ax_m, y_te_m, prob_combined_m, prob_sex_m, "Male"),
        (ax_f, y_te_f, prob_combined_f, prob_sex_f, "Female"),
    ]:
        for prob, label, color, ls in [
            (prob_comb, "Combined (M2)", "steelblue", "--"),
            (prob_sex,  "Sex-Specific",  "tomato",    "-"),
        ]:
            if len(np.unique(y_te)) < 2: continue
            fpr, tpr, _ = roc_curve(y_te, prob)
            auc = roc_auc_score(y_te, prob)
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                    label=f"{label} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(f"{title} (n={len(y_te)})"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/sex_roc_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 결과 보고서 Markdown ─────────────────────────────────────

def save_regression_report_md(reg_cv, reg_stats_test, pred_smi, true_smi,
                               y_bin_te, bin_pred, out_dir):
    """M6 SMI 회귀 결과 Markdown 저장."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# SMI Regression — Results (Model 6)",
        "",
        f"Generated: {now}  |  {N_FOLDS}-Fold CV  |  Target: SMI (cm²/m²)",
        "",
        "---",
        "",
        "## Cross-Validation Summary",
        "",
        "| Fold | MAE | RMSE | R² | AUC | F1 |",
        "|------|----:|-----:|---:|----:|---:|",
    ]
    for m in reg_cv:
        lines.append(f"| {m['fold']} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.4f}"
                     f" | {m['auc']:.4f} | {m['f1']:.4f} |")
    maes  = [m["mae"]  for m in reg_cv]
    rmses = [m["rmse"] for m in reg_cv]
    r2s   = [m["r2"]   for m in reg_cv]
    aucs  = [m["auc"]  for m in reg_cv]
    f1s   = [m["f1"]   for m in reg_cv]
    lines += [
        f"| **Mean** | **{np.mean(maes):.3f}** | **{np.mean(rmses):.3f}**"
        f" | **{np.mean(r2s):.4f}** | **{np.mean(aucs):.4f}** | **{np.mean(f1s):.4f}** |",
        f"| **±Std** | {np.std(maes):.3f} | {np.std(rmses):.3f}"
        f" | {np.std(r2s):.4f} | {np.std(aucs):.4f} | {np.std(f1s):.4f} |",
        "",
        "---",
        "",
        "## Test Set Performance",
        "",
        "### Regression Metrics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| MAE (cm²/m²) | {reg_stats_test['mae']:.3f} |",
        f"| RMSE (cm²/m²) | {reg_stats_test['rmse']:.3f} |",
        f"| R² | {reg_stats_test['r2']:.4f} |",
        "",
        "### Binary Classification (after thresholding)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| AUC-ROC | {reg_stats_test['auc']:.4f} |",
        f"| F1 | {reg_stats_test['f1']:.4f} |",
        f"| Accuracy | {reg_stats_test['acc']:.4f} |",
        "",
        "### Figures",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `regression_scatter.png` | Predicted vs Actual SMI scatter plot |",
        "| `calibration.png` | Probability calibration + PR curve |",
    ]
    with open(f"{out_dir}/results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_calibration_report_md(calib_stats, out_dir):
    """캘리브레이션 결과 Markdown 저장."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    s = calib_stats
    lines = [
        "# Temperature Scaling Calibration — Results",
        "",
        f"Generated: {now}  |  Model: M2 CrossAttn",
        "",
        f"Optimal Temperature T = **{s['T']:.4f}**",
        "",
        "---",
        "",
        "## Calibration Metrics",
        "",
        "| Metric | Before | After | Δ |",
        "|--------|-------:|------:|--:|",
        f"| ECE | {s['ece_before']:.4f} | {s['ece_after']:.4f}"
        f" | {s['ece_after']-s['ece_before']:+.4f} |",
        f"| Brier | {s['brier_before']:.4f} | {s['brier_after']:.4f}"
        f" | {s['brier_after']-s['brier_before']:+.4f} |",
        "",
        "> 음수 Δ = 개선. T > 1이면 over-confident 교정(softening).",
        "",
        "## Figures",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `calibration_compare.png` | Reliability diagram 전/후 + ECE bar |",
    ]
    with open(f"{out_dir}/results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_ensemble_report_md(ensemble_results, individual_aucs, y_te, out_dir):
    """앙상블 결과 Markdown 저장."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Ensemble — Results (Model 7)",
        "",
        f"Generated: {now}",
        "",
        "---",
        "",
        "## Individual Models",
        "",
        "| Model | AUC-ROC |",
        "|-------|--------:|",
    ]
    for name, auc in individual_aucs.items():
        lines.append(f"| {name} | {auc:.4f} |")
    lines += [
        "",
        "## Ensemble Results",
        "",
        "| Method | AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
        "|--------|--------:|------:|------:|---------:|---:|",
    ]
    for name, res in ensemble_results.items():
        lines.append(f"| {name} | {res['auc']:.4f} | {res['auprc']:.4f}"
                     f" | {res['brier']:.4f} | {res['acc']:.4f} | {res['f1']:.4f} |")
    if "stacking" in ensemble_results:
        coef = ensemble_results["stacking"].get("meta_coef", [])
        if coef:
            lines += ["", f"> Stacking meta-LR coefficients: M1={coef[0]:.3f}, M2={coef[1]:.3f}"]
    lines += [
        "",
        "## Figures",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `ensemble_roc.png` | Ensemble vs individual model ROC curves |",
    ]
    with open(f"{out_dir}/results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
