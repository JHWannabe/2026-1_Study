import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
mpl.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, f1_score,
    average_precision_score, brier_score_loss,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from config import N_FOLDS, PARENT_DIR, N_SEGMENTS, N_STATS
FOLD_COLORS = mpl.colormaps["tab10"](np.linspace(0, 0.45, N_FOLDS))

def plot_roc_curves(roc_folds, out_dir, norm=""):
    fig, ax = plt.subplots(figsize=(7, 6))
    norm_tag = f" · {norm}" if norm else ""
    fig.suptitle(f"ROC Curves — {N_FOLDS}-Fold CV  [SegmentFusion{norm_tag}]",
                 fontsize=13, fontweight="bold")
    for i, d in enumerate(roc_folds):
        ax.plot(d["fpr"], d["tpr"], color=FOLD_COLORS[i], alpha=0.7,
                label=f"Fold {i+1} (AUC={d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/cv_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_metric_distribution(cv_results, out_dir, norm=""):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    norm_tag = f" · {norm}" if norm else ""
    fig.suptitle(f"{N_FOLDS}-Fold CV Metric Distribution  [SegmentFusion{norm_tag}]",
                 fontsize=13, fontweight="bold")
    for ax, (mname, mkey) in zip(axes, [("AUC-ROC", "auc"), ("Accuracy", "acc"), ("F1-Score", "f1")]):
        vals = [m[mkey] for m in cv_results]
        bp = ax.boxplot([vals], labels=["SegmentFusion"], patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue"); patch.set_alpha(0.6)
        margin = max((max(vals) - min(vals)) * 0.4, 0.05)
        ax.set_ylim(max(0.0, min(vals) - margin), min(1.0, max(vals) + margin))
        ax.set_title(mname); ax.set_ylabel(mname)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/cv_metric_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrices(y_te, pred, sex_te, out_dir):
    sexes  = [s for s in ["M", "F"] if (sex_te == s).any()]
    n_cols = 1 + len(sexes)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices — Test Set  [SegmentFusion]",
                 fontsize=13, fontweight="bold")
    combos = [
        ("Overall", y_te, pred),
        *[(f"{'Male' if s == 'M' else 'Female'} (n={(sex_te == s).sum()})",
           y_te[sex_te == s], pred[sex_te == s]) for s in sexes],
    ]
    for ax, (title, yt, yp) in zip(axes, combos):
        sns.heatmap(confusion_matrix(yt, yp), annot=True, fmt="d", cmap="Blues", ax=ax,
                    annot_kws={"size": 13},
                    xticklabels=["Normal", "Sarco"],
                    yticklabels=["Normal", "Sarco"], cbar=False)
        ax.set_title(title); ax.set_ylabel("True"); ax.set_xlabel("Pred")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_test_roc(y_te, prob, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Test Set ROC Curves  [SegmentFusion]", fontsize=13, fontweight="bold")
    fpr, tpr, _ = roc_curve(y_te, prob)
    ax.plot(fpr, tpr, color="steelblue", linewidth=2,
            label=f"SegmentFusion (AUC={roc_auc_score(y_te, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/test_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_test_roc_by_sex(y_te, prob, sex_te, out_dir):
    sex_colors = {"M": "steelblue", "F": "tomato"}
    sex_labels = {"M": "Male", "F": "Female"}
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Test Set ROC Curves by Sex  [SegmentFusion]",
                 fontsize=13, fontweight="bold")
    for s, col in sex_colors.items():
        mask = sex_te == s
        if not mask.any():
            continue
        yt, ypr = y_te[mask], prob[mask]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, ypr)
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{sex_labels[s]} (n={mask.sum()}, AUC={roc_auc_score(yt, ypr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/test_roc_by_sex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_data_distribution(X_cv, y_cv, sex_cv, X_te, y_te, sex_te, out_dir):
    cls_colors = {"Normal": "steelblue", "Sarco": "tomato"}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
    fig.suptitle("Dataset Distribution — Train (CV) vs Test", fontsize=13, fontweight="bold")

    for row, (X, y, sex, name) in enumerate([
        (X_cv, y_cv, sex_cv, "Train (CV)"),
        (X_te, y_te, sex_te, "Test"),
    ]):
        sex_list = [s for s in ["M", "F"] if (sex == s).any()]
        x_pos    = np.arange(len(sex_list))
        ax = axes[row, 0]
        for i, (cls, col) in enumerate(cls_colors.items()):
            props = [(y[sex == s] == i).sum() / (sex == s).sum() for s in sex_list]
            ax.bar(x_pos + (i - 0.5) * 0.3, props, 0.3, label=cls, color=col, alpha=0.7)
            for j, p in enumerate(props):
                ax.text(x_pos[j] + (i - 0.5) * 0.3, p + 0.01, f"{p:.1%}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{'Male' if s == 'M' else 'Female'}\n(n={(sex == s).sum()})"
                            for s in sex_list])
        ax.set_xlim(-0.7, len(sex_list) - 0.3); ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(f"{name} — Class by Sex  (n={len(y)})"); ax.legend()

        for ax, feat_col, feat_name in [(axes[row, 1], 0, "Age"), (axes[row, 2], 2, "BMI")]:
            bins = np.linspace(X[:, feat_col].min(), X[:, feat_col].max(), 16)
            for i, (cls, col) in enumerate(cls_colors.items()):
                vals = X[y == i, feat_col]
                w    = np.ones(max(len(vals), 1)) / max(len(vals), 1)
                ax.hist(vals, bins=bins, weights=w, alpha=0.6, color=col,
                        label=f"{cls} (n={len(vals)}, {len(vals)/len(y):.0%})")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1%}"))
            ax.set_title(f"{name} — {feat_name}  (n={len(y)})")
            ax.set_xlabel(feat_name); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/data_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_calibration(entries, out_path, n_bins=10):
    """Calibration plot + Precision-Recall curve. entries: [(label, y_true, y_prob, color)]"""
    fig, (ax_cal, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Calibration & Precision-Recall Curves — Test Set",
                 fontsize=13, fontweight="bold")

    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    for label, y_true, y_prob, color in entries:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        ax_cal.plot(mean_pred, frac_pos, marker="o", color=color, linewidth=2,
                    label=f"{label}  (Brier={brier_score_loss(y_true, y_prob):.4f})")
    ax_cal.set_xlabel("Mean Predicted Probability"); ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.set_title("Calibration Plot"); ax_cal.legend(fontsize=9)

    for label, y_true, y_prob, color in entries:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax_pr.plot(rec, prec, color=color, linewidth=2,
                   label=f"{label}  (AUPRC={average_precision_score(y_true, y_prob):.4f})")
    if entries:
        baseline = entries[0][1].mean()
        ax_pr.axhline(baseline, color="gray", linestyle="--", alpha=0.5,
                      label=f"Baseline (prevalence={baseline:.3f})")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve"); ax_pr.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Markdown 보고서 ───────────────────────────────────────────
def save_report_md(cv_results, X_cv, y_cv, sex_cv, X_te, y_te, pred, prob,
                   sex_te, out_dir, ci_dict=None, norm=""):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    # dist_table
    dist_lines = ["| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |",
                  "|-------|-----|--:|-------:|---------:|------:|--------:|"]
    for split_name, y, sex in [("Train", y_cv, sex_cv), ("Test", y_te, sex_te)]:
        for s in ["M", "F"]:
            mask = sex == s
            if not mask.any():
                continue
            ys = y[mask]; n = len(ys); sarco = int(ys.sum()); normal = n - sarco
            dist_lines.append(f"| {split_name} | {s} | {n} | {normal} | {normal/n*100:.1f}%"
                               f" | {sarco} | {sarco/n*100:.1f}% |")
        n = len(y); sarco = int(y.sum()); normal = n - sarco
        dist_lines.append(f"| {split_name} | **All** | **{n}** | **{normal}** | **{normal/n*100:.1f}%**"
                           f" | **{sarco}** | **{sarco/n*100:.1f}%** |")
    dist_table = "\n".join(dist_lines)

    # feature_table
    feat_blocks = []
    for feat_name, col in [("Age", 0), ("BMI", 2)]:
        lines = [f"### {feat_name}", "",
                 "| Split | Sex | n | Mean ± Std | Min | Median | Max |",
                 "|-------|-----|--:|----------:|----:|-------:|----:|"]
        for split_name, X, sex in [("Train", X_cv, sex_cv), ("Test", X_te, sex_te)]:
            for s in ["M", "F"]:
                mask = sex == s
                if mask.any():
                    v = X[mask, col]
                    lines.append(f"| {split_name} | {s} | {len(v)} |"
                                  f" {v.mean():.2f} ± {v.std():.2f} |"
                                  f" {v.min():.2f} | {np.median(v):.2f} | {v.max():.2f} |")
            v = X[:, col]
            lines.append(f"| {split_name} | **All** | **{len(v)}** |"
                          f" **{v.mean():.2f} ± {v.std():.2f}** |"
                          f" **{v.min():.2f}** | **{np.median(v):.2f}** | **{v.max():.2f}** |")
        feat_blocks.append("\n".join(lines))
    feature_table = "\n\n".join(feat_blocks)

    # cv_table
    cv_lines = ["| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
                "|------|--------:|------:|------:|---------:|---:|"]
    for m in cv_results:
        cv_lines.append(f"| {m['fold']} | {m['auc']:.4f} | {m['auprc']:.4f}"
                        f" | {m['brier']:.4f} | {m['acc']:.4f} | {m['f1']:.4f} |")
    keys = ["auc", "auprc", "brier", "acc", "f1"]
    means = [np.mean([m[k] for m in cv_results]) for k in keys]
    stds  = [np.std( [m[k] for m in cv_results]) for k in keys]
    cv_lines.append("| **Mean** | " + " | ".join(f"**{v:.4f}**" for v in means) + " |")
    cv_lines.append("| **±Std** | " + " | ".join(f"{v:.4f}" for v in stds)   + " |")
    cv_table = "\n".join(cv_lines)

    # sex_rows
    sex_row_list = []
    for s in ["M", "F"]:
        mask = sex_te == s
        if not mask.sum():
            continue
        yt, yp, ypr = y_te[mask], pred[mask], prob[mask]
        has_both = len(np.unique(yt)) > 1
        auc   = roc_auc_score(yt, ypr)           if has_both else float("nan")
        auprc = average_precision_score(yt, ypr) if has_both else float("nan")
        brier = brier_score_loss(yt, ypr)        if has_both else float("nan")
        sex_row_list.append(f"| {s} | {mask.sum()} | {auc:.4f} | {auprc:.4f} | {brier:.4f}"
                            f" | {accuracy_score(yt, yp):.4f} | {f1_score(yt, yp, zero_division=0):.4f} |")
    sex_rows = "\n".join(sex_row_list)

    ci_rows = ""
    if ci_dict:
        for label, key in [("AUC-ROC", "auc"), ("AUPRC", "auprc"),
                            ("Brier", "brier"), ("Accuracy", "acc"), ("F1", "f1")]:
            if key in ci_dict:
                est, lo, hi = ci_dict[key]
                ci_rows += f"| {label} | {est:.4f} | {lo:.4f} | {hi:.4f} |\n"

    norm_tag = f" · {norm}" if norm else ""
    md = f"""# SMI Binary Classification — Results

Generated: {now}  |  {N_FOLDS}-Fold CV  |  SegmentFusion (AEC 8-Segment + Clinic{norm_tag})

---

## 0. Dataset Distribution

### Class Distribution

{dist_table}

{feature_table}

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

{cv_table}

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| SegmentFusion | {roc_auc_score(y_te, prob):.4f} | {average_precision_score(y_te, prob):.4f} | {brier_score_loss(y_te, prob):.4f} | {accuracy_score(y_te, pred):.4f} | {f1_score(y_te, pred, zero_division=0):.4f} |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
{sex_rows}

---

## 3. Confusion Matrix

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | {tn} | {fp} |
| **True: Sarco**  | {fn} | {tp} |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
{ci_rows}
---

## 5. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves |
| `cv_metric_distribution.png` | AUC / Acc / F1 boxplot across folds |
| `confusion_matrices.png` | Test-set confusion matrices (overall + by sex) |
| `test_roc_curves.png` | Test-set ROC curve (overall) |
| `test_roc_by_sex.png` | Test-set ROC curves by sex |
| `calibration.png` | Calibration plot + Precision-Recall curve |
"""
    md_path = f"{out_dir}/results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  {md_path}")

def plot_norm_comparison(results, out_path):
    """global / rowwise / raw 3가지 normalization 비교 (ROC, PR, Calibration, Metrics)."""
    norm_colors = {"global": "steelblue", "rowwise": "tomato", "raw": "seagreen"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("AEC Normalization Comparison — Test Set", fontsize=14, fontweight="bold")
    ax_roc, ax_pr, ax_cal, ax_bar = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")

    metric_keys  = ["AUC-ROC", "AUPRC", "F1", "Accuracy"]
    metric_vals  = {m: [] for m in metric_keys}
    norm_names   = list(results.keys())

    for norm, res in results.items():
        y_te  = res["y_te"]
        prob  = res["prob"]
        thresh = res.get("threshold", 0.5)
        pred  = (prob >= thresh).astype(int)
        col   = norm_colors.get(norm, "gray")

        fpr, tpr, _ = roc_curve(y_te, prob)
        auc = roc_auc_score(y_te, prob)
        ax_roc.plot(fpr, tpr, color=col, linewidth=2, label=f"{norm} (AUC={auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_te, prob)
        auprc = average_precision_score(y_te, prob)
        ax_pr.plot(rec, prec, color=col, linewidth=2, label=f"{norm} (AUPRC={auprc:.3f})")

        frac_pos, mean_pred = calibration_curve(y_te, prob, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_te, prob)
        ax_cal.plot(mean_pred, frac_pos, marker="o", color=col, linewidth=2,
                    label=f"{norm} (Brier={brier:.4f})")

        metric_vals["AUC-ROC"].append(auc)
        metric_vals["AUPRC"].append(auprc)
        metric_vals["F1"].append(f1_score(y_te, pred, zero_division=0))
        metric_vals["Accuracy"].append(accuracy_score(y_te, pred))

    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curve"); ax_roc.legend(fontsize=9)

    baseline = list(results.values())[0]["y_te"].mean()
    ax_pr.axhline(baseline, color="gray", linestyle="--", alpha=0.5,
                  label=f"Baseline (prev={baseline:.3f})")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve"); ax_pr.legend(fontsize=9)

    ax_cal.set_xlabel("Mean Predicted Probability"); ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.set_title("Calibration Plot"); ax_cal.legend(fontsize=9)

    x     = np.arange(len(metric_keys))
    width = 0.25
    for i, norm in enumerate(norm_names):
        col  = norm_colors.get(norm, "gray")
        vals = [metric_vals[m][i] for m in metric_keys]
        bars = ax_bar.bar(x + (i - 1) * width, vals, width,
                          label=norm, color=col, alpha=0.75)
        for bar, v in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metric_keys)
    ax_bar.set_ylim(0, 1.15); ax_bar.set_title("Metric Comparison"); ax_bar.legend(fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aec_heatmap(X_aec_raw, y_te, prob, out_dir):
    """Test set raw AEC 128-dim 히트맵 (predicted prob 기준 정렬)."""
    sort_idx = np.argsort(prob)
    X_s = X_aec_raw[sort_idx]
    y_s = y_te[sort_idx]
    prob_s = prob[sort_idx]
    n = len(sort_idx)

    fig, ax = plt.subplots(figsize=(13, max(6, n * 0.12)))
    im = ax.imshow(X_s, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='AEC value', shrink=0.6)

    boundary = int((prob_s < 0.5).sum())
    if 0 < boundary < n:
        ax.axhline(boundary - 0.5, color='gold', linewidth=2,
                   linestyle='--', label='p=0.5 boundary')
        ax.legend(fontsize=8, loc='upper right')

    ytick_pos = np.linspace(0, n - 1, min(15, n)).astype(int)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(
        [f"p={prob_s[i]:.2f}  y={y_s[i]}" for i in ytick_pos], fontsize=7
    )
    ax.set_xlabel('AEC Dimension (1–128)')
    ax.set_ylabel('Patient (sorted by predicted probability ↑)')
    ax.set_title('AEC Raw Signal Heatmap — Test Set')

    fig.tight_layout()
    fig.savefig(f"{out_dir}/aec_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_dir}/aec_heatmap.png")


def plot_cam_segment8(saliency, y_te, prob, out_dir):
    """model_aec_segment_8 gradient saliency 시각화.

    saliency: (N, 60) — 60 = N_SEGMENTS×N_STATS intra + (N_SEGMENTS-1)×N_STATS inter
    클래스별 평균 중요도 바차트 + 샘플별 히트맵을 저장.
    """
    n_seg = N_SEGMENTS
    n_st  = N_STATS
    sort_idx = np.argsort(prob)
    n = len(sort_idx)

    # intra: (N, n_seg) — 4 stats per segment 평균
    intra = saliency[:, :n_seg * n_st].reshape(-1, n_seg, n_st).mean(-1)
    # inter: (N, n_seg-1)
    inter = saliency[:, n_seg * n_st:].reshape(-1, n_seg - 1, n_st).mean(-1)

    seg_labels   = [f"S{i+1}" for i in range(n_seg)]
    inter_labels = [f"S{i+1}→{i+2}" for i in range(n_seg - 1)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Gradient Saliency — AEC 8-Segment MLP — Test Set',
                 fontsize=13, fontweight='bold')

    width = 0.35
    class_spec = [(0, 'steelblue', 'Normal'), (1, 'tomato', 'Sarcopenia')]

    # Top-left: intra-segment importance by class
    ax = axes[0, 0]
    x = np.arange(n_seg)
    for j, (label, col, name) in enumerate(class_spec):
        mask = y_te == label
        if mask.any():
            ax.bar(x + (j - 0.5) * width, intra[mask].mean(0), width,
                   label=f"{name} (n={mask.sum()})", color=col, alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels(seg_labels, fontsize=9)
    ax.set_ylabel('Mean |gradient|')
    ax.set_title('Intra-Segment Importance by Class')
    ax.legend(fontsize=9)

    # Top-right: inter-segment importance by class
    ax2 = axes[0, 1]
    x2 = np.arange(n_seg - 1)
    for j, (label, col, name) in enumerate(class_spec):
        mask = y_te == label
        if mask.any():
            ax2.bar(x2 + (j - 0.5) * width, inter[mask].mean(0), width,
                    label=f"{name} (n={mask.sum()})", color=col, alpha=0.75)
    ax2.set_xticks(x2); ax2.set_xticklabels(inter_labels, fontsize=8, rotation=20)
    ax2.set_ylabel('Mean |gradient|')
    ax2.set_title('Inter-Segment Importance by Class')
    ax2.legend(fontsize=9)

    # Bottom-left: intra saliency heatmap sorted by prob
    ax3 = axes[1, 0]
    im3 = ax3.imshow(intra[sort_idx], aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im3, ax=ax3, label='Saliency', shrink=0.8)
    boundary = int((prob[sort_idx] < 0.5).sum())
    if 0 < boundary < n:
        ax3.axhline(boundary - 0.5, color='blue', linewidth=2, linestyle='--')
    ax3.set_xticks(np.arange(n_seg)); ax3.set_xticklabels(seg_labels, fontsize=8)
    ax3.set_ylabel('Patient (sorted by prob ↑)')
    ax3.set_title('Intra-Segment Saliency Heatmap')

    # Bottom-right: inter saliency heatmap sorted by prob
    ax4 = axes[1, 1]
    im4 = ax4.imshow(inter[sort_idx], aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im4, ax=ax4, label='Saliency', shrink=0.8)
    if 0 < boundary < n:
        ax4.axhline(boundary - 0.5, color='blue', linewidth=2, linestyle='--')
    ax4.set_xticks(np.arange(n_seg - 1)); ax4.set_xticklabels(inter_labels, fontsize=8, rotation=20)
    ax4.set_ylabel('Patient (sorted by prob ↑)')
    ax4.set_title('Inter-Segment Saliency Heatmap')

    fig.tight_layout()
    fig.savefig(f"{out_dir}/cam.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {out_dir}/cam.png")


# ── Direction 3: 성별 AEC 기여도 비교 ─────────────────────────────────────────

def plot_cam_by_sex(saliency, y_te, prob, sex_te, out_dir):
    """성별별 Gradient Saliency 비교 시각화. (Direction 3)

    saliency: (N, 60) — compute_gradient_saliency() 출력
    남성/여성별 AEC 구간 중요도 차이를 바 차트 + 차이 그래프로 저장.
    """
    n_seg = N_SEGMENTS
    n_st  = N_STATS
    sort_idx = np.argsort(prob)

    intra = saliency[:, :n_seg * n_st].reshape(-1, n_seg, n_st).mean(-1)   # (N, n_seg)
    inter = saliency[:, n_seg * n_st:].reshape(-1, n_seg - 1, n_st).mean(-1)

    seg_labels   = [f"S{i+1}" for i in range(n_seg)]
    inter_labels = [f"S{i+1}-{i+2}" for i in range(n_seg - 1)]
    sex_spec = [("M", "steelblue", "Male"), ("F", "tomato", "Female")]
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Gradient Saliency by Sex — AEC 8-Segment MLP — Test Set",
                 fontsize=13, fontweight="bold")

    # 상단: 성별별 intra / inter 중요도
    for ax, labels, data in [
        (axes[0, 0], seg_labels,   intra),
        (axes[0, 1], inter_labels, inter),
    ]:
        x = np.arange(len(labels))
        for j, (s, col, name) in enumerate(sex_spec):
            mask = sex_te == s
            if mask.any():
                ax.bar(x + (j - 0.5) * width, data[mask].mean(0), width,
                       label=f"{name} (n={mask.sum()})", color=col, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=20 if len(labels) > 6 else 0)
        ax.set_ylabel("Mean |gradient|")
        tag = "Intra" if labels is seg_labels else "Inter"
        ax.set_title(f"{tag}-Segment Importance by Sex")
        ax.legend(fontsize=9)

    # 하단: 남성 − 여성 차이 (양수=남성 우세, 음수=여성 우세)
    m_mask = sex_te == "M"
    f_mask = sex_te == "F"
    for ax, labels, data in [
        (axes[1, 0], seg_labels,   intra),
        (axes[1, 1], inter_labels, inter),
    ]:
        x = np.arange(len(labels))
        if m_mask.any() and f_mask.any():
            diff = data[m_mask].mean(0) - data[f_mask].mean(0)
            colors = ["steelblue" if d > 0 else "tomato" for d in diff]
            ax.bar(x, diff, color=colors, alpha=0.75)
            ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=20 if len(labels) > 6 else 0)
        ax.set_ylabel("Male − Female |gradient|")
        tag = "Intra" if labels is seg_labels else "Inter"
        ax.set_title(f"{tag} Importance Difference (Male − Female)\n(blue=Male↑, red=Female↑)")

    fig.tight_layout()
    fig.savefig(f"{out_dir}/cam_by_sex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_dir}/cam_by_sex.png")


# ── Direction 5: 80번째 슬라이스 교차 패턴 분석 ───────────────────────────────

def plot_aec_crossing_pattern(X_aec_raw, y_te, sex_te, out_dir, cross_idx=79):
    """남성/여성 평균 AEC 교차 패턴 분석. (Direction 5)

    Row-normalized AEC 기준으로 Normal/Sarcopenia 그룹별 성별 교차 패턴 비교.
    2행 × 2열:
      [0,0] Normal     — Female vs Male 평균 row-norm AEC 곡선 (±std)
      [0,1] Normal     — Female − Male 차이 바 차트
      [1,0] Sarcopenia — Female vs Male 평균 row-norm AEC 곡선 (±std)
      [1,1] Sarcopenia — Female − Male 차이 바 차트
    cross_idx=79: 80번째 슬라이스(0-indexed) 교차 가설선.
    x축 방향: aec_1=pubis(caudal) → aec_128=liver_upper(cranial)
    """
    # 각 환자 row를 자신의 합으로 나눠 상대적 AEC 분포만 비교
    row_sum = X_aec_raw.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1, row_sum)
    X = X_aec_raw / row_sum

    x = np.arange(1, X.shape[1] + 1)
    class_spec = [(0, "Normal"), (1, "Sarcopenia")]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "AEC Crossing Pattern by Sex — Row-Normalized\n"
        f"(가설 교차 기점: Slice {cross_idx + 1},  gold 점선 / 실제 교차 gray 점선)",
        fontsize=13, fontweight="bold"
    )

    for row, (cls_label, cls_name) in enumerate(class_spec):
        cls_mask = (y_te == cls_label)
        f_mask = cls_mask & (sex_te == "F")
        m_mask = cls_mask & (sex_te == "M")

        f_mean = X[f_mask].mean(0);  f_std = X[f_mask].std(0)
        m_mean = X[m_mask].mean(0);  m_std = X[m_mask].std(0)
        diff    = f_mean - m_mean
        changes = np.where(np.diff(np.sign(diff)))[0]

        # ── 왼쪽: Female vs Male 평균 row-norm AEC 곡선 ─────────────────
        ax = axes[row, 0]
        ax.plot(x, f_mean, color="steelblue", linewidth=2,
                label=f"Female (n={int(f_mask.sum())})")
        ax.fill_between(x, f_mean - f_std, f_mean + f_std,
                        alpha=0.15, color="steelblue")
        ax.plot(x, m_mean, color="tomato", linewidth=2,
                label=f"Male (n={int(m_mask.sum())})")
        ax.fill_between(x, m_mean - m_std, m_mean + m_std,
                        alpha=0.15, color="tomato")
        ax.axvline(cross_idx + 1, color="gold", linewidth=2, linestyle="--",
                   label=f"Slice {cross_idx + 1} (가설)")
        for sc in changes:
            ax.axvline(sc + 1.5, color="gray", linewidth=1, linestyle=":", alpha=0.7)
        ax.set_xlabel("AEC Dimension  [pubis(1) → liver_upper(128)]")
        ax.set_ylabel("Row-Norm AEC")
        ax.set_title(f"{cls_name} — Mean Row-Norm AEC\n(실제 교차 {len(changes)}건)")
        ax.legend(fontsize=9)

        # ── 오른쪽: Female − Male 차이 바 차트 ─────────────────────────
        ax2 = axes[row, 1]
        bar_colors = ["steelblue" if d > 0 else "tomato" for d in diff]
        ax2.bar(x, diff, color=bar_colors, alpha=0.75, width=0.9)
        ax2.axhline(0, color="black", linewidth=1)
        ax2.axvline(cross_idx + 1, color="gold", linewidth=2, linestyle="--",
                    label=f"Slice {cross_idx + 1}")
        for sc in changes:
            ax2.axvline(sc + 1.5, color="gray", linewidth=1, linestyle=":", alpha=0.7)
        annotation_offset = np.abs(diff).max() * 0.7 if diff.any() else 0.01
        for sc in changes[:3]:
            ax2.annotate(f"↕{sc+1}", xy=(sc + 1.5, 0),
                         xytext=(sc + 1.5, annotation_offset),
                         fontsize=7, ha="center", color="dimgray",
                         arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
        ax2.set_xlabel("AEC Dimension  [pubis(1) → liver_upper(128)]")
        ax2.set_ylabel("Female − Male Row-Norm AEC")
        ax2.set_title(f"{cls_name} — AEC Difference: Female − Male\n(blue=Female↑  red=Male↑)")
        ax2.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"{out_dir}/aec_crossing_pattern.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


def plot_training_curves(fold_histories, out_dir, model_name="SegmentFusion", norm=""):
    """CV fold별 train/val loss & val AUC 커브 저장 → training_curves_cv.png"""
    norm_tag = f" · {norm}" if norm else ""
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {N_FOLDS}-Fold CV  [{model_name}{norm_tag}]",
                 fontsize=13, fontweight="bold")

    for i, h in enumerate(fold_histories):
        col = FOLD_COLORS[i]
        ep  = np.arange(1, len(h["train_loss"]) + 1)
        ax_loss.plot(ep, h["train_loss"], color=col, linestyle="--", alpha=0.6, linewidth=1.2)
        ax_loss.plot(ep, h["val_loss"],   color=col, linestyle="-",  alpha=0.9, linewidth=1.5,
                     label=f"Fold {i+1}")
        ax_auc.plot(ep, h["val_auc"], color=col, linewidth=1.5,
                    label=f"Fold {i+1} (best={max(h['val_auc']):.3f})")

    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Train Loss (--) & Val Loss (—)")
    ax_loss.legend(fontsize=8)

    ax_auc.set_xlabel("Epoch"); ax_auc.set_ylabel("Val AUC")
    ax_auc.set_ylim(0, 1.05)
    ax_auc.set_title("Validation AUC per Fold")
    ax_auc.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/training_curves_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_dir}/training_curves_cv.png")


def plot_final_training_curve(history, out_dir, model_name="SegmentFusion", norm=""):
    """최종 모델 train/val loss & val AUC 커브 저장 → training_curves_final.png"""
    norm_tag = f" · {norm}" if norm else ""
    ep = np.arange(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Final Model Training Curves  [{model_name}{norm_tag}]",
                 fontsize=13, fontweight="bold")

    ax_loss.plot(ep, history["train_loss"], color="steelblue", linestyle="--",
                 linewidth=1.5, label="Train Loss")
    ax_loss.plot(ep, history["val_loss"],   color="tomato",    linestyle="-",
                 linewidth=1.5, label="Val Loss")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Train & Val Loss"); ax_loss.legend()

    ax_auc.plot(ep, history["val_auc"], color="seagreen", linewidth=1.5, label="Val AUC")
    best_ep = int(np.argmax(history["val_auc"])) + 1
    ax_auc.axvline(best_ep, color="gray", linestyle=":", alpha=0.7,
                   label=f"Best epoch={best_ep} (AUC={max(history['val_auc']):.3f})")
    ax_auc.set_xlabel("Epoch"); ax_auc.set_ylabel("Val AUC")
    ax_auc.set_ylim(0, 1.05)
    ax_auc.set_title("Validation AUC"); ax_auc.legend()

    fig.tight_layout()
    fig.savefig(f"{out_dir}/training_curves_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_dir}/training_curves_final.png")


def save_all(roc_folds, cv_results, X_cv, y_cv, sex_cv,
             X_te, y_te, pred, prob, sex_te,
             out_dir=None, ci_dict=None, norm=""):
    """모든 시각화(7종 png)와 results.md를 out_dir에 저장."""
    out_dir = out_dir or PARENT_DIR
    plot_data_distribution(X_cv, y_cv, sex_cv, X_te, y_te, sex_te, out_dir)
    plot_roc_curves(roc_folds, out_dir, norm=norm)
    plot_metric_distribution(cv_results, out_dir, norm=norm)
    plot_confusion_matrices(y_te, pred, sex_te, out_dir)
    plot_test_roc(y_te, prob, out_dir)
    plot_test_roc_by_sex(y_te, prob, sex_te, out_dir)
    plot_calibration(
        [("SegmentFusion", y_te, prob, "steelblue")],
        out_path=f"{out_dir}/calibration.png",
    )

    save_report_md(cv_results, X_cv, y_cv, sex_cv, X_te, y_te, pred, prob,
                   sex_te, out_dir=out_dir, ci_dict=ci_dict, norm=norm)

    np.save(os.path.join(out_dir, "test_prob.npy"), prob)
    np.save(os.path.join(out_dir, "test_y.npy"), y_te)