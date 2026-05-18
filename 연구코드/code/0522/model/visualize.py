"""
시각화 및 Markdown 보고서 생성 유틸리티.

Model 1 (Clinic Only, LR):
  save_all()       — 7종 PNG + results.md 를 out_dir에 일괄 저장

Model 2/2_2/3 (Clinic + AEC / Scanner):
  save_all_cross() — 8종 PNG + results.md 를 out_dir에 일괄 저장

출력 경로 관리:
  _dir1, _dir2 전역 변수를 save_all/save_all_cross 호출 시 갱신한다.
  ProcessPoolExecutor 환경에서는 프로세스마다 전역 상태가 독립적으로 유지된다.

보고서 함수 (내부):
  _dist_table, _feature_table  — 데이터 분포 Markdown 테이블
  _cv_table, _sex_rows, _cm_block — CV/성별/혼동행렬 Markdown 블록
  save_report_md, _save_report_md_cross — results.md 생성
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

from config import N_FOLDS, EPOCHS, RESULTS_DIR, RESULTS_DIR_CROSS

FOLD_COLORS = plt.get_cmap("tab10")(np.linspace(0, 0.45, N_FOLDS))

# 현재 저장 디렉토리 (save_all / save_all_cross 호출 시 갱신)
_dir1: str = RESULTS_DIR
_dir2: str = RESULTS_DIR_CROSS


def plot_roc_curves(lr_roc_folds):
    """LR의 fold별 ROC 커브를 그려 cv_roc_curves.png로 저장."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"ROC Curves — {N_FOLDS}-Fold CV  [Logistic Regression]",
                 fontsize=13, fontweight="bold")

    for i, d in enumerate(lr_roc_folds):
        ax.plot(d["fpr"], d["tpr"], color=FOLD_COLORS[i], alpha=0.7,
                label=f"Fold {i+1} (AUC={d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("Logistic Regression"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{_dir1}/cv_roc_curves.png", dpi=150, bbox_inches="tight")


def plot_metric_distribution(lr_cv):
    """LR의 fold별 AUC·Accuracy·F1 박스플롯을 그려 cv_metric_distribution.png로 저장."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle(f"{N_FOLDS}-Fold CV Metric Distribution  [Logistic Regression]",
                 fontsize=13, fontweight="bold")

    for ax, (mname, mkey) in zip(axes, [("AUC-ROC", "auc"), ("Accuracy", "acc"), ("F1-Score", "f1")]):
        lr_vals = [m[mkey] for m in lr_cv]
        bp = ax.boxplot([lr_vals], labels=["Log. Reg."],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue"); patch.set_alpha(0.6)
        margin = max((max(lr_vals) - min(lr_vals)) * 0.4, 0.05)
        ax.set_ylim(max(0.0, min(lr_vals) - margin), min(1.0, max(lr_vals) + margin))
        ax.set_title(mname); ax.set_ylabel(mname)

    fig.tight_layout()
    fig.savefig(f"{_dir1}/cv_metric_distribution.png", dpi=150, bbox_inches="tight")


def plot_confusion_matrices(y_te, lr_pred, sex_te):
    """LR의 test set confusion matrix(전체 + 성별)를 confusion_matrices.png로 저장."""
    sexes = [s for s in ["M", "F"] if (sex_te == s).any()]
    n_cols = 1 + len(sexes)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices — Test Set  [Logistic Regression]",
                 fontsize=13, fontweight="bold")

    combos = [
        ("Logistic Regression — Overall", y_te, lr_pred),
        *[(f"Logistic Regression — {'Male' if s == 'M' else 'Female'} (n={(sex_te == s).sum()})",
           y_te[sex_te == s], lr_pred[sex_te == s]) for s in sexes],
    ]
    for ax, (title, yt, yp) in zip(axes, combos):
        cm = confusion_matrix(yt, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Sarco"],
                    yticklabels=["Normal", "Sarco"], cbar=False)
        ax.set_title(title)
        ax.set_ylabel("True")
        ax.set_xlabel("Pred")

    fig.tight_layout()
    fig.savefig(f"{_dir1}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(rn_histories, med_epoch):
    """ResNet1D의 fold별 train/val loss·val AUC 학습 커브(mean±std)를 training_curves.png로 저장."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"ResNet1D Training Curves ({N_FOLDS}-Fold Mean ± Std)",
                 fontsize=13, fontweight="bold")

    train_arr = np.array([h["train_loss"] for h in rn_histories])
    val_arr   = np.array([h["val_loss"]   for h in rn_histories])
    auc_arr   = np.array([h["val_auc"]    for h in rn_histories])
    ep_x      = np.arange(1, EPOCHS + 1)

    for arr, label, color in [(train_arr, "Train Loss", "steelblue"),
                               (val_arr,   "Val Loss",   "tomato")]:
        m, s = arr.mean(0), arr.std(0)
        ax_a.plot(ep_x, m, color=color, label=label)
        ax_a.fill_between(ep_x, m - s, m + s, color=color, alpha=0.2)

    m, s = auc_arr.mean(0), auc_arr.std(0)
    ax_b.plot(ep_x, m, color="seagreen", label="Val AUC")
    ax_b.fill_between(ep_x, m - s, m + s, color="seagreen", alpha=0.2)
    ax_b.axvline(med_epoch, color="gray", linestyle="--", alpha=0.7,
                 label=f"Median best ep={med_epoch}")

    for ax, title, ylabel in [(ax_a, "Loss", "Loss"), (ax_b, "Validation AUC", "AUC-ROC")]:
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend()

    fig.tight_layout()
    fig.savefig(f"{_dir1}/training_curves.png", dpi=150, bbox_inches="tight")


def plot_test_roc(y_te, lr_prob):
    """LR의 test set 전체 ROC 커브를 test_roc_curves.png로 저장."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Test Set ROC Curves  [Logistic Regression]", fontsize=13, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_te, lr_prob)
    auc = roc_auc_score(y_te, lr_prob)
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"Log. Reg. (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{_dir1}/test_roc_curves.png", dpi=150, bbox_inches="tight")


def plot_confusion_matrices_by_sex(y_te, lr_pred, sex_te):
    """LR의 성별 분리 confusion matrix를 confusion_matrices_by_sex.png로 저장."""
    sexes = [s for s in ["M", "F"] if (sex_te == s).any()]
    fig, axes = plt.subplots(1, len(sexes), figsize=(5 * len(sexes), 5))
    if len(sexes) == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices by Sex — Test Set  [Logistic Regression]",
                 fontsize=13, fontweight="bold")

    combos = [
        (f"LR — {'Male' if s=='M' else 'Female'} (n={(sex_te==s).sum()})",
         y_te[sex_te==s], lr_pred[sex_te==s]) for s in sexes
    ]
    for ax, (title, yt, yp) in zip(axes, combos):
        cm = confusion_matrix(yt, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Sarco"],
                    yticklabels=["Normal", "Sarco"], cbar=False)
        ax.set_title(title); ax.set_ylabel("True"); ax.set_xlabel("Pred")

    fig.tight_layout()
    fig.savefig(f"{_dir1}/confusion_matrices_by_sex.png", dpi=150, bbox_inches="tight")


def plot_test_roc_by_sex(y_te, lr_prob, sex_te):
    """LR의 test set 성별 분리 ROC 커브를 test_roc_by_sex.png로 저장."""
    sex_colors = {"M": "steelblue", "F": "tomato"}
    sex_labels = {"M": "Male", "F": "Female"}

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Test Set ROC Curves by Sex  [Logistic Regression]",
                 fontsize=13, fontweight="bold")

    for s, col in sex_colors.items():
        mask = sex_te == s
        if not mask.any():
            continue
        yt, ypr = y_te[mask], lr_prob[mask]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, ypr)
        auc = roc_auc_score(yt, ypr)
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{sex_labels[s]} (n={mask.sum()}, AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("Logistic Regression"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()

    fig.tight_layout()
    fig.savefig(f"{_dir1}/test_roc_by_sex.png", dpi=150, bbox_inches="tight")


def plot_data_distribution(X_cv, y_cv, sex_cv, X_te, y_te, sex_te, out_dir=None):
    """Train/Test 데이터셋의 클래스 비율(성별)·Age·BMI 분포를 data_distribution.png로 저장."""
    cls_colors = {"Normal": "steelblue", "Sarco": "tomato"}
    col_w = 4.5
    fig, axes = plt.subplots(2, 3, figsize=(col_w * 3, col_w * 2),
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    fig.suptitle("Dataset Distribution — Train (CV) vs Test", fontsize=13, fontweight="bold")

    for row, (X, y, sex, name) in enumerate([
        (X_cv, y_cv, sex_cv, "Train (CV)"),
        (X_te, y_te, sex_te, "Test"),
    ]):
        n_split = len(y)

        # Col 0: class proportion by sex (분모 = 해당 split의 성별 인원)
        ax = axes[row, 0]
        sex_list = [s for s in ["M", "F"] if (sex == s).any()]
        x_pos = np.arange(len(sex_list))
        width = 0.3
        for i, (cls, col) in enumerate(cls_colors.items()):
            props = [(y[sex == s] == i).sum() / (sex == s).sum() for s in sex_list]
            ax.bar(x_pos + (i - 0.5) * width, props, width,
                   label=cls, color=col, alpha=0.7)
            for j, p in enumerate(props):
                ax.text(x_pos[j] + (i - 0.5) * width, p + 0.01,
                        f"{p:.1%}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [f"{'Male' if s == 'M' else 'Female'}\n(n={(sex == s).sum()})"
             for s in sex_list]
        )
        ax.set_xlim(-0.7, len(sex_list) - 0.3)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(f"{name} — Class by Sex  (n={n_split})")
        ax.set_ylabel("Proportion within sex group"); ax.legend()

        # Col 1 / 2: Age & BMI — 클래스별 비율 히스토그램
        # 분모 = 해당 split × 해당 클래스 인원  →  각 클래스의 분포가 독립적으로 100%
        for ax, feat_col, feat_name in [
            (axes[row, 1], 0, "Age"),
            (axes[row, 2], 2, "BMI"),
        ]:
            shared_bins = np.linspace(X[:, feat_col].min(), X[:, feat_col].max(), 16)
            for i, (cls, col) in enumerate(cls_colors.items()):
                vals = X[y == i, feat_col]
                n_cls = max(len(vals), 1)          # 분모: 이 split에서 해당 클래스 인원
                weights = np.ones(n_cls) / n_cls
                ax.hist(vals, bins=shared_bins, weights=weights, alpha=0.6,
                        color=col, label=f"{cls} (n={len(vals)}, {len(vals)/n_split:.0%})")
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:.1%}"))
            ax.set_title(f"{name} — {feat_name}  (n={n_split})")
            ax.set_xlabel(feat_name)
            ax.set_ylabel("Proportion within class"); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_dir or RESULTS_DIR}/data_distribution.png", dpi=150, bbox_inches="tight")


def _dist_table(y_cv, sex_cv, y_te, sex_te):
    """
    비율(Normal %, Sarco %)은 각 split의 해당 그룹 인원 수를 분모로 계산.
    Train과 Test는 독립적으로 비율 산출.
    """
    lines = []
    lines.append("| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |")
    lines.append("|-------|-----|--:|-------:|---------:|------:|--------:|")
    for split_name, y, sex in [("Train", y_cv, sex_cv), ("Test", y_te, sex_te)]:
        for s in ["M", "F"]:
            mask = sex == s
            if not mask.any():
                continue
            ys = y[mask]
            n, sarco = len(ys), int(ys.sum())
            normal = n - sarco
            lines.append(
                f"| {split_name} | {s} | {n} "
                f"| {normal} | {normal/n*100:.1f}% "
                f"| {sarco} | {sarco/n*100:.1f}% |"
            )
        n, sarco = len(y), int(y.sum())
        normal = n - sarco
        lines.append(
            f"| {split_name} | **All** | **{n}** "
            f"| **{normal}** | **{normal/n*100:.1f}%** "
            f"| **{sarco}** | **{sarco/n*100:.1f}%** |"
        )
    return "\n".join(lines)


def _feature_table(X_cv, sex_cv, X_te, sex_te):
    """Age(col 0) / BMI(col 2) 통계를 Split × Sex 기준 마크다운 테이블로 반환."""
    def _stat_row(split, sex_label, vals, bold=False):
        b = "**" if bold else ""
        return (
            f"| {split} | {b}{sex_label}{b} | {b}{len(vals)}{b} |"
            f" {b}{vals.mean():.2f} ± {vals.std():.2f}{b} |"
            f" {b}{vals.min():.2f}{b} | {b}{np.median(vals):.2f}{b} | {b}{vals.max():.2f}{b} |"
        )

    blocks = []
    for feat_name, col in [("Age", 0), ("BMI", 2)]:
        lines = [
            f"### {feat_name}",
            "",
            "| Split | Sex | n | Mean ± Std | Min | Median | Max |",
            "|-------|-----|--:|----------:|----:|-------:|----:|",
        ]
        for split_name, X, sex in [("Train", X_cv, sex_cv), ("Test", X_te, sex_te)]:
            for s in ["M", "F"]:
                mask = sex == s
                if mask.any():
                    lines.append(_stat_row(split_name, s, X[mask, col]))
            lines.append(_stat_row(split_name, "All", X[:, col], bold=True))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _cv_table(fold_metrics):
    """fold별 AUC·AUPRC·Brier·Accuracy·F1과 mean/±std 행을 마크다운 테이블 문자열로 반환."""
    lines = []
    lines.append("| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |")
    lines.append("|------|--------:|------:|------:|---------:|---:|")
    for m in fold_metrics:
        lines.append(
            f"| {m['fold']} | {m['auc']:.4f} | {m['auprc']:.4f}"
            f" | {m['brier']:.4f} | {m['acc']:.4f} | {m['f1']:.4f} |"
        )
    aucs   = [m["auc"]   for m in fold_metrics]
    auprcs = [m["auprc"] for m in fold_metrics]
    briers = [m["brier"] for m in fold_metrics]
    accs   = [m["acc"]   for m in fold_metrics]
    f1s    = [m["f1"]    for m in fold_metrics]
    lines.append(
        f"| **Mean** | **{np.mean(aucs):.4f}** | **{np.mean(auprcs):.4f}**"
        f" | **{np.mean(briers):.4f}** | **{np.mean(accs):.4f}** | **{np.mean(f1s):.4f}** |"
    )
    lines.append(
        f"| **±Std** | {np.std(aucs):.4f} | {np.std(auprcs):.4f}"
        f" | {np.std(briers):.4f} | {np.std(accs):.4f} | {np.std(f1s):.4f} |"
    )
    return "\n".join(lines)


def _sex_rows(y_true, y_pred, y_prob, sex_te):
    """성별(M/F)별 AUC·AUPRC·Brier·Accuracy·F1을 마크다운 테이블 행 문자열로 반환."""
    rows = []
    for s in ["M", "F"]:
        mask = sex_te == s
        if mask.sum() == 0:
            continue
        yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
        has_both = len(np.unique(yt)) > 1
        auc   = roc_auc_score(yt, ypr)           if has_both else float("nan")
        auprc = average_precision_score(yt, ypr) if has_both else float("nan")
        brier = brier_score_loss(yt, ypr)        if has_both else float("nan")
        rows.append(
            f"| {s} | {mask.sum()} | {auc:.4f} | {auprc:.4f} | {brier:.4f}"
            f" | {accuracy_score(yt, yp):.4f} | {f1_score(yt, yp, zero_division=0):.4f} |"
        )
    return "\n".join(rows)


def _cm_block(y_true, y_pred):
    """confusion matrix를 2×2 마크다운 테이블 문자열로 반환."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return (
        f"|  | Pred: Normal | Pred: Sarco |\n"
        f"|--|-------------:|------------:|\n"
        f"| **True: Normal** | {tn} | {fp} |\n"
        f"| **True: Sarco**  | {fn} | {tp} |"
    )


def save_report_md(lr_cv,
                   X_cv, y_cv, sex_cv,
                   X_te, y_te, lr_pred, lr_prob,
                   sex_te):
    """LR CV 결과와 test set 성능 지표를 results.md 파일로 저장 (Model 1 전용)."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# SMI Binary Classification — Results",
        f"",
        f"Generated: {now}  |  {N_FOLDS}-Fold CV  |  Model 1 (Clinic Only, LR)",
        f"",
        f"---",
        f"",
        f"## 0. Dataset Distribution",
        f"",
        f"### Class Distribution",
        f"",
        _dist_table(y_cv, sex_cv, y_te, sex_te),
        f"",
        _feature_table(X_cv, sex_cv, X_te, sex_te),
        f"",
        f"![Data Distribution](data_distribution.png)",
        f"",
        f"---",
        f"",
        f"## 1. Cross-Validation Summary",
        f"",
        f"### Logistic Regression",
        f"",
        _cv_table(lr_cv),
        f"",
        f"---",
        f"",
        f"## 2. Test Set Performance",
        f"",
        f"### Overall",
        f"",
        f"| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
        f"|-------|--------:|------:|------:|---------:|---:|",
        f"| Log. Reg. | {roc_auc_score(y_te, lr_prob):.4f}"
        f" | {average_precision_score(y_te, lr_prob):.4f}"
        f" | {brier_score_loss(y_te, lr_prob):.4f}"
        f" | {accuracy_score(y_te, lr_pred):.4f}"
        f" | {f1_score(y_te, lr_pred, zero_division=0):.4f} |",
        f"",
        f"### By Sex",
        f"",
        f"| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
        f"|-----|--:|--------:|------:|------:|---------:|---:|",
        _sex_rows(y_te, lr_pred, lr_prob, sex_te),
        f"",
        f"---",
        f"",
        f"## 3. Confusion Matrix (Test Set)",
        f"",
        _cm_block(y_te, lr_pred),
        f"",
        f"---",
        f"",
        f"## 4. Figures",
        f"",
        f"| File | Description |",
        f"|------|-------------|",
        f"| `data_distribution.png` | Train/Test class·Age·BMI distributions |",
        f"| `cv_roc_curves.png` | Per-fold ROC curves (LR) |",
        f"| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |",
        f"| `confusion_matrices.png` | Test-set confusion matrices (overall + by sex) |",
        f"| `test_roc_curves.png` | Final test-set ROC curve (overall) |",
        f"| `test_roc_by_sex.png` | Final test-set ROC curves split by sex |",
        f"| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |",
    ]

    md_path = f"{_dir1}/results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {md_path}")


def plot_calibration(entries, out_path, n_bins=10):
    """
    Calibration plot (reliability diagram) + Precision-Recall curve를 나란히 그린다.
    entries: list of (label, y_true, y_prob, color)
    """
    fig, (ax_cal, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Calibration & Precision-Recall Curves — Test Set",
                 fontsize=13, fontweight="bold")

    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    for label, y_true, y_prob, color in entries:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        brier = brier_score_loss(y_true, y_prob)
        ax_cal.plot(mean_pred, frac_pos, marker="o", color=color, linewidth=2,
                    label=f"{label}  (Brier={brier:.4f})")
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.set_title("Calibration Plot")
    ax_cal.legend(fontsize=9)

    for label, y_true, y_prob, color in entries:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax_pr.plot(rec, prec, color=color, linewidth=2, label=f"{label}  (AUPRC={ap:.4f})")
    baseline = y_true.mean()  # noqa: uses last entry's y_true — ok for same dataset
    ax_pr.axhline(baseline, color="gray", linestyle="--", alpha=0.5,
                  label=f"Baseline (prevalence={baseline:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_all(lr_roc_folds, lr_cv,
             X_cv, y_cv, sex_cv,
             X_te, y_te, lr_pred, lr_prob,
             sex_te, out_dir=None):
    """Model 1용 시각화 전체(7종 png)와 results.md를 out_dir에 저장."""
    global _dir1
    _dir1 = out_dir or RESULTS_DIR
    plot_data_distribution(X_cv, y_cv, sex_cv, X_te, y_te, sex_te)
    plot_roc_curves(lr_roc_folds)
    plot_metric_distribution(lr_cv)
    plot_confusion_matrices(y_te, lr_pred, sex_te)
    plot_test_roc(y_te, lr_prob)
    plot_test_roc_by_sex(y_te, lr_prob, sex_te)
    plot_calibration(
        [("Log. Reg.", y_te, lr_prob, "steelblue")],
        out_path=f"{_dir1}/calibration.png",
    )

    print("\nSaved:")
    for fname in ["data_distribution", "cv_roc_curves", "cv_metric_distribution",
                  "confusion_matrices", "test_roc_curves", "test_roc_by_sex", "calibration"]:
        print(f"  {_dir1}/{fname}.png")

    save_report_md(lr_cv,
                   X_cv, y_cv, sex_cv,
                   X_te, y_te, lr_pred, lr_prob,
                   sex_te)


# ── Model 2 : Clinic + AEC Cross-Attention ───────────────────

def _save_cross(fname, fig):
    """fig를 _dir2 디렉토리에 fname 파일명으로 저장하고 닫는다."""
    fig.tight_layout()
    fig.savefig(f"{_dir2}/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv_roc_cross(ca_roc_folds):
    """CrossAttn의 fold별 ROC 커브를 그려 cv_roc_curves.png로 저장."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"CV ROC Curves — {N_FOLDS}-Fold  [CrossAttn]",
                 fontsize=13, fontweight="bold")
    for i, d in enumerate(ca_roc_folds):
        ax.plot(d["fpr"], d["tpr"], color=FOLD_COLORS[i], alpha=0.7,
                label=f"Fold {i+1} (AUC={d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("CrossAttn"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)
    _save_cross("cv_roc_curves.png", fig)


def plot_cv_metric_cross(ca_cv):
    """CrossAttn의 fold별 AUC·Accuracy·F1 박스플롯을 그려 cv_metric_distribution.png로 저장."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"CV Metric Distribution ({N_FOLDS}-Fold)",
                 fontsize=13, fontweight="bold")
    for ax, (mname, mkey) in zip(axes, [("AUC-ROC", "auc"), ("Accuracy", "acc"), ("F1-Score", "f1")]):
        ca_vals = [m[mkey] for m in ca_cv]
        bp = ax.boxplot([ca_vals], labels=["CrossAttn"],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("tomato"); patch.set_alpha(0.6)
        margin = max((max(ca_vals) - min(ca_vals)) * 0.4, 0.05)
        ax.set_ylim(max(0.0, min(ca_vals) - margin), min(1.0, max(ca_vals) + margin))
        ax.set_title(mname); ax.set_ylabel(mname)
    _save_cross("cv_metric_distribution.png", fig)


def plot_training_curves_cross(ca_histories, med_epoch):
    """CrossAttn의 fold별 train/val loss·val AUC 학습 커브(mean±std)를 training_curves.png로 저장."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"CrossAttn Training Curves ({N_FOLDS}-Fold Mean ± Std)",
                 fontsize=13, fontweight="bold")
    train_arr = np.array([h["train_loss"] for h in ca_histories])
    val_arr   = np.array([h["val_loss"]   for h in ca_histories])
    auc_arr   = np.array([h["val_auc"]    for h in ca_histories])
    ep_x      = np.arange(1, EPOCHS + 1)
    for arr, label, color in [(train_arr, "Train Loss", "steelblue"),
                               (val_arr,   "Val Loss",   "tomato")]:
        m, s = arr.mean(0), arr.std(0)
        ax_a.plot(ep_x, m, color=color, label=label)
        ax_a.fill_between(ep_x, m - s, m + s, color=color, alpha=0.2)
    m, s = auc_arr.mean(0), auc_arr.std(0)
    ax_b.plot(ep_x, m, color="seagreen", label="Val AUC")
    ax_b.fill_between(ep_x, m - s, m + s, color="mediumpurple", alpha=0.2)
    ax_b.axvline(med_epoch, color="gray", linestyle="--", alpha=0.7,
                 label=f"Median best ep={med_epoch}")
    for ax, title, ylabel in [(ax_a, "Loss", "Loss"), (ax_b, "Validation AUC", "AUC-ROC")]:
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend()
    _save_cross("training_curves.png", fig)


def plot_test_roc_cross(ca_true_te, ca_prob_te):
    """CrossAttn의 test set 전체 ROC 커브를 test_roc_curves.png로 저장."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Test Set ROC Curves", fontsize=13, fontweight="bold")
    fpr, tpr, _ = roc_curve(ca_true_te, ca_prob_te)
    auc = roc_auc_score(ca_true_te, ca_prob_te)
    ax.plot(fpr, tpr, color="tomato", linewidth=2, label=f"CrossAttn (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    _save_cross("test_roc_curves.png", fig)


def plot_test_roc_by_sex_cross(ca_true_te, ca_prob_te, sex_te):
    """CrossAttn의 test set 성별 분리 ROC 커브를 test_roc_by_sex.png로 저장."""
    sex_colors = {"M": "tomato", "F": "steelblue"}
    sex_labels = {"M": "Male", "F": "Female"}
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Test Set ROC Curves by Sex  [CrossAttn]", fontsize=13, fontweight="bold")
    for s, col in sex_colors.items():
        mask = sex_te == s
        if not mask.any():
            continue
        yt, ypr = ca_true_te[mask], ca_prob_te[mask]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, ypr)
        auc = roc_auc_score(yt, ypr)
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{sex_labels[s]} (n={mask.sum()}, AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("CrossAttn"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    _save_cross("test_roc_by_sex.png", fig)


def plot_confusion_matrices_cross(ca_true_te, ca_pred_te, sex_te):
    """CrossAttn의 test set confusion matrix(전체 + 성별)를 confusion_matrices.png로 저장."""
    sexes = [s for s in ["M", "F"] if (sex_te == s).any()]
    n_cols = 1 + len(sexes)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices — Test Set  [CrossAttn]", fontsize=13, fontweight="bold")

    combos = [
        ("CrossAttn — Overall", ca_true_te, ca_pred_te),
        *[(f"CrossAttn — {'Male' if s=='M' else 'Female'} (n={(sex_te==s).sum()})",
           ca_true_te[sex_te==s], ca_pred_te[sex_te==s]) for s in sexes],
    ]
    for ax, (title, yt, yp) in zip(axes, combos):
        cm = confusion_matrix(yt, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                    xticklabels=["Normal", "Sarco"],
                    yticklabels=["Normal", "Sarco"], cbar=False)
        ax.set_title(title); ax.set_ylabel("True"); ax.set_xlabel("Pred")
    _save_cross("confusion_matrices.png", fig)


def _save_report_md_cross(ca_cv, X_cv, y_cv, sex_cv, X_te, y_te,
                           ca_pred_te, ca_prob_te, ca_true_te,
                           sex_te, ca_histories, med_epoch):
    """CrossAttn CV 결과와 test set 성능 지표를 results.md 파일로 저장 (Model 2/2_2/3 공용)."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    auc_arr = np.array([h["val_auc"] for h in ca_histories])
    best_val_aucs = auc_arr.max(axis=1)

    lines = [
        "# SMI Binary Classification — CrossAttn Results",
        "",
        f"Generated: {now}  |  {N_FOLDS}-Fold CV  |  Median best epoch: {med_epoch}",
        "",
        "---",
        "",
        "## 0. Dataset Distribution",
        "",
        "### Class Distribution",
        "",
        _dist_table(y_cv, sex_cv, y_te, sex_te),
        "",
        _feature_table(X_cv, sex_cv, X_te, sex_te),
        "",
        "![Data Distribution](data_distribution.png)",
        "",
        "---",
        "",
        "## 1. Cross-Validation Summary",
        "",
        "### CrossAttn",
        "",
        _cv_table(ca_cv),
        "",
        "CrossAttn best val AUC per fold: " +
        ", ".join(f"Fold{i+1}={v:.4f}" for i, v in enumerate(best_val_aucs)),
        "",
        "---",
        "",
        "## 2. Test Set Performance",
        "",
        "### Overall",
        "",
        "| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
        "|-------|--------:|------:|------:|---------:|---:|",
        f"| CrossAttn | {roc_auc_score(ca_true_te, ca_prob_te):.4f}"
        f" | {average_precision_score(ca_true_te, ca_prob_te):.4f}"
        f" | {brier_score_loss(ca_true_te, ca_prob_te):.4f}"
        f" | {accuracy_score(ca_true_te, ca_pred_te):.4f}"
        f" | {f1_score(ca_true_te, ca_pred_te, zero_division=0):.4f} |",
        "",
        "### By Sex",
        "",
        "| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
        "|-----|--:|--------:|------:|------:|---------:|---:|",
        _sex_rows(ca_true_te, ca_pred_te, ca_prob_te, sex_te),
        "",
        "---",
        "",
        "## 3. Confusion Matrix (Test Set)",
        "",
        _cm_block(ca_true_te, ca_pred_te),
        "",
        "---",
        "",
        "## 4. Figures",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `data_distribution.png` | Train/Test class·Age·BMI distributions |",
        "| `cv_roc_curves.png` | Per-fold ROC curves (CrossAttn) |",
        "| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |",
        "| `training_curves.png` | Loss & AUC training curves (mean ± std) |",
        "| `test_roc_curves.png` | Final test-set ROC curve |",
        "| `test_roc_by_sex.png` | Final test-set ROC curves by sex |",
        "| `confusion_matrices.png` | Test-set confusion matrices |",
        "| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |",
    ]

    md_path = f"{_dir2}/results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {md_path}")


def save_all_cross(ca_cv, ca_roc_folds, ca_histories, med_epoch,
                   X_clin_cv, y_cv, sex_cv,
                   X_clin_te, y_te,
                   ca_pred_te, ca_true_te, sex_te, ca_prob_te,
                   model_label="model 2", out_dir=None):
    """Model 2/2_2/3용 시각화 전체(8종 png)와 results.md를 out_dir에 저장."""
    global _dir2
    _dir2 = out_dir or RESULTS_DIR_CROSS
    plot_data_distribution(X_clin_cv, y_cv, sex_cv, X_clin_te, y_te, sex_te,
                           out_dir=_dir2)
    plot_cv_roc_cross(ca_roc_folds)
    plot_cv_metric_cross(ca_cv)
    plot_training_curves_cross(ca_histories, med_epoch)
    plot_test_roc_cross(ca_true_te, ca_prob_te)
    plot_test_roc_by_sex_cross(ca_true_te, ca_prob_te, sex_te)
    plot_confusion_matrices_cross(ca_true_te, ca_pred_te, sex_te)
    plot_calibration(
        [("CrossAttn", ca_true_te, ca_prob_te, "tomato")],
        out_path=f"{_dir2}/calibration.png",
    )

    print(f"\nSaved ({model_label}):")
    for fname in ["data_distribution", "cv_roc_curves", "cv_metric_distribution",
                  "training_curves", "test_roc_curves", "test_roc_by_sex",
                  "confusion_matrices", "calibration"]:
        print(f"  {_dir2}/{fname}.png")

    _save_report_md_cross(ca_cv, X_clin_cv, y_cv, sex_cv, X_clin_te, y_te,
                          ca_pred_te, ca_prob_te, ca_true_te,
                          sex_te, ca_histories, med_epoch)
