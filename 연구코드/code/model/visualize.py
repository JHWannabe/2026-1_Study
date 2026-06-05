"""
시각화 및 Markdown 보고서 생성 유틸리티.

Model 1 (Clinic Only, LR):
  save_all()            — 7종 PNG + results.md 를 out_dir에 일괄 저장

Model 2/2_2/3 (Clinic + AEC / Scanner):
  save_all_cross()      — 8종 PNG + results.md 를 out_dir에 일괄 저장
  plot_attention_maps() — CrossAttn 모델의 Clinical→AEC attention 시각화
                          attention_map_c2a.png : 클래스별 토큰 평균 bar + AEC 신호 오버레이
                          attention_heatmap.png : 샘플별 heatmap (Sarco→Normal 순 정렬)

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

from config import N_FOLDS, EPOCHS, RESULTS_DIR, RESULTS_MODEL_2_DIR

FOLD_COLORS = plt.get_cmap("tab10")(np.linspace(0, 0.45, N_FOLDS))

# 현재 저장 디렉토리 (save_all / save_all_cross 호출 시 갱신)
_dir1: str = RESULTS_DIR
_dir2: str = RESULTS_MODEL_2_DIR


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
                    annot_kws={"size": 13},
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
                    annot_kws={"size": 13},
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
    save_dir = out_dir if out_dir is not None else RESULTS_DIR
    fig.savefig(f"{save_dir}/data_distribution.png", dpi=150, bbox_inches="tight")


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


def _ci_section(ci_dict):
    """Bootstrap CI dict → markdown 섹션 행 리스트. ci_dict=None이면 빈 테이블."""
    labels = [("AUC-ROC", "auc"), ("AUPRC", "auprc"),
              ("Brier", "brier"), ("Accuracy", "acc"), ("F1", "f1")]
    lines = [
        "## 4. Bootstrap 95% CI  (n_boot=2000)",
        "",
        "> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.",
        "",
        "| Metric | Estimate | CI Lower | CI Upper |",
        "|--------|--------:|---------:|---------:|",
    ]
    if ci_dict:
        for label, key in labels:
            if key in ci_dict:
                est, lo, hi = ci_dict[key]
                lines.append(f"| {label} | {est:.4f} | {lo:.4f} | {hi:.4f} |")
    return lines


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
                   sex_te, ci_dict=None):
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
        *_ci_section(ci_dict),
        f"",
        f"---",
        f"",
        f"## 5. Figures",
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
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
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


def plot_roc_all_models(aec_var,
                        r1_y, r1_prob,
                        r2_y, r2_prob,
                        r2_2_y, r2_2_prob,
                        r3_y, r3_prob,
                        out_path,
                        r4_y=None, r4_prob=None):
    """M1/M2/M2_2/M3(/M4) test-set ROC 커브를 하나의 이미지에 비교."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"ROC Comparison — AEC variant: {aec_var}",
                 fontsize=13, fontweight="bold")

    entries = [
        ("Model 1  LR (Clinic Only)",        r1_y,   r1_prob,   "steelblue", "--"),
        (f"Model 2  CrossAttn ({aec_var})",  r2_y,   r2_prob,   "tomato",    "-"),
        (f"Model 2_2 Unmatched ({aec_var})", r2_2_y, r2_2_prob, "orange",    "-."),
        (f"Model 3  CrossAttn3 ({aec_var})", r3_y,   r3_prob,   "seagreen",  "-"),
    ]
    if r4_y is not None and r4_prob is not None:
        entries.append((f"Model 4  AECOnly ({aec_var})", r4_y, r4_prob, "mediumpurple", ":"))

    for label, y_true, y_prob, color, ls in entries:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=2, linestyle=ls,
                label=f"{label}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_all(lr_roc_folds, lr_cv,
             X_cv, y_cv, sex_cv,
             X_te, y_te, lr_pred, lr_prob,
             sex_te, out_dir=None, ci_dict=None):
    """Model 1용 시각화 전체(7종 png)와 results.md를 out_dir에 저장."""
    global _dir1
    _dir1 = out_dir or RESULTS_DIR
    plot_data_distribution(X_cv, y_cv, sex_cv, X_te, y_te, sex_te, out_dir=_dir1)
    if _dir1 != RESULTS_DIR:
        import shutil
        shutil.copy2(f"{_dir1}/data_distribution.png",
                     f"{RESULTS_DIR}/data_distribution.png")
    plot_roc_curves(lr_roc_folds)
    plot_metric_distribution(lr_cv)
    plot_confusion_matrices(y_te, lr_pred, sex_te)
    plot_test_roc(y_te, lr_prob)
    plot_test_roc_by_sex(y_te, lr_prob, sex_te)
    plot_calibration(
        [("Log. Reg.", y_te, lr_prob, "steelblue")],
        out_path=f"{_dir1}/calibration_.png",
    )

    print("\nSaved:")
    for fname in ["data_distribution", "cv_roc_curves", "cv_metric_distribution",
                  "confusion_matrices", "test_roc_curves", "test_roc_by_sex", "calibration"]:
        print(f"  {_dir1}/{fname}.png")

    save_report_md(lr_cv,
                   X_cv, y_cv, sex_cv,
                   X_te, y_te, lr_pred, lr_prob,
                   sex_te, ci_dict=ci_dict)


# ── Model 2 : Clinic + AEC Cross-Attention ───────────────────

def _save_cross(fname, fig):
    """fig를 _dir2 디렉토리에 fname 파일명으로 저장하고 닫는다."""
    fig.tight_layout()
    fig.savefig(f"{_dir2}/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv_roc_cross(ca_roc_folds, model_label="CrossAttn"):
    """CrossAttn의 fold별 ROC 커브를 그려 cv_roc_curves.png로 저장."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"CV ROC Curves — {N_FOLDS}-Fold  [{model_label}]",
                 fontsize=13, fontweight="bold")
    for i, d in enumerate(ca_roc_folds):
        ax.plot(d["fpr"], d["tpr"], color=FOLD_COLORS[i], alpha=0.7,
                label=f"Fold {i+1} (AUC={d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title(model_label); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)
    _save_cross("cv_roc_curves.png", fig)


def plot_cv_metric_cross(ca_cv, model_label="CrossAttn"):
    """CrossAttn의 fold별 AUC·Accuracy·F1 박스플롯을 그려 cv_metric_distribution.png로 저장."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"CV Metric Distribution ({N_FOLDS}-Fold)",
                 fontsize=13, fontweight="bold")
    for ax, (mname, mkey) in zip(axes, [("AUC-ROC", "auc"), ("Accuracy", "acc"), ("F1-Score", "f1")]):
        ca_vals = [m[mkey] for m in ca_cv]
        bp = ax.boxplot([ca_vals], labels=[model_label],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue"); patch.set_alpha(0.6)
        margin = max((max(ca_vals) - min(ca_vals)) * 0.4, 0.05)
        ax.set_ylim(max(0.0, min(ca_vals) - margin), min(1.0, max(ca_vals) + margin))
        ax.set_title(mname); ax.set_ylabel(mname)
    _save_cross("cv_metric_distribution.png", fig)


def plot_training_curves_cross(ca_histories, med_epoch, model_label="CrossAttn"):
    """CrossAttn의 fold별 train/val loss·val AUC 학습 커브(mean±std)를 training_curves.png로 저장."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{model_label} Training Curves ({N_FOLDS}-Fold Mean ± Std)",
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
    ax_b.fill_between(ep_x, m - s, m + s, color="seagreen", alpha=0.2)
    ax_b.axvline(med_epoch, color="gray", linestyle="--", alpha=0.7,
                 label=f"Median best ep={med_epoch}")
    for ax, title, ylabel in [(ax_a, "Loss", "Loss"), (ax_b, "Validation AUC", "AUC-ROC")]:
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend()
    _save_cross("training_curves.png", fig)


def plot_test_roc_cross(ca_true_te, ca_prob_te, model_label="CrossAttn"):
    """CrossAttn의 test set 전체 ROC 커브를 test_roc_curves.png로 저장."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Test Set ROC Curves", fontsize=13, fontweight="bold")
    fpr, tpr, _ = roc_curve(ca_true_te, ca_prob_te)
    auc = roc_auc_score(ca_true_te, ca_prob_te)
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"{model_label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    _save_cross("test_roc_curves.png", fig)


def plot_test_roc_with_baseline(primary_true, primary_prob, primary_label,
                                 baseline_true, baseline_prob, baseline_label,
                                 out_path):
    """Primary 모델의 ROC 커브와 Baseline 모델의 ROC 커브를 함께 그려 out_path에 저장."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Test Set ROC Curves (vs Baseline)", fontsize=13, fontweight="bold")

    fpr_b, tpr_b, _ = roc_curve(baseline_true, baseline_prob)
    auc_b = roc_auc_score(baseline_true, baseline_prob)
    ax.plot(fpr_b, tpr_b, color="steelblue", linewidth=2, linestyle="--", alpha=0.8,
            label=f"{baseline_label} (AUC={auc_b:.3f})")

    fpr_p, tpr_p, _ = roc_curve(primary_true, primary_prob)
    auc_p = roc_auc_score(primary_true, primary_prob)
    ax.plot(fpr_p, tpr_p, color="tomato", linewidth=2,
            label=f"{primary_label} (AUC={auc_p:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_test_roc_by_sex_cross(ca_true_te, ca_prob_te, sex_te, model_label="CrossAttn"):
    """CrossAttn의 test set 성별 분리 ROC 커브를 test_roc_by_sex.png로 저장."""
    sex_colors = {"M": "steelblue", "F": "tomato"}
    sex_labels = {"M": "Male", "F": "Female"}
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"Test Set ROC Curves by Sex  [{model_label}]", fontsize=13, fontweight="bold")
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
    ax.set_title(model_label); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    _save_cross("test_roc_by_sex.png", fig)


def plot_confusion_matrices_cross(ca_true_te, ca_pred_te, sex_te, model_label="CrossAttn"):
    """CrossAttn의 test set confusion matrix(전체 + 성별)를 confusion_matrices.png로 저장."""
    sexes = [s for s in ["M", "F"] if (sex_te == s).any()]
    n_cols = 1 + len(sexes)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(f"Confusion Matrices — Test Set  [{model_label}]", fontsize=13, fontweight="bold")

    combos = [
        (f"{model_label} — Overall", ca_true_te, ca_pred_te),
        *[(f"{model_label} — {'Male' if s=='M' else 'Female'} (n={(sex_te==s).sum()})",
           ca_true_te[sex_te==s], ca_pred_te[sex_te==s]) for s in sexes],
    ]
    for ax, (title, yt, yp) in zip(axes, combos):
        cm = confusion_matrix(yt, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    annot_kws={"size": 15},
                    xticklabels=["Normal", "Sarco"],
                    yticklabels=["Normal", "Sarco"], cbar=False)
        ax.set_title(title); ax.set_ylabel("True"); ax.set_xlabel("Pred")
    _save_cross("confusion_matrices.png", fig)


def _save_report_md_cross(ca_cv, X_cv, y_cv, sex_cv, X_te, y_te,
                           ca_pred_te, ca_prob_te, ca_true_te,
                           sex_te, ca_histories, med_epoch, ci_dict=None):
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
        *_ci_section(ci_dict),
        "",
        "---",
        "",
        "## 5. Figures",
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


def plot_attention_maps(model, X_clin_te_s, X_aec_te_s, y_true_te,
                        out_dir, aec_var, model_label, X_mfr_te=None):
    """
    CrossAttn 모델의 양방향(Clinical↔AEC) attention을 시각화.

    네 파일 생성:
      attention_map_c2a.png     — Clinical→AEC: 클래스별 토큰 평균 bar + AEC 신호 오버레이
      attention_heatmap_c2a.png — Clinical→AEC: 샘플별 heatmap (Sarco→Normal 순 정렬)
      attention_map_a2c.png     — AEC→Clinical: 클래스별 clinical 토큰별 attention bar
      attention_heatmap_a2c.png — AEC→Clinical: 샘플별 heatmap

    keys: "clinical_to_aec"/"aec_to_clinical" (M2/M2_2) 또는 "cs_to_aec"/"aec_to_cs" (M3).
    c2a shape: (B, n_clin_tokens, n_aec_tokens) → query 평균 → (B, n_aec_tokens)
    a2c shape: (B, n_aec_tokens, n_clin_tokens) → query 평균 → (B, n_clin_tokens)
    """
    import torch
    from config import DEVICE

    model.eval()
    X_c = torch.tensor(X_clin_te_s, dtype=torch.float32).to(DEVICE)
    X_a = torch.tensor(X_aec_te_s,  dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        if X_mfr_te is not None:
            X_m = torch.tensor(X_mfr_te, dtype=torch.long).to(DEVICE)
            _, attn = model(X_c, X_a, X_m, return_attention=True)
            c2a_key, a2c_key = "cs_to_aec", "aec_to_cs"
        else:
            _, attn = model(X_c, X_a, return_attention=True)
            c2a_key, a2c_key = "clinical_to_aec", "aec_to_clinical"

    # Clinical→AEC: (B, n_clin_tokens, n_aec_tokens) → mean over clin query → (B, n_aec_tokens)
    attn_c2a = attn[c2a_key].cpu().numpy().mean(axis=1)
    # AEC→Clinical: (B, n_aec_tokens, n_clin_tokens) → mean over aec query → (B, n_clin_tokens)
    attn_a2c = attn[a2c_key].cpu().numpy().mean(axis=1)

    n_aec_tokens  = attn_c2a.shape[1]
    n_clin_tokens = attn_a2c.shape[1]
    aec_len = X_aec_te_s.shape[1]
    x_aec  = np.arange(n_aec_tokens)
    x_clin = np.arange(n_clin_tokens)

    _clin_label_map = {3: ["Age", "Sex", "BMI"], 4: ["Age", "Sex", "BMI", "Scanner"]}
    clin_labels = _clin_label_map.get(n_clin_tokens, [f"T{i}" for i in range(n_clin_tokens)])

    cls_info = [(0, "Normal", "steelblue"), (1, "Sarcopenia", "tomato")]

    s_idx   = np.where(y_true_te == 1)[0]
    n_idx   = np.where(y_true_te == 0)[0]
    ordered = np.concatenate([s_idx, n_idx])
    n_samp  = len(ordered)

    # ── Plot 1: Clinical→AEC bar chart + AEC signal overlay ──────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Attention Map (Clinical → AEC)  ·  {model_label}  [{aec_var}]",
        fontsize=11, fontweight="bold",
    )
    for ax, (cls_idx, cls_name, color) in zip(axes, cls_info):
        mask = y_true_te == cls_idx
        if not mask.any():
            ax.set_visible(False)
            continue
        mean_a = attn_c2a[mask].mean(0)
        std_a  = attn_c2a[mask].std(0)
        ax.bar(x_aec, mean_a, color=color, alpha=0.75,
               yerr=std_a, capsize=2, error_kw={"elinewidth": 0.8})
        ax2 = ax.twinx()
        ax2.plot(np.linspace(0, n_aec_tokens - 1, aec_len),
                 X_aec_te_s[mask].mean(0),
                 color="black", alpha=0.30, linewidth=1.0, label="Mean AEC")
        ax2.set_ylabel("Mean AEC (scaled)", color="dimgray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="dimgray", labelsize=7)
        ax.set_title(f"{cls_name}  (n={mask.sum()})")
        ax.set_xlabel("AEC Token Index  (left = scan start,  right = scan end)")
        ax.set_ylabel("Mean Attention Weight")
        ax.set_xlim(-0.5, n_aec_tokens - 0.5)

    fig.tight_layout()
    path1 = f"{out_dir}/attention_map_c2a.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: Clinical→AEC heatmap (Sarco → Normal 순) ─────────
    hmap_c2a = attn_c2a[ordered]
    figH = max(5, n_samp * 0.20)
    fig, ax = plt.subplots(figsize=(14, figH))
    im = ax.imshow(hmap_c2a, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   vmin=0, vmax=hmap_c2a.max())
    plt.colorbar(im, ax=ax, shrink=0.6, label="Attention Weight")
    if len(s_idx) > 0 and len(n_idx) > 0:
        ax.axhline(len(s_idx) - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.text(-0.8, len(s_idx) / 2 - 0.5,
            "Sarco",  va="center", ha="right",
            color="tomato",    fontsize=8, fontweight="bold", clip_on=False)
    ax.text(-0.8, len(s_idx) + len(n_idx) / 2 - 0.5,
            "Normal", va="center", ha="right",
            color="steelblue", fontsize=8, fontweight="bold", clip_on=False)
    ax.set_xlabel("AEC Token Index")
    ax.set_ylabel("Sample")
    ax.set_title(
        f"Sample-level Attention Map (Clinical → AEC)  ·  {model_label}  [{aec_var}]",
        fontsize=10, fontweight="bold",
    )
    tick_step = max(1, n_aec_tokens // 8)
    ax.set_xticks(x_aec[::tick_step])
    ax.set_xticklabels(x_aec[::tick_step])
    ax.set_yticks([])
    fig.tight_layout()
    path2 = f"{out_dir}/attention_heatmap_c2a.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: AEC→Clinical bar chart ───────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(max(6, n_clin_tokens * 1.8 + 2), 8))
    fig.suptitle(
        f"Attention Map (AEC → Clinical)  ·  {model_label}  [{aec_var}]",
        fontsize=11, fontweight="bold",
    )
    for ax, (cls_idx, cls_name, color) in zip(axes, cls_info):
        mask = y_true_te == cls_idx
        if not mask.any():
            ax.set_visible(False)
            continue
        mean_a = attn_a2c[mask].mean(0)
        std_a  = attn_a2c[mask].std(0)
        ax.bar(x_clin, mean_a, color=color, alpha=0.75,
               yerr=std_a, capsize=3, error_kw={"elinewidth": 0.8})
        ax.set_title(f"{cls_name}  (n={mask.sum()})")
        ax.set_xlabel("Clinical Token")
        ax.set_ylabel("Mean Attention Weight")
        ax.set_xlim(-0.5, n_clin_tokens - 0.5)
        ax.set_xticks(x_clin)
        ax.set_xticklabels(clin_labels, fontsize=10)

    fig.tight_layout()
    path3 = f"{out_dir}/attention_map_a2c.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: AEC→Clinical heatmap (Sarco → Normal 순) ─────────
    hmap_a2c = attn_a2c[ordered]
    figH = max(5, n_samp * 0.20)
    fig, ax = plt.subplots(figsize=(max(6, n_clin_tokens * 1.8 + 2), figH))
    im = ax.imshow(hmap_a2c, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   vmin=0, vmax=hmap_a2c.max())
    plt.colorbar(im, ax=ax, shrink=0.6, label="Attention Weight")
    if len(s_idx) > 0 and len(n_idx) > 0:
        ax.axhline(len(s_idx) - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.text(-0.3, len(s_idx) / 2 - 0.5,
            "Sarco",  va="center", ha="right",
            color="tomato",    fontsize=8, fontweight="bold", clip_on=False)
    ax.text(-0.3, len(s_idx) + len(n_idx) / 2 - 0.5,
            "Normal", va="center", ha="right",
            color="steelblue", fontsize=8, fontweight="bold", clip_on=False)
    ax.set_xlabel("Clinical Token")
    ax.set_ylabel("Sample")
    ax.set_title(
        f"Sample-level Attention Map (AEC → Clinical)  ·  {model_label}  [{aec_var}]",
        fontsize=10, fontweight="bold",
    )
    ax.set_xticks(x_clin)
    ax.set_xticklabels(clin_labels, fontsize=10)
    ax.set_yticks([])
    fig.tight_layout()
    path4 = f"{out_dir}/attention_heatmap_a2c.png"
    fig.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  {path1}")
    print(f"  {path2}")
    print(f"  {path3}")
    print(f"  {path4}")


def _compute_gradcam_aec(model, X_clin_s, X_aec_s, X_mfr=None):
    """
    ResNet1DEncoder의 마지막 ResBlock 출력에 Grad-CAM(Selvaraju et al. 2017)을 적용.

    분류 logit(Sarcopenia 방향)에 대한 기울기로 AEC 신호의 위치별 중요도를 산출한다.

    Parameters
    ----------
    model    : ClinAECCrossAttn / ClinAECScanCrossAttn
    X_clin_s : np.ndarray (N, n_clin)  — 스케일링된 임상 피처
    X_aec_s  : np.ndarray (N, n_aec)   — 스케일링된 AEC 피처
    X_mfr    : np.ndarray (N,) or None — Model 3 MFR 인덱스

    Returns
    -------
    cam : np.ndarray (N, n_aec_len)
        샘플별 0~1 정규화된 Grad-CAM 중요도 맵.
        값이 클수록 해당 AEC 위치가 분류에 기여한다.
        aec_encoder에 'blocks' 속성이 없으면 None 반환.
    """
    import torch
    from config import DEVICE

    if not hasattr(model, "aec_encoder") or not hasattr(model.aec_encoder, "blocks"):
        print("  [CAM] aec_encoder.blocks 없음 — Grad-CAM 미지원.")
        return None

    _activations = [None]
    _gradients   = [None]

    def _fwd_hook(*args):                    # args: (module, input, output)
        _activations[0] = args[2]           # (B, d_model, n_aec)

    def _bwd_hook(*args):                   # args: (module, grad_input, grad_output)
        _gradients[0] = args[2][0]          # (B, d_model, n_aec)

    last_block = model.aec_encoder.blocks[-1]
    fwd_h = last_block.register_forward_hook(_fwd_hook)
    bwd_h = last_block.register_full_backward_hook(_bwd_hook)

    model.eval()
    xc = torch.tensor(X_clin_s, dtype=torch.float32).to(DEVICE)
    xa = torch.tensor(X_aec_s,  dtype=torch.float32).to(DEVICE)

    try:
        if X_mfr is not None:
            xm = torch.tensor(X_mfr, dtype=torch.long).to(DEVICE)
            logits = model(xc, xa, xm)
        else:
            logits = model(xc, xa)

        model.zero_grad()
        # sum(logits) ← 각 샘플의 logit이 자신의 activation에만 의존하므로
        # d(sum)/d(act_i) == d(logit_i)/d(act_i) (per-sample 그래디언트)
        logits.sum().backward()
    finally:
        fwd_h.remove()
        bwd_h.remove()

    if _activations[0] is None or _gradients[0] is None:
        print("  [CAM] 훅이 트리거되지 않음 — Grad-CAM 계산 불가.")
        return None

    act  = _activations[0].detach().cpu().numpy()   # (B, d_model, n_aec)
    grad = _gradients[0].detach().cpu().numpy()      # (B, d_model, n_aec)

    # Grad-CAM: gradient를 공간 축(AEC 축)으로 평균 → 채널별 가중치
    weights = grad.mean(axis=-1, keepdims=True)      # (B, d_model, 1)
    cam     = (weights * act).sum(axis=1)            # (B, n_aec)
    cam     = np.maximum(cam, 0)                     # ReLU

    # 샘플별 0~1 정규화
    c_min = cam.min(axis=1, keepdims=True)
    c_max = cam.max(axis=1, keepdims=True)
    cam   = np.where(c_max > c_min,
                     (cam - c_min) / (c_max - c_min + 1e-8),
                     0.0)
    return cam                                       # (N, n_aec_len)


def plot_cam_aec(model, X_clin_te_s, X_aec_te_s, y_true_te,
                 out_dir, aec_var, model_label, X_mfr_te=None,
                 n_examples=5):
    """
    AEC 신호에 대한 Grad-CAM을 계산해 세 가지 시각화를 저장.

    출력 파일 (out_dir):
      cam_aec_mean.png     — 클래스별 평균 AEC ± std 위에 평균 CAM 배경 히트맵
      cam_aec_lines.png    — Figure-13 스타일: 클래스별 모든 샘플 선을 CAM 값으로
                             coloring (blue=낮음, red=높음, jet colormap, 0~100%)
      cam_aec_heatmap.png  — 전체 샘플 × AEC position 히트맵
                             (Sarco → Normal 순, attention heatmap 과 동일 형식)

    Parameters
    ----------
    model        : ClinAECCrossAttn / ClinAECScanCrossAttn
    X_clin_te_s  : np.ndarray (N, n_clin)
    X_aec_te_s   : np.ndarray (N, n_aec)
    y_true_te    : np.ndarray (N,)  — 정수 레이블 (0=Normal, 1=Sarco)
    out_dir      : str
    aec_var      : str  — AEC 변환 이름 (제목 표기용)
    model_label  : str
    X_mfr_te     : np.ndarray (N,) or None — Model 3 전용
    n_examples   : int  — cam_aec_examples.png 에 표시할 샘플 수
    """
    cam = _compute_gradcam_aec(model, X_clin_te_s, X_aec_te_s, X_mfr_te)
    if cam is None:
        return

    n_pts  = cam.shape[1]           # AEC 신호 포인트 수
    x_pts  = np.arange(n_pts)
    cls_info = [(0, "Normal", "steelblue"), (1, "Sarcopenia", "tomato")]

    s_idx   = np.where(y_true_te == 1)[0]
    n_idx   = np.where(y_true_te == 0)[0]
    ordered = np.concatenate([s_idx, n_idx])

    # ── Figure 1: Class-average CAM + mean AEC signal ─────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Grad-CAM — AEC Signal  ·  {model_label}  [{aec_var}]",
        fontsize=12, fontweight="bold",
    )

    for ax, (cls_idx, cls_name, color) in zip(axes, cls_info):
        mask = y_true_te == cls_idx
        if not mask.any():
            ax.set_visible(False)
            continue

        aec_cls  = X_aec_te_s[mask]    # (n_cls, n_aec)
        cam_cls  = cam[mask]             # (n_cls, n_aec)

        mean_aec = aec_cls.mean(0)
        std_aec  = aec_cls.std(0)
        mean_cam = cam_cls.mean(0)       # (n_aec,)

        pad = (mean_aec.max() - mean_aec.min()) * 0.15 + 0.05
        ymin = (mean_aec - std_aec).min() - pad
        ymax = (mean_aec + std_aec).max() + pad

        # CAM을 배경 히트맵으로 표시
        im = ax.imshow(
            mean_cam.reshape(1, -1),
            aspect="auto",
            extent=[-0.5, n_pts - 0.5, ymin, ymax],
            cmap="YlOrRd", alpha=0.55,
            vmin=0, vmax=1,
            origin="lower",
        )

        # 평균 AEC ± std
        ax.fill_between(x_pts, mean_aec - std_aec, mean_aec + std_aec,
                         color=color, alpha=0.18)
        ax.plot(x_pts, mean_aec, color=color, linewidth=2.0,
                label=f"Mean AEC (n={mask.sum()})")

        ax.set_xlim(-0.5, n_pts - 0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{cls_name}  (n={mask.sum()})", fontsize=10)
        ax.set_xlabel("AEC Position  (scan start → scan end)")
        ax.set_ylabel("Scaled AEC Value")
        ax.legend(fontsize=8)
        plt.colorbar(im, ax=ax, label="Grad-CAM (mean)", shrink=0.6, pad=0.01)

    fig.tight_layout()
    path1 = f"{out_dir}/cam_aec_mean.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: 클래스별 샘플 10개 — 샘플마다 고유색 ──────────────────────
    _N_LINES   = 10
    _line_cmap = plt.colormaps["tab10"]
    _line_colors = [_line_cmap(i) for i in range(_N_LINES)]
    _rng = np.random.default_rng(42)

    x_f = x_pts.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(
        f"Class Activation Map — AEC  ·  {model_label}  [{aec_var}]",
        fontsize=12, fontweight="bold",
    )

    for ax, (cls_idx, cls_name, _) in zip(axes, cls_info):
        mask_idx = np.where(y_true_te == cls_idx)[0]
        if len(mask_idx) == 0:
            ax.set_visible(False)
            continue

        # 클래스당 최대 _N_LINES 개 무작위 선택 (재현성 seed=42)
        if len(mask_idx) > _N_LINES:
            sel_idx = _rng.choice(mask_idx, _N_LINES, replace=False)
            sel_idx = np.sort(sel_idx)
        else:
            sel_idx = mask_idx

        aec_sel = X_aec_te_s[sel_idx]   # (n_sel, n_aec)

        for i, color in enumerate(zip(sel_idx, _line_colors)):
            _, c = color
            ax.plot(x_f, aec_sel[i].astype(float),
                    color=c, linewidth=1.2, alpha=0.85, label=f"S{i + 1}")

        pad = (aec_sel.max() - aec_sel.min()) * 0.08 + 0.05
        ax.set_xlim(x_f[0] - 0.5, x_f[-1] + 0.5)
        ax.set_ylim(aec_sel.min() - pad, aec_sel.max() + pad)
        ax.set_title(
            f"{cls_name}  ({len(sel_idx)} of {len(mask_idx)} samples)", fontsize=10
        )
        ax.set_xlabel("AEC Position  (scan start → scan end)", fontsize=9)
        ax.set_ylabel("Scaled AEC Value", fontsize=9)
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    path2 = f"{out_dir}/cam_aec_lines.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Sample-level heatmap (Sarco → Normal 정렬) ──────────────
    hmap = cam[ordered]                   # (N, n_aec)
    n_samp = len(ordered)
    fig_h  = max(5, n_samp * 0.20)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    im = ax.imshow(hmap, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Grad-CAM")

    if len(s_idx) > 0 and len(n_idx) > 0:
        ax.axhline(len(s_idx) - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.text(-0.8, len(s_idx) / 2 - 0.5,
            "Sarco",  va="center", ha="right",
            color="tomato",    fontsize=8, fontweight="bold", clip_on=False)
    ax.text(-0.8, len(s_idx) + len(n_idx) / 2 - 0.5,
            "Normal", va="center", ha="right",
            color="steelblue", fontsize=8, fontweight="bold", clip_on=False)

    tick_step = max(1, n_pts // 8)
    ax.set_xticks(x_pts[::tick_step])
    ax.set_xticklabels(x_pts[::tick_step])
    ax.set_yticks([])
    ax.set_xlabel("AEC Position  (scan start → scan end)")
    ax.set_ylabel("Sample")
    ax.set_title(
        f"Sample-level Grad-CAM Heatmap  ·  {model_label}  [{aec_var}]",
        fontsize=10, fontweight="bold",
    )

    fig.tight_layout()
    path3 = f"{out_dir}/cam_aec_heatmap.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  {path1}")   # cam_aec_mean.png
    print(f"  {path2}")   # cam_aec_lines.png  (Figure-13 style)
    print(f"  {path3}")   # cam_aec_heatmap.png


def plot_individual_aec_normalization(X_aec_raw_cv, X_aec_raw_te, y_te, sex_te,
                                      out_dir, n_per_class=3, seed=42):
    """
    환자 개별 AEC 신호 — 정규화 전후 비교 그래프.

    test set에서 (진단 × 성별) 4그룹별 n_per_class명을 무작위 선택해 4가지 정규화
    (① Raw / ② Column-wise / ③ Row-wise / ④ Global Z-score)를 나란히 시각화.
    Column-wise와 Global Z-score는 train(CV) set으로 scaler를 fit한다.

    색상 체계:
      Blues 계열 (진한→연) = Male,   실선 = Sarcopenia
      Reds  계열 (진한→연) = Female, 점선 = Normal

    Parameters
    ----------
    X_aec_raw_cv : (N_cv, P)  — 정규화 전 원본 AEC (train/CV set)
    X_aec_raw_te : (N_te, P)  — 정규화 전 원본 AEC (test set)
    y_te         : (N_te,)    — test set 레이블 (0=Normal, 1=Sarco)
    sex_te       : (N_te,)    — test set 성별 ("M" / "F")
    out_dir      : str
    n_per_class  : int         — 그룹별 표시 환자 수
    """
    from sklearn.preprocessing import StandardScaler

    rng   = np.random.default_rng(seed)
    n_aec = X_aec_raw_te.shape[1]
    x_pos = np.arange(n_aec)

    sc_col = StandardScaler().fit(X_aec_raw_cv)
    g_mean = float(X_aec_raw_cv.mean())
    g_std  = max(float(X_aec_raw_cv.std()), 1e-8)

    def _row_norm(X):
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-8
        return (X - mu) / sd

    transforms = [
        ("① Raw\n(전처리 없음)",                    lambda X: X),
        ("② Column-wise\n(StandardScaler, 열 방향)", lambda X: sc_col.transform(X)),
        ("③ Row-wise\n(환자별 z-score, 행 방향)",     _row_norm),
        ("④ Global Z-score\n(Train 전체 단일 μ/σ)",  lambda X: (X - g_mean) / g_std),
    ]

    # ── 4그룹 정의: (y, sex, 레이블, colormap, 선스타일, 색조 범위) ──
    _cmap_blues = plt.colormaps["Blues"]
    _cmap_reds  = plt.colormaps["Reds"]
    group_defs = [
        (1, "M", "Sarco-M",  _cmap_blues, "-",  (0.55, 0.95)),
        (1, "F", "Sarco-F",  _cmap_reds,  "-",  (0.55, 0.95)),
        (0, "M", "Normal-M", _cmap_blues, "--", (0.30, 0.55)),
        (0, "F", "Normal-F", _cmap_reds,  "--", (0.30, 0.55)),
    ]

    sex_arr = np.asarray(sex_te)
    sel_idx_all, labels_all, colors_all, ls_all = [], [], [], []
    for y_val, sex_val, grp_lbl, cmap_g, ls, (clo, chi) in group_defs:
        mask = (y_te == y_val) & (sex_arr == sex_val)
        pool = np.where(mask)[0]
        n = min(n_per_class, len(pool))
        if n == 0:
            continue
        chosen = np.sort(rng.choice(pool, n, replace=False))
        shades = np.linspace(clo, chi, n) if n > 1 else [(clo + chi) / 2]
        for j, idx in enumerate(chosen):
            sel_idx_all.append(idx)
            labels_all.append(f"{grp_lbl} #{j + 1}")
            colors_all.append(cmap_g(shades[j]))
            ls_all.append(ls)

    n_plots = len(transforms)
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 6.5, 5.5), sharey=False)
    fig.suptitle(
        "환자 개별 AEC 신호 — 정규화 전후 비교\n"
        "실선=Sarcopenia  점선=Normal  │  파랑=Male  빨강=Female",
        fontsize=12, fontweight="bold",
    )

    for ax, (title, tfm) in zip(axes, transforms):
        X_tfm = tfm(X_aec_raw_te.astype(np.float64)).astype(np.float32)
        for idx, lbl, col, ls in zip(sel_idx_all, labels_all, colors_all, ls_all):
            ax.plot(x_pos, X_tfm[idx], color=col, linewidth=1.4, linestyle=ls,
                    alpha=0.88, label=lbl)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("AEC Position\n(scan start → scan end)", fontsize=8)
        ax.set_ylabel("AEC Value", fontsize=8)
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(-0.5, n_aec - 0.5)

    fig.tight_layout()
    out_path = f"{out_dir}/aec_individual_normalization_compare.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")

    # ── Raw AEC 데이터 Excel 저장 ──────────────────────────────
    import pandas as pd
    X_raw_sel = X_aec_raw_te[sel_idx_all].astype(np.float64)
    pos_cols   = [f"pos_{i}" for i in range(1, n_aec + 1)]
    df_raw     = pd.DataFrame(X_raw_sel, columns=pos_cols)
    df_raw.insert(0, "group_label", labels_all)
    df_raw.insert(1, "y_true",      [int(y_te[i]) for i in sel_idx_all])
    df_raw.insert(2, "sex",         [sex_te[i]    for i in sel_idx_all])
    xl_path    = f"{out_dir}/aec_individual_normalization_compare_raw.xlsx"
    df_raw.to_excel(xl_path, index=False)
    print(f"  {xl_path}")

    return out_path


def save_all_cross(ca_cv, ca_roc_folds, ca_histories, med_epoch,
                   X_clin_cv, y_cv, sex_cv,
                   X_clin_te, y_te,
                   ca_pred_te, ca_true_te, sex_te, ca_prob_te,
                   model_label="model 2", out_dir=None, ci_dict=None):
    """Model 2/2_2/3용 시각화 전체(8종 png)와 results.md를 out_dir에 저장."""
    global _dir2
    _dir2 = out_dir or RESULTS_MODEL_2_DIR
    plot_data_distribution(X_clin_cv, y_cv, sex_cv, X_clin_te, y_te, sex_te,
                           out_dir=_dir2)
    plot_cv_roc_cross(ca_roc_folds)
    plot_cv_metric_cross(ca_cv)
    plot_training_curves_cross(ca_histories, med_epoch)
    plot_test_roc_cross(ca_true_te, ca_prob_te)
    plot_test_roc_by_sex_cross(ca_true_te, ca_prob_te, sex_te)
    plot_confusion_matrices_cross(ca_true_te, ca_pred_te, sex_te)
    plot_calibration(
        [("CrossAttn", ca_true_te, ca_prob_te, "steelblue")],
        out_path=f"{_dir2}/calibration_.png",
    )

    print(f"\nSaved ({model_label}):")
    for fname in ["data_distribution", "cv_roc_curves", "cv_metric_distribution",
                  "training_curves", "test_roc_curves", "test_roc_by_sex",
                  "confusion_matrices", "calibration"]:
        print(f"  {_dir2}/{fname}.png")

    _save_report_md_cross(ca_cv, X_clin_cv, y_cv, sex_cv, X_clin_te, y_te,
                          ca_pred_te, ca_prob_te, ca_true_te,
                          sex_te, ca_histories, med_epoch, ci_dict=ci_dict)


# ── Model 4: AEC Only ───────────────────────────────────────

def plot_label_distribution(y_cv, sex_cv, y_te, sex_te, out_dir=None):
    """레이블·성별 분포 그래프 (AEC-only 모델용 — clinical 특징 미포함)."""
    cls_colors = {"Normal": "steelblue", "Sarco": "tomato"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle("Dataset Distribution — Train (CV) vs Test  [AEC Only]",
                 fontsize=13, fontweight="bold")
    for ax, (y, sex, name) in zip(axes, [
        (y_cv, sex_cv, "Train (CV)"),
        (y_te, sex_te, "Test"),
    ]):
        n_split = len(y)
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
        ax.set_title(f"{name}  (n={n_split})")
        ax.set_ylabel("Proportion within sex group")
        ax.legend()
    fig.tight_layout()
    save_dir = out_dir or RESULTS_DIR
    fig.savefig(f"{save_dir}/label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_report_md_aec_only(cv, y_cv, sex_cv, y_te,
                              pred_te, prob_te, true_te,
                              sex_te, histories, med_epoch, ci_dict=None):
    """AECOnly CV 결과와 test set 성능 지표를 results.md로 저장 (Model 4 전용)."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    auc_arr = np.array([h["val_auc"] for h in histories])
    best_val_aucs = auc_arr.max(axis=1)

    lines = [
        "# SMI Binary Classification — AECOnly Results",
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
        "![Label Distribution](label_distribution.png)",
        "",
        "---",
        "",
        "## 1. Cross-Validation Summary",
        "",
        "### AECOnly",
        "",
        _cv_table(cv),
        "",
        "AECOnly best val AUC per fold: " +
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
        f"| AECOnly | {roc_auc_score(true_te, prob_te):.4f}"
        f" | {average_precision_score(true_te, prob_te):.4f}"
        f" | {brier_score_loss(true_te, prob_te):.4f}"
        f" | {accuracy_score(true_te, pred_te):.4f}"
        f" | {f1_score(true_te, pred_te, zero_division=0):.4f} |",
        "",
        "### By Sex",
        "",
        "| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
        "|-----|--:|--------:|------:|------:|---------:|---:|",
        _sex_rows(true_te, pred_te, prob_te, sex_te),
        "",
        "---",
        "",
        "## 3. Confusion Matrix (Test Set)",
        "",
        _cm_block(true_te, pred_te),
        "",
        "---",
        "",
        *_ci_section(ci_dict),
        "",
        "---",
        "",
        "## 5. Figures",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `label_distribution.png` | Train/Test class·sex distributions |",
        "| `cv_roc_curves.png` | Per-fold ROC curves (AECOnly) |",
        "| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |",
        "| `training_curves.png` | Loss & AUC training curves (mean ± std) |",
        "| `test_roc_curves.png` | Final test-set ROC curve |",
        "| `test_roc_by_sex.png` | Final test-set ROC curves by sex |",
        "| `confusion_matrices.png` | Test-set confusion matrices |",
        "| `calibration.png` | Calibration plot + Precision-Recall curve |",
        "| `cam_aec_mean.png` | Grad-CAM mean ± std per class |",
        "| `cam_aec_lines.png` | Grad-CAM individual samples per class |",
        "| `cam_aec_heatmap.png` | Grad-CAM sample-level heatmap |",
    ]

    md_path = f"{_dir2}/results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {md_path}")


def save_all_aec_only(cv, roc_folds, histories, med_epoch,
                      y_cv, sex_cv, y_te, true_te,
                      pred_te, prob_te, sex_te,
                      model_label="model 4", out_dir=None, ci_dict=None):
    """Model 4(AEC Only)용 시각화 전체(8종 png)와 results.md를 out_dir에 저장."""
    global _dir2
    _dir2 = out_dir or RESULTS_DIR
    plot_label_distribution(y_cv, sex_cv, y_te, sex_te, out_dir=_dir2)
    plot_cv_roc_cross(roc_folds, model_label="AECOnly")
    plot_cv_metric_cross(cv, model_label="AECOnly")
    plot_training_curves_cross(histories, med_epoch, model_label="AECOnly")
    plot_test_roc_cross(true_te, prob_te, model_label="AECOnly")
    plot_test_roc_by_sex_cross(true_te, prob_te, sex_te, model_label="AECOnly")
    plot_confusion_matrices_cross(true_te, pred_te, sex_te, model_label="AECOnly")
    plot_calibration(
        [("AECOnly", true_te, prob_te, "mediumpurple")],
        out_path=f"{_dir2}/calibration_.png",
    )

    print(f"\nSaved ({model_label}):")
    for fname in ["label_distribution", "cv_roc_curves", "cv_metric_distribution",
                  "training_curves", "test_roc_curves", "test_roc_by_sex",
                  "confusion_matrices", "calibration"]:
        print(f"  {_dir2}/{fname}.png")

    _save_report_md_aec_only(cv, y_cv, sex_cv, y_te,
                             pred_te, prob_te, true_te,
                             sex_te, histories, med_epoch, ci_dict=ci_dict)


def _compute_gradcam_aec_only(model, X_aec_s):
    """
    AECOnlyNet의 마지막 ResBlock에 Grad-CAM(Selvaraju et al. 2017)을 적용.

    Returns
    -------
    cam : np.ndarray (N, n_aec) — 샘플별 0~1 정규화된 Grad-CAM 중요도 맵 또는 None.
    """
    import torch
    from config import DEVICE

    if not hasattr(model, "blocks"):
        print("  [CAM] model.blocks 없음 — AECOnly Grad-CAM 미지원.")
        return None

    _activations = [None]
    _gradients   = [None]

    def _fwd_hook(*args):
        _activations[0] = args[2]

    def _bwd_hook(*args):
        _gradients[0] = args[2][0]

    last_block = model.blocks[-1]
    fwd_h = last_block.register_forward_hook(_fwd_hook)
    bwd_h = last_block.register_full_backward_hook(_bwd_hook)

    model.eval()
    xa = torch.tensor(X_aec_s, dtype=torch.float32).to(DEVICE)

    try:
        logits = model(xa)
        model.zero_grad()
        logits.sum().backward()
    finally:
        fwd_h.remove()
        bwd_h.remove()

    if _activations[0] is None or _gradients[0] is None:
        print("  [CAM] 훅이 트리거되지 않음 — Grad-CAM 계산 불가.")
        return None

    act  = _activations[0].detach().cpu().numpy()   # (B, d_model, n_aec)
    grad = _gradients[0].detach().cpu().numpy()      # (B, d_model, n_aec)

    weights = grad.mean(axis=-1, keepdims=True)      # (B, d_model, 1)
    cam     = (weights * act).sum(axis=1)            # (B, n_aec)
    cam     = np.maximum(cam, 0)

    c_min = cam.min(axis=1, keepdims=True)
    c_max = cam.max(axis=1, keepdims=True)
    cam   = np.where(c_max > c_min,
                     (cam - c_min) / (c_max - c_min + 1e-8),
                     0.0)
    return cam


def plot_cam_aec_only(model, X_aec_te_s, y_true_te,
                      out_dir, aec_var, model_label):
    """
    AECOnlyNet의 Grad-CAM을 계산해 세 가지 시각화를 저장.

    출력 파일 (out_dir):
      cam_aec_mean.png     — 클래스별 평균 AEC ± std + 평균 CAM 배경 히트맵
      cam_aec_lines.png    — 클래스별 샘플 선 시각화
      cam_aec_heatmap.png  — 전체 샘플 × AEC position 히트맵
    """
    cam = _compute_gradcam_aec_only(model, X_aec_te_s)
    if cam is None:
        return

    n_pts    = cam.shape[1]
    x_pts    = np.arange(n_pts)
    cls_info = [(0, "Normal", "steelblue"), (1, "Sarcopenia", "tomato")]

    s_idx   = np.where(y_true_te == 1)[0]
    n_idx   = np.where(y_true_te == 0)[0]
    ordered = np.concatenate([s_idx, n_idx])

    # ── Figure 1: Class-average CAM + mean AEC signal ────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Grad-CAM — AEC Signal  ·  {model_label}  [{aec_var}]",
        fontsize=12, fontweight="bold",
    )
    for ax, (cls_idx, cls_name, color) in zip(axes, cls_info):
        mask = y_true_te == cls_idx
        if not mask.any():
            ax.set_visible(False)
            continue
        aec_cls  = X_aec_te_s[mask]
        cam_cls  = cam[mask]
        mean_aec = aec_cls.mean(0)
        std_aec  = aec_cls.std(0)
        mean_cam = cam_cls.mean(0)
        pad  = (mean_aec.max() - mean_aec.min()) * 0.15 + 0.05
        ymin = (mean_aec - std_aec).min() - pad
        ymax = (mean_aec + std_aec).max() + pad
        im = ax.imshow(
            mean_cam.reshape(1, -1), aspect="auto",
            extent=[-0.5, n_pts - 0.5, ymin, ymax],
            cmap="YlOrRd", alpha=0.55, vmin=0, vmax=1, origin="lower",
        )
        ax.fill_between(x_pts, mean_aec - std_aec, mean_aec + std_aec,
                        color=color, alpha=0.18)
        ax.plot(x_pts, mean_aec, color=color, linewidth=2.0,
                label=f"Mean AEC (n={mask.sum()})")
        ax.set_xlim(-0.5, n_pts - 0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{cls_name}  (n={mask.sum()})", fontsize=10)
        ax.set_xlabel("AEC Position  (scan start → scan end)")
        ax.set_ylabel("Scaled AEC Value")
        ax.legend(fontsize=8)
        plt.colorbar(im, ax=ax, label="Grad-CAM (mean)", shrink=0.6, pad=0.01)
    fig.tight_layout()
    path1 = f"{out_dir}/cam_aec_mean.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: 클래스별 샘플 선 시각화 ────────────────────
    _N_LINES    = 10
    _line_cmap  = plt.colormaps["tab10"]
    _line_colors = [_line_cmap(i) for i in range(_N_LINES)]
    _rng = np.random.default_rng(42)
    x_f  = x_pts.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(
        f"Class Activation Map — AEC  ·  {model_label}  [{aec_var}]",
        fontsize=12, fontweight="bold",
    )
    for ax, (cls_idx, cls_name, _) in zip(axes, cls_info):
        mask_idx = np.where(y_true_te == cls_idx)[0]
        if len(mask_idx) == 0:
            ax.set_visible(False)
            continue
        if len(mask_idx) > _N_LINES:
            sel_idx = _rng.choice(mask_idx, _N_LINES, replace=False)
            sel_idx = np.sort(sel_idx)
        else:
            sel_idx = mask_idx
        aec_sel = X_aec_te_s[sel_idx]
        for i, (_, c) in enumerate(zip(sel_idx, _line_colors)):
            ax.plot(x_f, aec_sel[i].astype(float),
                    color=c, linewidth=1.2, alpha=0.85, label=f"S{i + 1}")
        pad = (aec_sel.max() - aec_sel.min()) * 0.08 + 0.05
        ax.set_xlim(x_f[0] - 0.5, x_f[-1] + 0.5)
        ax.set_ylim(aec_sel.min() - pad, aec_sel.max() + pad)
        ax.set_title(f"{cls_name}  ({len(sel_idx)} of {len(mask_idx)} samples)", fontsize=10)
        ax.set_xlabel("AEC Position  (scan start → scan end)", fontsize=9)
        ax.set_ylabel("Scaled AEC Value", fontsize=9)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    path2 = f"{out_dir}/cam_aec_lines.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Sample-level heatmap (Sarco → Normal 정렬) ─
    hmap   = cam[ordered]
    n_samp = len(ordered)
    fig_h  = max(5, n_samp * 0.20)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    im = ax.imshow(hmap, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Grad-CAM")
    if len(s_idx) > 0 and len(n_idx) > 0:
        ax.axhline(len(s_idx) - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.text(-0.8, len(s_idx) / 2 - 0.5,
            "Sarco",  va="center", ha="right",
            color="tomato",    fontsize=8, fontweight="bold", clip_on=False)
    ax.text(-0.8, len(s_idx) + len(n_idx) / 2 - 0.5,
            "Normal", va="center", ha="right",
            color="steelblue", fontsize=8, fontweight="bold", clip_on=False)
    tick_step = max(1, n_pts // 8)
    ax.set_xticks(x_pts[::tick_step])
    ax.set_xticklabels(x_pts[::tick_step])
    ax.set_yticks([])
    ax.set_xlabel("AEC Position  (scan start → scan end)")
    ax.set_ylabel("Sample")
    ax.set_title(
        f"Sample-level Grad-CAM Heatmap  ·  {model_label}  [{aec_var}]",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    path3 = f"{out_dir}/cam_aec_heatmap.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  {path1}")
    print(f"  {path2}")
    print(f"  {path3}")
