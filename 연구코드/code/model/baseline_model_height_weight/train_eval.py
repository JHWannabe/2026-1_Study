import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, f1_score,
    classification_report, average_precision_score, brier_score_loss,
)
from config import N_FOLDS, SEED
from metrics import group_metrics, bootstrap_ci_md

def run_cross_validation(X_cv, y_cv):
    """LR 5-Fold CV. (lr_cv, lr_roc_folds, lr_best_thresholds, md_text) 반환."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    lr_cv, lr_roc_folds, lr_best_thresholds = [], [], []
    fold_lines = [
        f"## LR — {N_FOLDS}-Fold CV Summary\n",
        "| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 | Threshold |",
        "|------|---------|-------|-------|----------|----|-----------|",
    ]

    for fold, (tr_i, val_i) in enumerate(skf.split(X_cv, y_cv), 1):
        sc = StandardScaler()
        Xtr_s, Xval_s = X_cv[tr_i].copy(), X_cv[val_i].copy()
        Xtr_s[:, [0, 2]]  = sc.fit_transform(X_cv[tr_i][:, [0, 2]])
        Xval_s[:, [0, 2]] = sc.transform(X_cv[val_i][:, [0, 2]])
        y_ftr, y_fval = y_cv[tr_i], y_cv[val_i]

        lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        lr.fit(Xtr_s, y_ftr)
        lr_prob = lr.predict_proba(Xval_s)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_fval, lr_prob)
        best_thresh = float(thresholds[np.argmax(tpr - fpr)])
        lr_pred     = (lr_prob >= best_thresh).astype(int)
        lr_best_thresholds.append(best_thresh)

        m = group_metrics(y_fval, lr_pred, lr_prob)
        lr_cv.append({"fold": fold, **m})
        lr_roc_folds.append({"fpr": fpr, "tpr": tpr, "auc": m["auc"]})

        fold_lines.append(
            f"| {fold} | {m['auc']:.4f} | {m['auprc']:.4f} |"
            f" {m['brier']:.4f} | {m['acc']:.4f} | {m['f1']:.4f} | {best_thresh:.3f} |"
        )

    keys = ["auc", "auprc", "brier", "acc", "f1"]
    vals = {k: [m[k] for m in lr_cv] for k in keys}
    fold_lines.append(
        f"| **Mean** | {np.mean(vals['auc']):.4f} | {np.mean(vals['auprc']):.4f} |"
        f" {np.mean(vals['brier']):.4f} | {np.mean(vals['acc']):.4f} | {np.mean(vals['f1']):.4f} | — |"
    )
    fold_lines.append(
        f"| **±Std** | {np.std(vals['auc']):.4f} | {np.std(vals['auprc']):.4f} |"
        f" {np.std(vals['brier']):.4f} | {np.std(vals['acc']):.4f} | {np.std(vals['f1']):.4f} | — |"
    )

    return lr_cv, lr_roc_folds, lr_best_thresholds, "\n".join(fold_lines)

def evaluate_test(X_cv, y_cv, X_te, y_te, sex_te, threshold=0.5):
    """전체 CV 세트로 LR 최종 학습 후 test set 평가. (lr_pred, lr_prob, ci_dict, md_text) 반환."""
    sc = StandardScaler()
    X_cv_s, X_te_s = X_cv.copy(), X_te.copy()
    X_cv_s[:, [0, 2]] = sc.fit_transform(X_cv[:, [0, 2]])
    X_te_s[:, [0, 2]] = sc.transform(X_te[:, [0, 2]])

    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr.fit(X_cv_s, y_cv)
    lr_prob = lr.predict_proba(X_te_s)[:, 1]
    lr_pred = (lr_prob >= threshold).astype(int)

    lines = [f"## Test Set Evaluation\n\n**Threshold:** {threshold:.3f}\n"]
    lines += [
        "### Overall\n",
        "| AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
        "|---------|-------|-------|----------|----|",
        f"| {roc_auc_score(y_te, lr_prob):.4f}"
        f" | {average_precision_score(y_te, lr_prob):.4f}"
        f" | {brier_score_loss(y_te, lr_prob):.4f}"
        f" | {accuracy_score(y_te, lr_pred):.4f}"
        f" | {f1_score(y_te, lr_pred, zero_division=0):.4f} |",
    ]

    for s in ["M", "F"]:
        mask = sex_te == s
        if not mask.any():
            continue
        yt, yp, ypr = y_te[mask], lr_pred[mask], lr_prob[mask]
        has_both = len(np.unique(yt)) > 1
        auc   = roc_auc_score(yt, ypr)           if has_both else float("nan")
        auprc = average_precision_score(yt, ypr) if has_both else float("nan")
        brier = brier_score_loss(yt, ypr)        if has_both else float("nan")
        sex_label = "Male" if s == "M" else "Female"
        lines += [
            f"\n### {sex_label} (n={mask.sum()})\n",
            "| AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
            "|---------|-------|-------|----------|----|",
            f"| {auc:.4f} | {auprc:.4f} | {brier:.4f}"
            f" | {accuracy_score(yt, yp):.4f} | {f1_score(yt, yp, zero_division=0):.4f} |",
            f"\n```\n{classification_report(yt, yp, target_names=['Normal', 'Sarcopenia'], zero_division=0)}```",
        ]

    ci, ci_text = bootstrap_ci_md("LR", y_te, lr_pred, lr_prob)
    lines.append("\n" + ci_text)

    return lr_pred, lr_prob, ci, "\n".join(lines)
