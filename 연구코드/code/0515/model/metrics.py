import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    average_precision_score, brier_score_loss,
)

from config import N_FOLDS


def group_metrics(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return dict(
            acc=accuracy_score(y_true, y_pred),
            auc=float("nan"),
            auprc=float("nan"),
            brier=float("nan"),
            f1=f1_score(y_true, y_pred, zero_division=0),
        )
    return dict(
        acc=accuracy_score(y_true, y_pred),
        auc=roc_auc_score(y_true, y_prob),
        auprc=average_precision_score(y_true, y_prob),
        brier=brier_score_loss(y_true, y_prob),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )


def print_cv_summary(name, fold_metrics):
    aucs   = [m["auc"]   for m in fold_metrics]
    auprcs = [m["auprc"] for m in fold_metrics]
    briers = [m["brier"] for m in fold_metrics]
    accs   = [m["acc"]   for m in fold_metrics]
    f1s    = [m["f1"]    for m in fold_metrics]
    print(f"\n{'='*75}")
    print(f"{name}  —  {N_FOLDS}-Fold CV Summary")
    print(f"{'='*75}")
    print(f"  {'Fold':<6} {'AUC-ROC':>10} {'AUPRC':>10} {'Brier':>10} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*58}")
    for m in fold_metrics:
        print(f"  {m['fold']:<6} {m['auc']:>10.4f} {m['auprc']:>10.4f} {m['brier']:>10.4f}"
              f" {m['acc']:>10.4f} {m['f1']:>10.4f}")
    print(f"  {'─'*58}")
    print(f"  {'Mean':<6} {np.mean(aucs):>10.4f} {np.mean(auprcs):>10.4f} {np.mean(briers):>10.4f}"
          f" {np.mean(accs):>10.4f} {np.mean(f1s):>10.4f}")
    print(f"  {'±Std':<6} {np.std(aucs):>10.4f} {np.std(auprcs):>10.4f} {np.std(briers):>10.4f}"
          f" {np.std(accs):>10.4f} {np.std(f1s):>10.4f}")


def compare_fold_metrics(name1: str, fold_metrics1: list,
                         name2: str, fold_metrics2: list) -> dict:
    """
    Fold별 성능(AUC, Acc, F1)에 대해 paired t-test와 Wilcoxon signed-rank test를 수행.

    - Paired t-test (scipy.stats.ttest_rel): 정규분포 가정
    - Wilcoxon signed-rank test: 비모수, n=5처럼 샘플이 적을 때 보완적으로 사용
    - Δ Mean = name2 - name1 (양수면 name2가 더 높음)
    """
    n = len(fold_metrics1)
    metric_keys   = ["auc",     "auprc", "brier", "acc",      "f1"]
    metric_labels = ["AUC-ROC", "AUPRC", "Brier", "Accuracy", "F1"]

    print(f"\n{'='*65}")
    print(f"  Statistical Comparison: [{name1}]  vs  [{name2}]")
    print(f"  Paired t-test + Wilcoxon signed-rank  (n={n} folds)")
    print(f"  Δ Mean = {name2} − {name1}  (양수 → {name2}가 우세)")
    print(f"{'='*65}")
    print(f"  {'Metric':<12} {'Mean1':>8} {'Mean2':>8} {'Δ Mean':>9}"
          f" {'t-stat':>8} {'t p-val':>12} {'W p-val':>12} {'':>4}")
    print(f"  {'-'*71}")

    results = {}
    for key, label in zip(metric_keys, metric_labels):
        v1 = np.array([m[key] for m in fold_metrics1], dtype=float)
        v2 = np.array([m[key] for m in fold_metrics2], dtype=float)
        diff = v2 - v1

        t_stat, t_pval = stats.ttest_rel(v1, v2)

        try:
            _, w_pval = stats.wilcoxon(diff)
        except ValueError:
            # diff가 모두 0이면 wilcoxon 불가
            w_pval = float("nan")

        sig = ""
        if t_pval < 0.001:
            sig = "***"
        elif t_pval < 0.01:
            sig = "**"
        elif t_pval < 0.05:
            sig = "*"
        elif t_pval < 0.10:
            sig = "†"

        print(f"  {label:<12} {v1.mean():>8.4f} {v2.mean():>8.4f} {diff.mean():>+9.4f}"
              f" {t_stat:>8.3f} {t_pval:>12.2e} {w_pval:>12.2e} {sig:>4}")

        results[key] = {
            "mean1": float(v1.mean()), "mean2": float(v2.mean()),
            "delta_mean": float(diff.mean()),
            "t_stat": float(t_stat), "t_pval": float(t_pval),
            "w_pval": float(w_pval),
        }

    print(f"  {'─'*71}")
    print(f"  *** p<0.001  ** p<0.01  * p<0.05  † p<0.10")
    print(f"  주의: n={n} folds — 검정력이 낮으므로 방향성(Δ) 참고 병행 권장")
    return results
