"""
CV 성능 지표 계산·출력 유틸리티.

  group_metrics            — 단일 fold/split 에서 5개 지표(Acc·AUC·AUPRC·Brier·F1) dict 계산
  print_cv_summary         — fold별 결과 콘솔 테이블 + mean±std 출력
  compare_fold_metrics     — Paired t-test + Wilcoxon signed-rank 통계 검정 (n=5 folds)
  delong_test              — DeLong (1988) ROC AUC 비교 검정 (동일 test set, 두 모델)
  bootstrap_ci             — Bootstrap 95% CI (ACC·AUC·AUPRC·Brier·F1)
  print_bootstrap_ci       — Bootstrap CI 콘솔 테이블 출력
  print_delong_comparison  — DeLong test 결과 콘솔 출력
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    average_precision_score, brier_score_loss,
)

from config import N_FOLDS


def group_metrics(y_true, y_pred, y_prob):
    """ACC·AUC·AUPRC·Brier·F1 dict 반환. 클래스가 1개뿐이면 AUC 관련 항목은 nan."""
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
    """fold별 성능 지표(AUC·AUPRC·Brier·Acc·F1)와 mean±std를 콘솔 테이블로 출력."""
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

        ttest_res = stats.ttest_rel(v1, v2)
        t_stat = float(ttest_res.statistic)
        t_pval = float(ttest_res.pvalue)

        try:
            _, w_pval_raw = stats.wilcoxon(diff)
            w_pval = float(w_pval_raw)  # type: ignore[arg-type]
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


# ── DeLong test + Bootstrap CI ───────────────────────────────────────────────

def _v10_v01(pos, neg):
    """Vectorized structural components for DeLong test."""
    diff   = pos[:, None] - neg[None, :]              # (n_pos, n_neg)
    kernel = (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)
    return kernel.mean(axis=1), kernel.mean(axis=0)   # V10, V01


def delong_test(y_true, y_prob1, y_prob2):
    """DeLong (1988) test — 동일 test set에서 두 모델의 ROC AUC를 비교.

    Returns: (auc1, auc2, z_stat, p_val). 퇴화 입력이면 모두 nan.
    """
    y_true  = np.asarray(y_true)
    y_prob1 = np.asarray(y_prob1, dtype=float)
    y_prob2 = np.asarray(y_prob2, dtype=float)

    pos1 = y_prob1[y_true == 1];  neg1 = y_prob1[y_true == 0]
    pos2 = y_prob2[y_true == 1];  neg2 = y_prob2[y_true == 0]
    n_pos, n_neg = len(pos1), len(neg1)

    if n_pos < 2 or n_neg < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    V10_1, V01_1 = _v10_v01(pos1, neg1)
    V10_2, V01_2 = _v10_v01(pos2, neg2)

    auc1, auc2 = float(V10_1.mean()), float(V10_2.mean())

    cov10    = np.cov(V10_1, V10_2)
    cov01    = np.cov(V01_1, V01_2)
    var_diff = (cov10[0, 0] + cov10[1, 1] - 2 * cov10[0, 1]) / n_pos \
             + (cov01[0, 0] + cov01[1, 1] - 2 * cov01[0, 1]) / n_neg

    if var_diff <= 0:
        return auc1, auc2, float("nan"), float("nan")

    z = float((auc1 - auc2) / np.sqrt(var_diff))
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return auc1, auc2, z, p


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=2000, seed=42):
    """Bootstrap 95% CI for ACC, AUC, AUPRC, Brier, F1 on a test set.

    Returns dict: {metric_key: (point_est, ci_lower, ci_upper)}.
    point_est는 원본 데이터 기준. 각 bootstrap sample에서 클래스가 1개뿐이면 skip.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    boot: dict = {k: [] for k in ["acc", "auc", "auprc", "brier", "f1"]}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot["acc"].append(accuracy_score(yt, yp))
        boot["auc"].append(roc_auc_score(yt, ypr))
        boot["auprc"].append(average_precision_score(yt, ypr))
        boot["brier"].append(brier_score_loss(yt, ypr))
        boot["f1"].append(f1_score(yt, yp, zero_division=0))

    has_both = len(np.unique(y_true)) >= 2
    point = {
        "acc":   float(accuracy_score(y_true, y_pred)),
        "auc":   float(roc_auc_score(y_true, y_prob))            if has_both else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob))  if has_both else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob))         if has_both else float("nan"),
        "f1":    float(f1_score(y_true, y_pred, zero_division=0)),
    }
    result = {}
    for k, vals in boot.items():
        arr = np.array(vals)
        if len(arr):
            pct = np.percentile(arr, [2.5, 97.5])
            lo, hi = float(pct[0]), float(pct[1])
        else:
            lo, hi = float("nan"), float("nan")
        result[k] = (point[k], lo, hi)
    return result


def print_bootstrap_ci(name, y_true, y_pred, y_prob, n_boot=2000):
    """Bootstrap 95% CI 콘솔 테이블 출력 후 CI dict 반환."""
    ci = bootstrap_ci(y_true, y_pred, y_prob, n_boot=n_boot)
    labels = [("AUC-ROC", "auc"), ("AUPRC", "auprc"),
              ("Brier",   "brier"), ("Accuracy", "acc"), ("F1", "f1")]
    print(f"\n  Bootstrap 95% CI — {name}  (n_boot={n_boot})")
    print(f"  {'Metric':<12} {'Estimate':>10} {'95% CI Lower':>14} {'95% CI Upper':>14}")
    print(f"  {'-'*52}")
    for label, key in labels:
        est, lo, hi = ci[key]
        print(f"  {label:<12} {est:>10.4f} {lo:>14.4f} {hi:>14.4f}")
    return ci


def print_delong_comparison(name1, name2, y_true, y_prob1, y_prob2):
    """DeLong AUC 검정 결과를 콘솔 출력 후 결과 dict 반환."""
    auc1, auc2, z, p = delong_test(y_true, y_prob1, y_prob2)
    delta = auc2 - auc1
    sig = ("***" if p < 0.001 else "**" if p < 0.01 else
           "*"   if p < 0.05  else "†"  if p < 0.10 else "ns")
    print(f"\n  DeLong Test (ROC AUC)  [{name1}] vs [{name2}]  (paired, 동일 test set)")
    print(f"  AUC({name1})={auc1:.4f}  AUC({name2})={auc2:.4f}"
          f"  Δ={delta:+.4f}  z={z:.3f}  p={p:.4e}  {sig}")
    print(f"  *** p<0.001  ** p<0.01  * p<0.05  † p<0.10  ns p≥0.10")
    return {"auc1": float(auc1), "auc2": float(auc2),
            "delta": float(delta), "z": float(z), "p_val": float(p)}
