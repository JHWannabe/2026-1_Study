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
        return dict(acc=accuracy_score(y_true, y_pred), auc=float("nan"),
                    auprc=float("nan"), brier=float("nan"),
                    f1=f1_score(y_true, y_pred, zero_division=0))
    return dict(
        acc=accuracy_score(y_true, y_pred),
        auc=roc_auc_score(y_true, y_prob),
        auprc=average_precision_score(y_true, y_prob),
        brier=brier_score_loss(y_true, y_prob),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )

def cv_summary_md(name, fold_metrics):
    """fold별 성능 지표와 mean±std를 마크다운 테이블 문자열로 반환."""
    keys = ["auc", "auprc", "brier", "acc", "f1"]
    vals = {k: [m[k] for m in fold_metrics] for k in keys}
    lines = [
        f"## {name} — {N_FOLDS}-Fold CV Summary\n",
        "| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |",
        "|------|---------|-------|-------|----------|----|",
    ]
    for m in fold_metrics:
        lines.append(
            f"| {m['fold']} | {m['auc']:.4f} | {m['auprc']:.4f} |"
            f" {m['brier']:.4f} | {m['acc']:.4f} | {m['f1']:.4f} |"
        )
    lines.append(
        f"| **Mean** | {np.mean(vals['auc']):.4f} | {np.mean(vals['auprc']):.4f} |"
        f" {np.mean(vals['brier']):.4f} | {np.mean(vals['acc']):.4f} | {np.mean(vals['f1']):.4f} |"
    )
    lines.append(
        f"| **±Std** | {np.std(vals['auc']):.4f} | {np.std(vals['auprc']):.4f} |"
        f" {np.std(vals['brier']):.4f} | {np.std(vals['acc']):.4f} | {np.std(vals['f1']):.4f} |"
    )
    return "\n".join(lines)

# ── DeLong test ───────────────────────────────────────────────────────────────
def delong_test(y_true, y_prob1, y_prob2):
    """DeLong (1988) — 동일 test set에서 두 모델 ROC AUC 비교. (auc1, auc2, z, p) 반환."""
    y_true  = np.asarray(y_true)
    y_prob1 = np.asarray(y_prob1, dtype=float)
    y_prob2 = np.asarray(y_prob2, dtype=float)

    pos1, neg1 = y_prob1[y_true == 1], y_prob1[y_true == 0]
    pos2, neg2 = y_prob2[y_true == 1], y_prob2[y_true == 0]
    n_pos, n_neg = len(pos1), len(neg1)

    if n_pos < 2 or n_neg < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    d1 = pos1[:, None] - neg1[None, :]
    k1 = (d1 > 0).astype(float) + 0.5 * (d1 == 0).astype(float)
    V10_1, V01_1 = k1.mean(axis=1), k1.mean(axis=0)

    d2 = pos2[:, None] - neg2[None, :]
    k2 = (d2 > 0).astype(float) + 0.5 * (d2 == 0).astype(float)
    V10_2, V01_2 = k2.mean(axis=1), k2.mean(axis=0)

    auc1, auc2   = float(V10_1.mean()), float(V10_2.mean())

    cov10    = np.cov(V10_1, V10_2)
    cov01    = np.cov(V01_1, V01_2)
    var_diff = (cov10[0, 0] + cov10[1, 1] - 2 * cov10[0, 1]) / n_pos \
             + (cov01[0, 0] + cov01[1, 1] - 2 * cov01[0, 1]) / n_neg

    if var_diff <= 0:
        return auc1, auc2, float("nan"), float("nan")

    z = float((auc1 - auc2) / np.sqrt(var_diff))
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return auc1, auc2, z, p

# ── Bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_ci(y_true, y_pred, y_prob, n_boot=2000, seed=42):
    """Bootstrap 95% CI for ACC, AUC, AUPRC, Brier, F1. {metric: (point, lo, hi)} 반환."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=float)
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    boot   = {k: [] for k in ["acc", "auc", "auprc", "brier", "f1"]}

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
        "auc":   float(roc_auc_score(y_true, y_prob))           if has_both else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if has_both else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob))        if has_both else float("nan"),
        "f1":    float(f1_score(y_true, y_pred, zero_division=0)),
    }
    result = {}
    for k, arr in {k: np.array(v) for k, v in boot.items()}.items():
        lo, hi = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) \
                 if len(arr) else (float("nan"), float("nan"))
        result[k] = (point[k], lo, hi)
    return result

def bootstrap_ci_md(name, y_true, y_pred, y_prob, n_boot=2000):
    """Bootstrap 95% CI 마크다운 테이블 문자열 생성. (ci, text) 반환."""
    ci = bootstrap_ci(y_true, y_pred, y_prob, n_boot=n_boot)
    labels = [("AUC-ROC", "auc"), ("AUPRC", "auprc"),
              ("Brier",   "brier"), ("Accuracy", "acc"), ("F1", "f1")]
    lines = [
        f"### Bootstrap 95% CI — {name}  (n_boot={n_boot})\n",
        "| Metric | Estimate | 95% CI Lower | 95% CI Upper |",
        "|--------|----------|--------------|--------------|",
    ]
    for label, key in labels:
        est, lo, hi = ci[key]
        lines.append(f"| {label} | {est:.4f} | {lo:.4f} | {hi:.4f} |")
    return ci, "\n".join(lines)

def print_delong_comparison(name1, name2, y_true, y_prob1, y_prob2):
    """DeLong AUC 검정 결과를 콘솔 출력 후 결과 dict 반환."""
    auc1, auc2, z, p = delong_test(y_true, y_prob1, y_prob2)
    delta = auc2 - auc1
    sig = ("***" if p < 0.001 else "**" if p < 0.01 else
           "*"   if p < 0.05  else "†"  if p < 0.10 else "ns")
    print(f"\n  DeLong Test  [{name1}] vs [{name2}]  (paired, 동일 test set)")
    print(f"  AUC({name1})={auc1:.4f}  AUC({name2})={auc2:.4f}"
          f"  Δ={delta:+.4f}  z={z:.3f}  p={p:.4e}  {sig}")
    print(f"  *** p<0.001  ** p<0.01  * p<0.05  † p<0.10  ns p≥0.10")
    return {"auc1": float(auc1), "auc2": float(auc2),
            "delta": float(delta), "z": float(z), "p_val": float(p)}
