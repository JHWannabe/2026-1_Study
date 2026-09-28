"""
통계 검정 유틸리티
  - DeLong test: AUC vs chance, 두 모델 AUC 쌍 비교
  - Bootstrap 95% CI: AUC, Sensitivity, Specificity, PPV, NPV
  - McNemar's test: 두 모델 예측 쌍 비교
"""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix


# ── DeLong test ─────────────────────────────────────────────────────────────

def _placement_values(y_true, y_score):
    """DeLong structural components (V10, V01)."""
    pos = y_score[y_true == 1].astype(float)
    neg = y_score[y_true == 0].astype(float)
    n, m = len(pos), len(neg)

    # V10[i]: P(pos[i] > neg)  — 양성 샘플 i가 음성 샘플 전체를 이기는 비율
    V10 = np.array([np.mean((xi > neg) + 0.5 * (xi == neg)) for xi in pos])
    # V01[j]: P(pos > neg[j])  — 음성 샘플 j가 양성 샘플 전체에 지는 비율
    V01 = np.array([np.mean((pos > xj) + 0.5 * (pos == xj)) for xj in neg])

    auc = float(V10.mean())
    var = np.var(V10, ddof=1) / n + np.var(V01, ddof=1) / m
    return auc, var, V10, V01


def delong_test_vs_chance(y_true, y_score):
    """AUC가 0.5(무작위 분류)보다 유의하게 높은지 검정.

    Returns
    -------
    auc : float
    z   : float  (z-statistic)
    p   : float  (two-sided p-value)
    """
    auc, var, _, _ = _placement_values(y_true, y_score)
    se = np.sqrt(max(var, 1e-12))
    z = (auc - 0.5) / se
    p = 2 * stats.norm.sf(abs(z))
    return auc, z, p


def delong_test_pair(y_true, y_score1, y_score2):
    """같은 피험자에서 얻은 두 AUC를 DeLong 방법으로 쌍 비교.

    Returns
    -------
    auc1, auc2 : float
    z          : float  (AUC1 - AUC2 기준)
    p          : float  (two-sided)
    """
    auc1, var1, V10_1, V01_1 = _placement_values(y_true, y_score1)
    auc2, var2, V10_2, V01_2 = _placement_values(y_true, y_score2)

    n = len(V10_1)
    m = len(V01_1)

    cov = (np.cov(V10_1, V10_2)[0, 1] / n
           + np.cov(V01_1, V01_2)[0, 1] / m)

    var_diff = var1 + var2 - 2 * cov
    se = np.sqrt(max(var_diff, 1e-12))
    z = (auc1 - auc2) / se
    p = 2 * stats.norm.sf(abs(z))
    return auc1, auc2, z, p


# ── Bootstrap CI ────────────────────────────────────────────────────────────

def _safe_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size < 4:
        return dict(sens=np.nan, spec=np.nan, ppv=np.nan, npv=np.nan)
    tn, fp, fn, tp = cm.ravel()
    return {
        "sens": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "spec": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "ppv":  tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "npv":  tn / (tn + fn) if (tn + fn) > 0 else np.nan,
    }


def bootstrap_ci(y_true, y_score, y_pred, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap percentile 95% CI.

    Parameters
    ----------
    y_true  : array-like  실제 레이블
    y_score : array-like  연속 확률값 (AUC 계산)
    y_pred  : array-like  이진 예측 (Sens/Spec/PPV/NPV 계산)

    Returns
    -------
    dict  {metric: (lower, upper)}
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    boot = {k: [] for k in ("auc", "sens", "spec", "ppv", "npv")}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys, yp = y_true[idx], y_score[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot["auc"].append(roc_auc_score(yt, ys))
        m = _safe_metrics(yt, yp)
        for k in ("sens", "spec", "ppv", "npv"):
            boot[k].append(m[k])

    lo_p, hi_p = alpha / 2 * 100, (1 - alpha / 2) * 100
    return {
        "AUC":         (np.nanpercentile(boot["auc"],  lo_p), np.nanpercentile(boot["auc"],  hi_p)),
        "Sensitivity": (np.nanpercentile(boot["sens"], lo_p), np.nanpercentile(boot["sens"], hi_p)),
        "Specificity": (np.nanpercentile(boot["spec"], lo_p), np.nanpercentile(boot["spec"], hi_p)),
        "PPV":         (np.nanpercentile(boot["ppv"],  lo_p), np.nanpercentile(boot["ppv"],  hi_p)),
        "NPV":         (np.nanpercentile(boot["npv"],  lo_p), np.nanpercentile(boot["npv"],  hi_p)),
    }


# ── McNemar's test ───────────────────────────────────────────────────────────

def mcnemar_test(y_true, y_pred1, y_pred2):
    """McNemar's test (Edwards 연속 수정 포함): 두 모델 예측 쌍 비교.

    b: Model1 정답, Model2 오답
    c: Model1 오답, Model2 정답

    Returns
    -------
    chi2 : float
    p    : float  (two-sided)
    """
    y_true = np.asarray(y_true)
    b = int(np.sum((y_pred1 == y_true) & (y_pred2 != y_true)))
    c = int(np.sum((y_pred1 != y_true) & (y_pred2 == y_true)))
    if b + c == 0:
        return np.nan, np.nan
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.chi2.sf(chi2, df=1))
    return float(chi2), p
