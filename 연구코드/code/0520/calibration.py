"""
Temperature Scaling 캘리브레이션 모듈.

Temperature Scaling (Guo et al., 2017):
  - 모델 logit에 단일 스칼라 T를 나눠 확률을 조정
  - T > 1: 확률을 0.5 방향으로 모아 과신을 완화 (over-confident 교정)
  - T < 1: 극단 확률을 강화 (under-confident 교정)

사용법:
  1. OOF(Out-of-Fold) logit과 레이블로 최적 T 탐색
  2. 탐색된 T를 test logit에 적용해 calibrated prob 반환
  3. ECE(Expected Calibration Error) 계산으로 개선도 확인

캘리브레이션 흐름:
  CV fold → 각 fold validation logit/label 수집
  → find_optimal_temperature(pooled_logits, pooled_labels)
  → calibrate_probs(test_logits, T)
  → compute_ece(y_true, probs_before/after)
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit   # sigmoid


def calibrate_probs(logits: np.ndarray, T: float) -> np.ndarray:
    """logits / T 후 sigmoid — calibrated 확률 반환."""
    return expit(logits / T).astype(np.float32)


def find_optimal_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    T_range: tuple = (0.1, 10.0),
) -> float:
    """
    NLL(Negative Log-Likelihood)을 최소화하는 최적 온도 T를 탐색.

    Parameters
    ----------
    logits : (N,) raw logits (sigmoid 적용 전)
    y_true : (N,) binary labels (0 or 1)
    T_range: (lo, hi) 탐색 범위

    Returns
    -------
    T_opt : float — 최적 온도 파라미터
    """
    def neg_log_likelihood(T):
        probs = calibrate_probs(logits, T)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))

    result = minimize_scalar(neg_log_likelihood, bounds=T_range, method="bounded")
    return float(result.x)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) 계산.

    n_bins 개의 균등 확률 구간으로 분할 후,
    각 구간에서 |mean_confidence - fraction_positive| 의 가중 평균.
    """
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    n     = len(y_true)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos   = y_true[mask].mean()
        mean_conf  = y_prob[mask].mean()
        ece += mask.sum() / n * abs(frac_pos - mean_conf)

    return float(ece)


def collect_oof_logits(fold_logits: list, fold_labels: list):
    """
    CV fold별 validation logit/label 리스트를 하나의 배열로 pooling.

    Parameters
    ----------
    fold_logits : list of (n_val,) arrays — fold별 raw logit
    fold_labels : list of (n_val,) arrays — fold별 true label

    Returns
    -------
    pooled_logits : (N_total,) np.ndarray
    pooled_labels : (N_total,) np.ndarray
    """
    return (
        np.concatenate(fold_logits).astype(np.float32),
        np.concatenate(fold_labels).astype(np.int64),
    )


def calibration_summary(label: str,
                         y_true: np.ndarray,
                         probs_before: np.ndarray,
                         probs_after: np.ndarray,
                         T: float) -> dict:
    """캘리브레이션 전/후 ECE 비교 요약 dict 반환 + 콘솔 출력."""
    from sklearn.metrics import brier_score_loss
    ece_before = compute_ece(y_true, probs_before)
    ece_after  = compute_ece(y_true, probs_after)
    brier_before = brier_score_loss(y_true, probs_before)
    brier_after  = brier_score_loss(y_true, probs_after)

    print(f"\n{'='*55}")
    print(f"  Calibration — {label}  (T={T:.4f})")
    print(f"{'='*55}")
    print(f"  {'Metric':<20} {'Before':>10} {'After':>10} {'Δ':>10}")
    print(f"  {'-'*50}")
    print(f"  {'ECE':<20} {ece_before:>10.4f} {ece_after:>10.4f}"
          f" {ece_after - ece_before:>+10.4f}")
    print(f"  {'Brier':<20} {brier_before:>10.4f} {brier_after:>10.4f}"
          f" {brier_after - brier_before:>+10.4f}")

    return {
        "label": label, "T": T,
        "ece_before": ece_before, "ece_after": ece_after,
        "brier_before": brier_before, "brier_after": brier_after,
    }
