"""
AEC 커브에서 해석가능한 핸드크래프트 피처를 추출하는 모듈.

추출 피처 11종:
  aec_mean       — 평균값 (전체 방사선량 수준)
  aec_std        — 표준편차 (곡선 가변성)
  aec_max        — 최댓값
  aec_min        — 최솟값
  aec_range      — 범위 (max - min)
  aec_auc        — 사다리꼴 적분 (곡선 아래 면적)
  aec_peak_pos   — 최댓값 위치 (정규화 [0,1])
  aec_skew       — 왜도 (분포 비대칭성)
  aec_kurtosis   — 첨도 (분포 뾰족함)
  aec_slope_1h   — 전반부 기울기 (선형 회귀)
  aec_slope_2h   — 후반부 기울기 (선형 회귀)

임상 의미:
  - mean/auc     : 환자 체형에 따른 방사선량 필요량
  - std/range    : 스캔 중 체내 감쇠 변화 (근육·지방 분포)
  - peak_pos     : 최대 감쇠 위치 (복부 중심 위치 반영)
  - slope_*      : 체부 형태 변화 방향성 (상/하복부 대칭)
"""
import numpy as np
from scipy import stats as scipy_stats

AEC_FEATURE_NAMES = [
    "aec_mean", "aec_std", "aec_max", "aec_min", "aec_range",
    "aec_auc", "aec_peak_pos", "aec_skew", "aec_kurtosis",
    "aec_slope_1h", "aec_slope_2h",
]
N_AEC_FEATURES = len(AEC_FEATURE_NAMES)   # 11


def extract_aec_features(X_aec: np.ndarray) -> np.ndarray:
    """
    AEC 커브 배열에서 핸드크래프트 피처를 추출한다.

    Parameters
    ----------
    X_aec : (N, P) float32  — 원본 AEC 커브 (P=256)

    Returns
    -------
    feats : (N, 11) float32  — 피처 행렬 (AEC_FEATURE_NAMES 순서)
    """
    N, P = X_aec.shape
    feats = np.zeros((N, N_AEC_FEATURES), dtype=np.float32)

    x_norm = np.linspace(0.0, 1.0, P)   # 정규화된 위치 축 [0, 1]
    mid    = P // 2

    # ── 기술 통계 ─────────────────────────────────────────────
    feats[:, 0] = X_aec.mean(axis=1)                               # mean
    feats[:, 1] = X_aec.std(axis=1)                                # std
    feats[:, 2] = X_aec.max(axis=1)                                # max
    feats[:, 3] = X_aec.min(axis=1)                                # min
    feats[:, 4] = feats[:, 2] - feats[:, 3]                        # range

    # ── 곡선 아래 면적 (사다리꼴) ────────────────────────────
    feats[:, 5] = np.trapz(X_aec, dx=1.0 / P, axis=1)             # auc

    # ── 최댓값 위치 (정규화) ─────────────────────────────────
    feats[:, 6] = X_aec.argmax(axis=1).astype(np.float32) / (P - 1)  # peak_pos

    # ── 분포 형태 통계 ────────────────────────────────────────
    feats[:, 7] = scipy_stats.skew(X_aec, axis=1).astype(np.float32)       # skew
    feats[:, 8] = scipy_stats.kurtosis(X_aec, axis=1).astype(np.float32)   # kurtosis

    # ── 전·후반부 기울기 (선형 회귀) ─────────────────────────
    # 스캔 방향으로 방사선량이 어떻게 변하는지 (상→하복부 기울기)
    x1 = x_norm[:mid]
    x2 = x_norm[mid:]
    for i in range(N):
        slope1 = np.polyfit(x1, X_aec[i, :mid], 1)[0]
        slope2 = np.polyfit(x2, X_aec[i, mid:], 1)[0]
        feats[i, 9]  = slope1   # slope_1h (전반부)
        feats[i, 10] = slope2   # slope_2h (후반부)

    return feats


def extract_aec_features_fast(X_aec: np.ndarray) -> np.ndarray:
    """
    extract_aec_features의 벡터화 버전 (루프 없이 선형 회귀).
    데이터 수가 많을 때 속도 이점.
    """
    N, P = X_aec.shape
    feats = np.zeros((N, N_AEC_FEATURES), dtype=np.float32)

    x_norm = np.linspace(0.0, 1.0, P)
    mid    = P // 2

    feats[:, 0] = X_aec.mean(axis=1)
    feats[:, 1] = X_aec.std(axis=1)
    feats[:, 2] = X_aec.max(axis=1)
    feats[:, 3] = X_aec.min(axis=1)
    feats[:, 4] = feats[:, 2] - feats[:, 3]
    feats[:, 5] = np.trapz(X_aec, dx=1.0 / P, axis=1)
    feats[:, 6] = X_aec.argmax(axis=1).astype(np.float32) / (P - 1)
    feats[:, 7] = scipy_stats.skew(X_aec, axis=1).astype(np.float32)
    feats[:, 8] = scipy_stats.kurtosis(X_aec, axis=1).astype(np.float32)

    # 전반부 선형 회귀 (벡터화: X @ (X^T X)^{-1} X^T y)
    def _batch_slope(X_sub: np.ndarray, x_ax: np.ndarray) -> np.ndarray:
        xm = x_ax - x_ax.mean()
        denom = (xm ** 2).sum()
        if denom == 0:
            return np.zeros(N, dtype=np.float32)
        return ((X_sub - X_sub.mean(axis=1, keepdims=True)) @ xm / denom).astype(np.float32)

    feats[:, 9]  = _batch_slope(X_aec[:, :mid], x_norm[:mid])
    feats[:, 10] = _batch_slope(X_aec[:, mid:], x_norm[mid:])

    return feats
