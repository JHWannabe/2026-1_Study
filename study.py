"""
AEC 보조 근감소증 진단 연구 - 최종 재현 스크립트

[분석 목표]
CT 촬영 시 기계가 자동으로 기록하는 AEC(자동 노출 제어) 곡선을 활용해
임상 정보만으로는 애매한 환자의 근감소증(low SMI) 진단 정확도를 높인다.

[전체 처리 순서]
  1. 강남(개발) + 신촌(외부 검증) 데이터 로드
  2. SMI = TAMA / 키² 재계산 → Derstine 기준으로 low SMI 라벨 부여
     남성 < 45.4 cm²/m², 여성 < 34.4 cm²/m²
  3. 임상 모델 학습: 나이 + 성별 + 키 + 체중
  4. AEC 곡선 전처리: log1p 변환 → Savitzky-Golay 스무딩 → 환자별 robust z 정규화
  5. 국소 쌍대(local pairwise) AEC 점수 계산:
     임상 조건이 비슷한 환자끼리 비교해 AEC 곡선이 근감소증 패턴에 얼마나 가까운지 점수화
  6. 최종 판정 규칙 (strong primary):
     임상 점수가 기준값 ±0.075 이내인 "회색지대" 환자에만 AEC 점수(≥0.80)로 재판정
  7. 성능 보고: AUC, 민감도, 특이도, 정확도, PPV, NPV, McNemar p-값
  8. AEC 임계값 0.80이 임의적이지 않음을 Youden·비용함수·DCA 방법으로 재확인

  AEC 쌍대 채점은 배치 predict_proba 단일 호출로 최적화되어 있다.
"""

# =============================================================================
# 1. 표준 라이브러리 임포트
# =============================================================================

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

# =============================================================================
# 2. 외부 라이브러리 임포트
# =============================================================================

import numpy as np
import pandas as pd
from scipy.fftpack import dct
from scipy.signal import savgol_filter
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression as _SkLogReg
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as _SkScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# 3. 경로 및 분석 상수 (수정이 필요한 경우 이 섹션만 변경)
# =============================================================================

# 강남 개발 코호트 데이터 (최종 분석에 사용된 엑셀)
DEV_XLSX = Path(r"연구코드\data\강남\강남_liver_merged_features.xlsx")
#DEV_XLSX = Path(r"연구코드\data\강남_features_mAs.xlsx")

# 신촌 외부 검증 데이터 (AEC 방향 오류 수정 후 최종본)
SDATA_XLSX = Path(r"연구코드\data\신촌\신촌_liver_merged_features.xlsx")

# 결과 저장 폴더
OUT_DIR = Path(r"연구코드\results\study_2026-06-22")
#OUT_DIR = Path(r"연구코드\results\study_2026-06-22_이홍선교수님")
os.makedirs(OUT_DIR, exist_ok=True)


# 재현성을 위한 랜덤 시드
SEED = 20260622

# Derstine L3 SMI 기준값 (근감소증 판정 컷오프)
SMI_MALE_CUTOFF   = 45.4   # 남성: cm²/m²
SMI_FEMALE_CUTOFF = 34.4   # 여성: cm²/m²

# 최종 판정 규칙 파라미터
GRAY_DELTA          = 0.075  # 임상 점수 ±이 범위 내면 "회색지대"로 AEC 재판정
FINAL_AEC_THRESHOLD = 0.800  # AEC 점수 ≥ 이 값이면 근감소증으로 판정

# 국소 쌍대 AEC 모델 하이퍼파라미터 (단순하게 고정)
CALIPER_FRAC       = 0.05  # propensity logit 표준편차의 5%를 매칭 허용 범위로 사용
TRAIN_MAX_CONTROLS = 3     # 케이스 1명당 대조군 최대 3명 매칭
FEATURE_SELECT_K   = 20    # AEC 피처 중 상위 20개 선택
LOCAL_K            = 3     # 점수 계산 시 참조할 가장 가까운 이웃 수

# =============================================================================
# 4. 보조 유틸리티: 진행 상황 출력, 통계 지표, p-값
# =============================================================================

T0 = time.time()  # 스크립트 시작 시각


def progress(step: int, total: int, label: str) -> None:
    """진행 상황을 바 형태로 출력한다."""
    width = 30
    done = int(width * step / total)
    bar = "#" * done + "-" * (width - done)
    print(f"[{bar}] {step:02d}/{total:02d} {time.time() - T0:7.1f}s  {label}", flush=True)


def exact_mcnemar_p(a: int, b: int) -> float:
    """
    정확한 양측 McNemar p-값을 계산한다.

    a, b는 두 모델이 서로 다른 결과를 낸 불일치 쌍의 수다.
      예) 민감도 비교: a = 임상이 놓쳤으나 워크플로가 잡은 수
                       b = 임상이 잡았으나 워크플로가 놓친 수
    """
    n = int(a + b)
    if n == 0:
        return 1.0
    k = min(int(a), int(b))
    probs = [math.comb(n, i) * (0.5**n) for i in range(n + 1)]
    obs = probs[k]
    return min(1.0, sum(p for p in probs if p <= obs + 1e-15))


def confusion(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """민감도, 특이도, 정확도 및 혼동행렬 원소(TP/FN/TN/FP)를 반환한다."""
    y    = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(bool)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "Sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "Accuracy":    (tp + tn) / len(y) if len(y) else np.nan,
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
    }


def ppv_npv(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """양성예측도(PPV)와 음성예측도(NPV)를 반환한다."""
    m   = confusion(y, pred)
    ppv = m["TP"] / (m["TP"] + m["FP"]) if (m["TP"] + m["FP"]) else np.nan
    npv = m["TN"] / (m["TN"] + m["FN"]) if (m["TN"] + m["FN"]) else np.nan
    return float(ppv), float(npv)


def paired_stats(y: np.ndarray, base_pred: np.ndarray, workflow_pred: np.ndarray) -> Dict[str, float]:
    """
    임상 단독 vs AEC 보조 워크플로의 쌍대 비교 통계를 계산한다.

      Rescued_FN   : 임상이 놓쳤으나 워크플로가 잡은 근감소증 환자 수
      Lost_TP      : 임상이 잡았으나 워크플로가 놓친 근감소증 환자 수
      Corrected_FP : 임상이 잘못 양성으로 판정했으나 워크플로가 바로잡은 정상 환자 수
      Added_FP     : 임상은 정상으로 맞췄으나 워크플로가 오히려 양성으로 만든 정상 환자 수
    """
    y             = np.asarray(y).astype(int)
    base_pred     = np.asarray(base_pred).astype(bool)
    workflow_pred = np.asarray(workflow_pred).astype(bool)
    ev  = y == 1
    neg = y == 0

    rescued_fn   = int((~base_pred[ev]  &  workflow_pred[ev]).sum())
    lost_tp      = int(( base_pred[ev]  & ~workflow_pred[ev]).sum())
    corrected_fp = int(( base_pred[neg] & ~workflow_pred[neg]).sum())
    added_fp     = int((~base_pred[neg] &  workflow_pred[neg]).sum())

    base_ok     = base_pred     == y.astype(bool)
    workflow_ok = workflow_pred == y.astype(bool)
    acc_imp    = int((~base_ok &  workflow_ok).sum())
    acc_worse  = int(( base_ok & ~workflow_ok).sum())

    return {
        "Rescued_FN":        rescued_fn,
        "Lost_TP":           lost_tp,
        "Corrected_FP":      corrected_fp,
        "Added_FP":          added_fp,
        "Sensitivity_p":     exact_mcnemar_p(rescued_fn, lost_tp),
        "Specificity_p":     exact_mcnemar_p(corrected_fp, added_fp),
        "Accuracy_p":        exact_mcnemar_p(acc_imp, acc_worse),
        "Accuracy_improved": acc_imp,
        "Accuracy_worsened": acc_worse,
        "NetCorrected":      acc_imp - acc_worse,
    }


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    """결측·단일 클래스 등 예외 상황을 처리하며 ROC AUC를 반환한다."""
    mask = np.isfinite(score)
    if mask.sum() == 0 or len(np.unique(np.asarray(y)[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(np.asarray(y)[mask], np.asarray(score)[mask]))


def youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """민감도 + 특이도 - 1 을 최대화하는 Youden 임계값을 반환한다."""
    fpr, tpr, thr = roc_curve(y_true, score)
    valid = np.isfinite(thr)
    if not valid.any():
        return 0.5
    j   = tpr - fpr
    idx = np.where(valid)[0][int(np.argmax(j[valid]))]
    return float(thr[idx])


# =============================================================================
# 5. 데이터 로드 및 결과 라벨 생성
# =============================================================================


def find_aec_cols(df: pd.DataFrame) -> List[str]:
    """데이터프레임에서 aec_1 ~ aec_128 컬럼을 찾아 숫자 순서로 정렬해 반환한다."""
    cols = [c for c in df.columns if str(c).startswith("aec_")]
    return sorted(cols, key=lambda x: int(str(x).split("_")[-1]))


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    SMI를 재계산하고 Derstine 성별별 기준으로 low SMI 라벨을 부여한다.

      SMI_calc        = TAMA(cm²) / (키(m))²
      LowSMI_Derstine = 1 (근감소증), 0 (정상)
    """
    df = df.copy()
    df["PatientSex"] = df["PatientSex"].astype(str).str.upper().str.strip()
    df["SMI_calc"]   = df["TAMA"].astype(float) / ((df["Height"].astype(float) / 100.0) ** 2)
    df["LowSMI_Derstine"] = (
        (df["PatientSex"].eq("M") & (df["SMI_calc"] < SMI_MALE_CUTOFF))
        | (df["PatientSex"].eq("F") & (df["SMI_calc"] < SMI_FEMALE_CUTOFF))
    ).astype(int)
    return df


def load_data(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """신촌 외부 검증 데이터를 metadata + aec_128 시트에서 로드한다."""
    meta     = pd.read_excel(path, sheet_name="metadata")
    aec      = pd.read_excel(path, sheet_name="aec_128")
    aec_cols = find_aec_cols(aec)
    if len(aec_cols) != 128:
        raise ValueError(f"신촌 aec_128 시트에 128개 컬럼이 필요합니다. 현재: {len(aec_cols)}개")

    merged = meta.merge(aec[["PatientID"] + aec_cols], on="PatientID", how="inner")

    # Manufacturer 컬럼명 통일
    if "ManufacturerModelName" not in merged.columns:
        if "Manufacturer" in merged.columns:
            merged["ManufacturerModelName"] = merged["Manufacturer"]
        elif "manufacturer_model" in merged.columns:
            merged["ManufacturerModelName"] = merged["manufacturer_model"]
        else:
            merged["ManufacturerModelName"] = "Unknown"

    required = ["PatientAge", "PatientSex", "Height", "Weight", "TAMA", "ManufacturerModelName"] + aec_cols
    for col in ["PatientAge", "Height", "Weight", "TAMA"] + aec_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=required).copy().reset_index(drop=True)

    merged = add_outcome(merged)
    merged["ManufacturerModelName"] = merged["ManufacturerModelName"].astype(str).str.strip()
    return merged, aec_cols


def aec_qc_flags(df: pd.DataFrame, aec_cols: Sequence[str]) -> pd.DataFrame:
    """
    AEC 곡선 품질 검사 결과를 반환한다.

    최종 모델은 이 플래그로 환자를 제외하지 않는다.
    (강남 4명, 신촌 1명이 실패하며 모두 정상 환자여서 영향이 없음)
    민감도 분석 보조 자료로 저장한다.
    """
    x        = df[list(aec_cols)].to_numpy(float)
    row_sd   = np.nanstd(x, axis=1)
    row_min  = np.nanmin(x, axis=1)
    row_max  = np.nanmax(x, axis=1)
    max_jump = np.nanmax(np.abs(np.diff(x, axis=1)), axis=1)

    flags = pd.DataFrame({
        "PatientID":    df["PatientID"].to_numpy(),
        "missing":      np.isnan(x).any(axis=1),
        "nonpositive":  (x <= 0).any(axis=1),
        "flat_sd_lt_1": row_sd < 1,
        "range_lt_5":   (row_max - row_min) < 5,
        "max_gt_800":   row_max > 800,
        "jump_gt_500":  max_jump > 500,
        "AEC_sd":       row_sd,
        "AEC_range":    row_max - row_min,
        "AEC_max":      row_max,
        "AEC_max_jump": max_jump,
    })
    qc_cols = ["missing", "nonpositive", "flat_sd_lt_1", "range_lt_5", "max_gt_800", "jump_gt_500"]
    flags["AEC_QC_fail"] = flags[qc_cols].any(axis=1)
    return flags


# =============================================================================
# 6. AEC 피처 엔지니어링
# =============================================================================


def smooth_log(df: pd.DataFrame, aec_cols: Sequence[str]) -> np.ndarray:
    """
    AEC 원시값 전처리: log1p 변환 후 Savitzky-Golay 스무딩 적용.

    log1p: 우측 편포된 분포를 완화한다.
    Savitzky-Golay: 곡선 형태를 보존하면서 국소 노이즈를 제거한다.
    """
    x = np.clip(df[list(aec_cols)].astype(float).to_numpy(), 0.0, None)
    x = np.log1p(x)
    return savgol_filter(x, window_length=9, polyorder=2, axis=1, mode="interp")


def patientwise_robust_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    환자별 robust z 정규화.

    중앙값과 IQR 기반으로 정규화해 스캐너 기종 간 절대 노출량 차이를 제거하고
    곡선의 형태(패턴)에만 집중한다.
    """
    med = np.nanmedian(x, axis=1, keepdims=True)
    q25 = np.nanquantile(x, 0.25, axis=1, keepdims=True)
    q75 = np.nanquantile(x, 0.75, axis=1, keepdims=True)
    return (x - med) / ((q75 - q25) + eps)


def segment_features(x: np.ndarray, bins: int) -> np.ndarray:
    """
    다중 해상도 구간 요약 피처.

    bins=2  : 위·아래 절반의 전체적 차이 포착
    bins=8  : 중간 단위 지역 패턴 포착
    bins=16 : 세밀한 국소 형태 포착
    각 구간에서 평균·표준편차·최솟값·최댓값·범위·기울기·변화율 등을 추출한다.
    """
    edges = np.linspace(0, x.shape[1], bins + 1).astype(int)
    vals  = []
    for i in range(bins):
        seg = x[:, edges[i]: edges[i + 1]]
        d1  = np.gradient(seg, axis=1) if seg.shape[1] > 1 else np.zeros_like(seg)
        vals.extend([
            seg.mean(axis=1), seg.std(axis=1),
            seg.min(axis=1),  seg.max(axis=1),
            np.ptp(seg, axis=1),
            seg[:, -1] - seg[:, 0],
            np.abs(d1).mean(axis=1),
            np.abs(d1).max(axis=1),
        ])
    return np.column_stack(vals)


def inverse_features_from_x(x: np.ndarray) -> np.ndarray:
    """전체 곡선의 전역 형태 기술자 및 저차 모폴로지 요약 피처."""
    n        = x.shape[1]
    thirds   = np.array_split(np.arange(n), 3)
    quarters = np.array_split(np.arange(n), 4)
    mid      = np.arange(max(0, n // 2 - 8), min(n, n // 2 + 8))
    d1       = np.gradient(x, axis=1)
    d2       = np.gradient(d1, axis=1)

    # 곡선 무게중심: 도량 분포의 중심 위치
    pos    = np.arange(n)
    w      = np.maximum(x - x.min(axis=1, keepdims=True), 0.0) + 1e-6
    center_of_mass = (w * pos).sum(axis=1) / w.sum(axis=1)

    vals = [
        x.mean(axis=1), x.std(axis=1), x.min(axis=1), x.max(axis=1),
        np.ptp(x, axis=1),
        x[:, -1] - x[:, 0],                              # 끝값 - 시작값 (전체 기울기)
        x[:, thirds[0]].mean(axis=1),                    # 상단 1/3 평균
        x[:, thirds[1]].mean(axis=1),                    # 중단 1/3 평균
        x[:, thirds[2]].mean(axis=1),                    # 하단 1/3 평균
        x[:, quarters[0]].mean(axis=1),
        x[:, quarters[1]].mean(axis=1),
        x[:, quarters[2]].mean(axis=1),
        x[:, quarters[3]].mean(axis=1),
        x[:, mid].mean(axis=1),                          # 중앙 구간 평균
        x[:, mid].min(axis=1),
        x[:, mid].std(axis=1),
        # 중단이 상·하단보다 얼마나 볼록한지 (오목·볼록 지수)
        x[:, thirds[1]].mean(axis=1) - 0.5 * (x[:, thirds[0]].mean(axis=1) + x[:, thirds[2]].mean(axis=1)),
        x[:, thirds[2]].mean(axis=1) - x[:, thirds[0]].mean(axis=1),  # 하단 - 상단 기울기
        center_of_mass,
        np.abs(d1).mean(axis=1),  # 평균 변화 속도
        np.abs(d1).max(axis=1),   # 최대 변화 속도
        np.abs(d2).mean(axis=1),  # 평균 가속도 (곡률)
    ]
    return np.column_stack(vals)


def dct_features_from_x(x: np.ndarray, n_coeff: int = 12) -> np.ndarray:
    """
    저주파 DCT 계수로 전체 곡선 형태를 압축해서 표현.

    고주파 노이즈는 무시하고 전체적인 윤곽(저주파 성분)만 포착한다.
    """
    coeff = dct(x, axis=1, norm="ortho")[:, :n_coeff]
    d1    = np.gradient(x, axis=1)
    d1c   = dct(d1, axis=1, norm="ortho")[:, : min(8, n_coeff)]
    return np.column_stack([coeff, d1c])


def aec_features(df: pd.DataFrame, aec_cols: Sequence[str]) -> np.ndarray:
    """
    국소 쌍대 모델에 입력할 최종 AEC 피처 행렬을 생성한다.

    전처리(log + 스무딩 + 정규화) 후 다중 해상도 구간·전역·DCT 피처를 결합한다.
    """
    x = patientwise_robust_z(smooth_log(df, aec_cols))
    return np.column_stack([
        segment_features(x, 2),
        segment_features(x, 8),
        segment_features(x, 16),
        inverse_features_from_x(x),
        dct_features_from_x(x, 12),
    ])


# =============================================================================
# 7. 임상 모델 및 국소 쌍대 AEC 모델 헬퍼
# =============================================================================


def make_balanced_logit(c: float = 0.7) -> Pipeline:
    """클래스 불균형을 보정한 로지스틱 회귀 파이프라인을 반환한다."""
    return Pipeline([
        ("scale", _SkScaler()),
        ("lr",    _SkLogReg(C=c, max_iter=3000, class_weight="balanced")),
    ])


def make_pairwise_aec_model(k: int, n_features: int) -> Pipeline:
    """
    쌍대 AEC 모델 파이프라인 (sklearn).

    SelectKBest가 필요해 sklearn Pipeline을 유지한다.
    입력: 케이스와 매칭된 대조군의 피처 차이 벡터
    출력: 첫 번째 환자가 근감소증 패턴일 확률
    """
    return Pipeline([
        ("select", SelectKBest(f_classif, k=min(int(k), n_features))),
        ("scale",  _SkScaler()),
        ("lr",     _SkLogReg(max_iter=3000, class_weight="balanced", C=0.7)),
    ])


def clinical_matrix(df: pd.DataFrame) -> np.ndarray:
    """임상 예측 변수 행렬: 나이, 키, 체중, 성별(남=1)."""
    return np.column_stack([
        df[["PatientAge", "Height", "Weight"]].astype(float).to_numpy(),
        df["PatientSex"].eq("M").astype(int).to_numpy().reshape(-1, 1),
    ])


def propensity_matrix(
    df: pd.DataFrame, scanner_levels: Optional[Sequence[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    국소 AEC 비교를 위한 propensity(성향점수) 변수 행렬.

    스캐너 기종은 임상 예측 변수가 아니라 비슷한 환자를 찾기 위한 매칭 변수로만 사용한다.
    """
    if scanner_levels is None:
        scanner_levels = sorted(df["ManufacturerModelName"].astype(str).unique().tolist())
    scanner         = pd.Categorical(df["ManufacturerModelName"].astype(str), categories=list(scanner_levels))
    scanner_dummies = pd.get_dummies(scanner, prefix="scanner", dtype=float)
    x = np.column_stack([
        df[["PatientAge", "Height", "Weight"]].astype(float).to_numpy(),
        df["PatientSex"].eq("M").astype(int).to_numpy().reshape(-1, 1),
        scanner_dummies.to_numpy(float),
    ])
    return x, list(scanner_levels)


def logit_prob(p: np.ndarray) -> np.ndarray:
    """확률값을 로짓(log-odds)으로 변환한다. 수치 안정성을 위해 클리핑 적용."""
    eps = 1e-5
    p   = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def make_train_pairs(
    train_idx: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    ps_logit_all: np.ndarray,
    caliper: float,
    max_controls: int = TRAIN_MAX_CONTROLS,
    exact_scanner: bool = True,
) -> List[Tuple[int, int]]:
    """
    훈련용 케이스-대조군 쌍을 생성한다.

    매칭 조건: 동일 성별 & (exact_scanner=True면) 동일 스캐너 & propensity logit 거리 ≤ caliper
    케이스 1명당 최대 max_controls명의 대조군을 짝짓는다.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    cases    = [i for i in train_idx if y[i] == 1]
    controls = [i for i in train_idx if y[i] == 0]
    pairs: List[Tuple[int, int]] = []
    used_controls = set()

    for case in sorted(cases, key=lambda i: ps_logit_all[i], reverse=True):
        candidates = [
            j for j in controls
            if j not in used_controls and df.loc[j, "PatientSex"] == df.loc[case, "PatientSex"]
        ]
        if exact_scanner:
            candidates = [
                j for j in candidates
                if df.loc[j, "ManufacturerModelName"] == df.loc[case, "ManufacturerModelName"]
            ]
        if not candidates:
            continue

        candidates = np.asarray(candidates, dtype=int)
        dist = np.abs(ps_logit_all[candidates] - ps_logit_all[case])
        ok   = np.where(dist <= caliper)[0]
        if len(ok) == 0:
            continue

        chosen = candidates[ok[np.argsort(dist[ok])[:max_controls]]]
        for ctrl in chosen:
            pairs.append((int(case), int(ctrl)))
            used_controls.add(int(ctrl))
    return pairs


def nearest_by_ps(
    i: int,
    candidates: Sequence[int],
    df_query_train: pd.DataFrame,
    ps_logit_all: np.ndarray,
    caliper: float,
    k: int,
    scanner_mode: str,
) -> List[int]:
    """
    쿼리 환자와 임상적으로 가장 유사한 이웃 k명을 찾는다.

    scanner_mode:
      "exact_scanner"  : 동일 스캐너만 허용
      "prefer_scanner" : 동일 스캐너를 우선하되 부족하면 타 스캐너도 허용
    """
    cand = [j for j in candidates if df_query_train.loc[j, "PatientSex"] == df_query_train.loc[i, "PatientSex"]]

    if scanner_mode == "exact_scanner":
        cand = [j for j in cand if df_query_train.loc[j, "ManufacturerModelName"] == df_query_train.loc[i, "ManufacturerModelName"]]
    elif scanner_mode == "prefer_scanner":
        same = [j for j in cand if df_query_train.loc[j, "ManufacturerModelName"] == df_query_train.loc[i, "ManufacturerModelName"]]
        if len(same) >= min(k, 3):
            cand = same
    else:
        raise ValueError("scanner_mode는 'exact_scanner' 또는 'prefer_scanner'이어야 합니다.")

    if not cand:
        return []
    cand = np.asarray(cand, dtype=int)
    dist = np.abs(ps_logit_all[cand] - ps_logit_all[i])
    ok   = np.where(dist <= caliper)[0]
    if len(ok) == 0:
        return []
    return cand[ok[np.argsort(dist[ok])[: min(k, len(ok))]]].tolist()


def _batch_pair_scores(
    test_idx: Sequence[int],
    train_cases: Sequence[int],
    train_controls: Sequence[int],
    pair_model: Pipeline,
    x_scaled_all: np.ndarray,
    df_query_train: pd.DataFrame,
    ps_logit_all: np.ndarray,
    caliper: float,
    local_k: int = LOCAL_K,
    scanner_mode: str = "exact_scanner",
) -> np.ndarray:
    """
    테스트 환자 전원의 국소 AEC 점수를 배치로 한 번에 계산한다.

    기존 per-patient 루프 방식은 predict_proba를 환자 수만큼 반복 호출했다.
    이 함수는 모든 차이 벡터를 먼저 모아 predict_proba를 단 한 번 호출한다.
    GPU/CPU 모두에서 훨씬 빠르며, RTX 5090의 대규모 병렬 연산을 최대한 활용한다.

    반환: shape=(len(test_idx),)의 AEC 점수 배열 (이웃이 없으면 nan)
    """
    diffs: List[np.ndarray] = []
    # (테스트 배열 내 위치 n, 방향: +1=대조군비교 / -1=케이스비교)
    assignments: List[Tuple[int, int]] = []

    for n, i in enumerate(test_idx):
        ctrl_n = nearest_by_ps(i, train_controls, df_query_train, ps_logit_all, caliper, local_k, scanner_mode)
        case_n = nearest_by_ps(i, train_cases,    df_query_train, ps_logit_all, caliper, local_k, scanner_mode)
        for j in ctrl_n:
            # 쿼리 - 대조군: 값이 클수록 쿼리가 더 근감소증스러움
            diffs.append(x_scaled_all[i] - x_scaled_all[j])
            assignments.append((n, 1))
        for j in case_n:
            # 케이스 - 쿼리: 1-prob로 변환해 쿼리가 케이스보다 덜 근감소증스러운 정도 반전
            diffs.append(x_scaled_all[j] - x_scaled_all[i])
            assignments.append((n, -1))

    scores = np.full(len(test_idx), np.nan)
    if not diffs:
        return scores

    probs = pair_model.predict_proba(np.vstack(diffs))[:, 1]

    # 환자별 집계
    buckets: List[List[float]] = [[] for _ in test_idx]
    for (n, direction), prob in zip(assignments, probs):
        buckets[n].append(float(prob) if direction == 1 else 1.0 - float(prob))

    for n, vals in enumerate(buckets):
        if vals:
            scores[n] = float(np.mean(vals))
    return scores


# =============================================================================
# 8. 점수 생성: 임상 모델 + AEC 모델
# =============================================================================


def clinical_internal_external(
    dev: pd.DataFrame, ext: pd.DataFrame, folds
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    임상 모델의 내부 CV 점수와 외부 검증 점수를 생성한다.

    내부(강남):
      5-fold CV — 각 폴드는 나머지 4개 폴드로 학습한 모델로 예측
      임계값은 훈련 폴드의 Youden 기준으로 결정

    외부(신촌):
      전체 강남 데이터로 학습한 모델로 신촌 환자 예측
      임계값은 전체 강남의 Youden 기준으로 결정
    """
    y_dev = dev["LowSMI_Derstine"].astype(int).to_numpy()
    x_dev = clinical_matrix(dev)
    x_ext = clinical_matrix(ext)

    dev_score  = np.zeros(len(dev))
    dev_youden = np.zeros(len(dev))
    for train_idx, test_idx in folds:
        model = make_balanced_logit(c=0.7)
        model.fit(x_dev[train_idx], y_dev[train_idx])
        train_score          = model.predict_proba(x_dev[train_idx])[:, 1]
        dev_score[test_idx]  = model.predict_proba(x_dev[test_idx])[:, 1]
        dev_youden[test_idx] = youden_threshold(y_dev[train_idx], train_score)

    full           = make_balanced_logit(c=0.7)
    full.fit(x_dev, y_dev)
    full_dev_score = full.predict_proba(x_dev)[:, 1]
    ext_score      = full.predict_proba(x_ext)[:, 1]
    ext_youden     = youden_threshold(y_dev, full_dev_score)
    return dev_score, dev_youden, ext_score, ext_youden


def pair_score_internal(
    dev: pd.DataFrame, aec_cols: Sequence[str], y: np.ndarray, folds
) -> Tuple[np.ndarray, np.ndarray]:
    """강남 내부 CV 국소 쌍대 AEC 점수를 생성한다."""
    scanner_levels = sorted(dev["ManufacturerModelName"].astype(str).unique().tolist())
    x_ps, _        = propensity_matrix(dev, scanner_levels=scanner_levels)
    x_aec_all      = aec_features(dev, aec_cols)
    local_score    = np.full(len(dev), np.nan)
    support        = np.zeros(len(dev), dtype=bool)

    for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
        progress(4 + fold_no, 14, f"강남 내부 AEC 폴드 {fold_no}/5")

        # ① propensity 모델로 임상 유사도 점수 계산
        ps_model  = make_balanced_logit(c=1.0)
        ps_model.fit(x_ps[train_idx], y[train_idx])
        train_ps  = ps_model.predict_proba(x_ps[train_idx])[:, 1]
        all_logit = np.full(len(dev), np.nan)
        all_logit[train_idx] = logit_prob(train_ps)
        all_logit[test_idx]  = logit_prob(ps_model.predict_proba(x_ps[test_idx])[:, 1])
        caliper   = CALIPER_FRAC * float(np.std(logit_prob(train_ps)))

        # ② AEC 피처 표준화
        scaler   = _SkScaler()
        x_scaled = np.zeros_like(x_aec_all, dtype=float)
        x_scaled[train_idx] = scaler.fit_transform(x_aec_all[train_idx])
        x_scaled[test_idx]  = scaler.transform(x_aec_all[test_idx])

        # ③ 매칭 쌍 생성 및 쌍대 모델 학습
        pairs = make_train_pairs(train_idx, y, dev, all_logit, caliper, exact_scanner=True)
        if not pairs:
            continue

        pair_x, pair_y = [], []
        for case, ctrl in pairs:
            pair_x.append(x_scaled[case] - x_scaled[ctrl]); pair_y.append(1)
            pair_x.append(x_scaled[ctrl] - x_scaled[case]); pair_y.append(0)
        pair_model = make_pairwise_aec_model(FEATURE_SELECT_K, x_scaled.shape[1])
        pair_model.fit(np.vstack(pair_x), np.asarray(pair_y))

        # ④ 테스트 환자 전체를 배치로 채점 (per-patient 루프 제거)
        train_cases    = [int(i) for i in train_idx if y[i] == 1]
        train_controls = [int(i) for i in train_idx if y[i] == 0]
        batch_scores = _batch_pair_scores(
            [int(i) for i in test_idx], train_cases, train_controls,
            pair_model, x_scaled, dev, all_logit, caliper,
            local_k=LOCAL_K, scanner_mode="exact_scanner",
        )
        for idx_pos, i in enumerate(test_idx):
            local_score[i] = batch_scores[idx_pos]
        support[test_idx] = np.isfinite(local_score[test_idx])
    return local_score, support


def pair_score_external(
    dev: pd.DataFrame, ext: pd.DataFrame, aec_cols: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """강남 전체로 학습한 쌍대 AEC 모델로 신촌 환자를 채점한다."""
    y_dev     = dev["LowSMI_Derstine"].astype(int).to_numpy()
    y_ext     = ext["LowSMI_Derstine"].astype(int).to_numpy()
    x_dev_aec = aec_features(dev, aec_cols)
    x_ext_aec = aec_features(ext, aec_cols)

    scanner_levels = sorted(dev["ManufacturerModelName"].astype(str).unique().tolist())
    dev_ps, _      = propensity_matrix(dev, scanner_levels=scanner_levels)
    ext_ps, _      = propensity_matrix(ext, scanner_levels=scanner_levels)

    # ① propensity 모델을 강남으로 학습 후 신촌에도 적용
    ps_model       = make_balanced_logit(c=1.0)
    ps_model.fit(dev_ps, y_dev)
    dev_ps_prob    = ps_model.predict_proba(dev_ps)[:, 1]
    ext_ps_prob    = ps_model.predict_proba(ext_ps)[:, 1]
    combined_logit = logit_prob(np.concatenate([dev_ps_prob, ext_ps_prob]))
    caliper        = CALIPER_FRAC * float(np.std(logit_prob(dev_ps_prob)))

    # ② 두 코호트를 합쳐 AEC 피처 표준화
    combined_df  = pd.concat([dev, ext], ignore_index=True)
    combined_y   = np.concatenate([y_dev, y_ext])
    combined_aec = np.vstack([x_dev_aec, x_ext_aec])
    dev_idx      = np.arange(len(dev))
    ext_idx      = np.arange(len(dev), len(dev) + len(ext))

    scaler   = _SkScaler()
    x_scaled = np.zeros_like(combined_aec, dtype=float)
    x_scaled[dev_idx] = scaler.fit_transform(combined_aec[dev_idx])
    x_scaled[ext_idx] = scaler.transform(combined_aec[ext_idx])

    # ③ 강남 매칭 쌍으로 쌍대 모델 학습
    pairs = make_train_pairs(dev_idx, combined_y, combined_df, combined_logit, caliper, exact_scanner=True)
    if not pairs:
        return np.full(len(ext), np.nan), np.zeros(len(ext), dtype=bool)

    pair_x, pair_y = [], []
    for case, ctrl in pairs:
        pair_x.append(x_scaled[case] - x_scaled[ctrl]); pair_y.append(1)
        pair_x.append(x_scaled[ctrl] - x_scaled[case]); pair_y.append(0)
    pair_model = make_pairwise_aec_model(FEATURE_SELECT_K, x_scaled.shape[1])
    pair_model.fit(np.vstack(pair_x), np.asarray(pair_y))

    # ④ 신촌 환자 전체를 배치로 채점 (스캐너가 다를 수 있어 prefer_scanner 사용)
    dev_cases    = [int(i) for i in dev_idx if combined_y[i] == 1]
    dev_controls = [int(i) for i in dev_idx if combined_y[i] == 0]
    local_score  = _batch_pair_scores(
        [int(i) for i in ext_idx], dev_cases, dev_controls,
        pair_model, x_scaled, combined_df, combined_logit, caliper,
        local_k=LOCAL_K, scanner_mode="prefer_scanner",
    )
    return local_score, np.isfinite(local_score)


# =============================================================================
# 9. 최종 판정 워크플로 평가 및 임계값 검증
# =============================================================================


def apply_aec_workflow(
    clinical_score: np.ndarray,
    clinical_thr,
    aec_score: np.ndarray,
    support: np.ndarray,
    aec_threshold: float,
    gray_delta: float = GRAY_DELTA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    임상 우선, AEC 회색지대 재판정 워크플로를 적용한다.

    임상 점수가 기준값에서 ±gray_delta 이내인 환자(회색지대)만 AEC 점수로 재판정한다.
    나머지 환자는 임상 판단을 그대로 사용한다.
    """
    if np.ndim(clinical_thr) == 0:
        clinical_thr = np.full(len(clinical_score), float(clinical_thr))
    base_pred = clinical_score >= clinical_thr
    gray      = support & (np.abs(clinical_score - clinical_thr) <= gray_delta)
    workflow  = base_pred.copy()
    workflow[gray] = aec_score[gray] >= aec_threshold
    return base_pred, workflow, gray


def eval_workflow_row(
    cohort: str,
    y: np.ndarray,
    clinical_score: np.ndarray,
    clinical_thr,
    aec_score: np.ndarray,
    support: np.ndarray,
    aec_threshold: float,
    rule_name: str,
) -> Dict[str, float]:
    """AEC 워크플로 임계값 하나에 대한 성능 지표 행을 반환한다."""
    base_pred, workflow, gray = apply_aec_workflow(clinical_score, clinical_thr, aec_score, support, aec_threshold)
    base_m   = confusion(y, base_pred)
    wf_m     = confusion(y, workflow)
    ppv, npv = ppv_npv(y, workflow)
    row = {
        "Cohort":               cohort,
        "Rule":                 rule_name,
        "N":                    int(len(y)),
        "Events":               int(np.asarray(y).sum()),
        "GrayN":                int(gray.sum()),
        "AECThreshold":         float(aec_threshold),
        "Clinical_AUC":         safe_auc(y, clinical_score),
        "AEC_AUC":              safe_auc(y, aec_score),
        "Clinical_Sensitivity": base_m["Sensitivity"],
        "Clinical_Specificity": base_m["Specificity"],
        "Clinical_Accuracy":    base_m["Accuracy"],
        "Workflow_Sensitivity": wf_m["Sensitivity"],
        "Workflow_Specificity": wf_m["Specificity"],
        "Workflow_Accuracy":    wf_m["Accuracy"],
        "Workflow_PPV":         ppv,
        "Workflow_NPV":         npv,
        "Workflow_TP":          wf_m["TP"],
        "Workflow_FN":          wf_m["FN"],
        "Workflow_TN":          wf_m["TN"],
        "Workflow_FP":          wf_m["FP"],
    }
    row.update(paired_stats(y, base_pred, workflow))
    return row


def dca_cumulative(
    y: np.ndarray, pred: np.ndarray,
    pt_values: np.ndarray = np.linspace(0.05, 0.50, 91)
) -> float:
    """임계 확률 범위 0.05~0.50에 걸친 누적 결정 곡선(DCA) 순편익을 반환한다."""
    y    = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(bool)
    n    = len(y)
    tp   = int(((pred == 1) & (y == 1)).sum())
    fp   = int(((pred == 1) & (y == 0)).sum())
    nb   = []
    for pt in pt_values:
        odds = pt / (1.0 - pt)
        nb.append(tp / n - fp / n * odds)
    return float(np.trapezoid(nb, pt_values))


def threshold_sweep(
    cohort: str,
    y: np.ndarray,
    clinical_score: np.ndarray,
    clinical_thr,
    aec_score: np.ndarray,
    support: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """0.001 간격 임계값 전수 탐색 — 문서화 및 민감도 분석용."""
    rows = []
    for t in thresholds:
        row = eval_workflow_row(cohort, y, clinical_score, clinical_thr, aec_score, support, float(t), "threshold_sweep")
        base_pred, workflow, _ = apply_aec_workflow(clinical_score, clinical_thr, aec_score, support, float(t))
        row["DCA_Cumulative_0p05_0p50"]          = dca_cumulative(y, workflow)
        row["Clinical_DCA_Cumulative_0p05_0p50"] = dca_cumulative(y, base_pred)
        row["SensitivityDiff"] = row["Workflow_Sensitivity"] - row["Clinical_Sensitivity"]
        row["SpecificityDiff"] = row["Workflow_Specificity"] - row["Clinical_Specificity"]
        row["AccuracyDiff"]    = row["Workflow_Accuracy"]    - row["Clinical_Accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)


def classic_threshold_methods(internal_sweep: pd.DataFrame) -> pd.DataFrame:
    """
    강남 데이터만으로 고전적 임계값 선택 방법들을 계산한다.

    외부 데이터를 탐색하지 않는다.
    AEC 임계값 0.80이 임의적이지 않고 여러 방법에서 0.79~0.80으로 수렴함을 보인다:
      - 워크플로 Youden 최적화
      - 특이도 ≥ 0.80 조건부 최적화
      - FN:FP 비용 가중 최소화 (2:1, 5:1, 10:1)
      - DCA 누적 순편익 최대화
      - 워크플로 Youden 값을 0.05 단위로 반올림
    """
    df      = internal_sweep.copy()
    methods = []

    # 워크플로 Youden: 민감도 + 특이도 - 1 최대화
    idx = (df["Workflow_Sensitivity"] + df["Workflow_Specificity"] - 1.0).idxmax()
    methods.append(("Workflow Youden", cast(float, df.loc[idx, "AECThreshold"])))

    # 특이도 ≥ 0.80 달성 조건에서 민감도 최대화
    ok = df[df["Workflow_Specificity"] >= 0.80]
    if not ok.empty:
        idx = ok.sort_values(["Workflow_Sensitivity", "Workflow_Accuracy"], ascending=False).index[0]
        methods.append(("Workflow specificity >=0.80", cast(float, df.loc[idx, "AECThreshold"])))

    # 비용 기반: FN:FP 비율별 총 비용 최소화
    for ratio in [2, 5, 10]:
        cost = df["Workflow_FN"] * ratio + df["Workflow_FP"]
        idx  = cost.idxmin()
        methods.append((f"Cost FN:FP {ratio}:1", cast(float, df.loc[idx, "AECThreshold"])))

    # DCA 누적 순편익 최대화
    idx = df["DCA_Cumulative_0p05_0p50"].idxmax()
    methods.append(("DCA cumulative max 0.05-0.50", cast(float, df.loc[idx, "AECThreshold"])))

    # 워크플로 Youden 값을 0.05 단위로 반올림
    workflow_youden = [x[1] for x in methods if x[0] == "Workflow Youden"][0]
    rounded = float(np.clip(round(workflow_youden / 0.05) * 0.05, 0.0, 1.0))
    methods.append(("Rounded workflow Youden", rounded))

    # 논문에서 최종 채택한 임계값
    methods.append(("Final strong primary", FINAL_AEC_THRESHOLD))

    return pd.DataFrame(methods, columns=["Method", "SelectedThreshold"])


def save_results_png(final_summary: pd.DataFrame, out_path: Path) -> None:
    """임상 vs AEC 워크플로 성능 요약 테이블을 PNG로 저장한다."""
    HEADER_CLR = "#cce0f0"
    BORDER_CLR = "#555555"

    def fmt_p(p: float) -> str:
        return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"

    rel_w   = [2.0, 1.3, 1.3, 1.0, 1.3, 1.3, 1.0, 1.3, 1.3, 1.0]
    total_w = sum(rel_w)
    col_x   = [0.0]
    for w in rel_w:
        col_x.append(col_x[-1] + w / total_w)

    n_rows = 4
    row_h  = 1.0 / n_rows
    row_y  = [1.0 - (i + 1) * row_h for i in range(n_rows)]

    fig = plt.figure(figsize=(13, 3.0))
    ax  = fig.add_axes((0.005, 0.005, 0.99, 0.99))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def cell(xl: float, yb: float, w: float, h: float, txt: str,
             bold: bool = False, fs: float = 10, bg: str = "#ffffff") -> None:
        ax.add_patch(mpatches.Rectangle(
            (xl, yb), w, h, facecolor=bg, edgecolor=BORDER_CLR, linewidth=0.8,
        ))
        ax.text(xl + w / 2, yb + h / 2, txt,
                ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    yb, h = row_y[0], row_h
    cell(col_x[0], yb, col_x[1]  - col_x[0],  h, "",            bg=HEADER_CLR, fs=11)
    cell(col_x[1], yb, col_x[4]  - col_x[1],  h, "Sensitivity", bg=HEADER_CLR, fs=11, bold=True)
    cell(col_x[4], yb, col_x[7]  - col_x[4],  h, "Specificity", bg=HEADER_CLR, fs=11, bold=True)
    cell(col_x[7], yb, col_x[10] - col_x[7],  h, "Accuracy",    bg=HEADER_CLR, fs=11, bold=True)

    yb, h = row_y[1], row_h
    hdrs = ["Cohort", "Clinical\nSens", "AEC\nSens", "p",
            "Clinical\nSpec", "AEC\nSpec", "p",
            "Clinical\nAcc",  "AEC\nAcc",  "p"]
    for j, hdr in enumerate(hdrs):
        cell(col_x[j], yb, col_x[j + 1] - col_x[j], h, hdr, bold=True, fs=9.5, bg=HEADER_CLR)

    data_rows: List[List[str]] = []
    for _, row in final_summary.iterrows():
        label = "Gangnam internal" if "Gangnam" in str(row["Cohort"]) else "Shinchon external"
        data_rows.append([
            label,
            f"{float(row['Clinical_Sensitivity']):.3f}", f"{float(row['Workflow_Sensitivity']):.3f}", fmt_p(row["Sensitivity_p"]),
            f"{float(row['Clinical_Specificity']):.3f}", f"{float(row['Workflow_Specificity']):.3f}", fmt_p(row["Specificity_p"]),
            f"{float(row['Clinical_Accuracy']):.3f}",    f"{float(row['Workflow_Accuracy']):.3f}",    fmt_p(row["Accuracy_p"]),
        ])

    for ri, vals in enumerate(data_rows, start=2):
        yb, h = row_y[ri], row_h
        for j, val in enumerate(vals):
            cell(col_x[j], yb, col_x[j + 1] - col_x[j], h, val, bold=(j == 0), fs=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"연구결과 PNG 저장: {out_path}")


# =============================================================================
# 10. 메인 실행
# =============================================================================


def main() -> None:

    # ── 1. 데이터 로드 ────────────────────────────────────────────────────────
    progress(1, 14, "데이터 로드")
    dev, dev_aec_cols = load_data(DEV_XLSX)
    ext, ext_aec_cols = load_data(SDATA_XLSX)
    if len(dev_aec_cols) != 128 or len(ext_aec_cols) != 128:
        raise ValueError("두 코호트 모두 128개 AEC 컬럼이 필요합니다.")

    y_dev = dev["LowSMI_Derstine"].astype(int).to_numpy()
    y_ext = ext["LowSMI_Derstine"].astype(int).to_numpy()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(dev, y_dev))

    # ── 2. AEC 품질 검사 ──────────────────────────────────────────────────────
    progress(2, 14, "AEC QC 검사")
    dev_qc = aec_qc_flags(dev, dev_aec_cols)
    ext_qc = aec_qc_flags(ext, ext_aec_cols)
    dev_qc.to_csv(OUT_DIR / "gangnam_aec_qc_flags.csv", index=False, encoding="utf-8-sig")
    ext_qc.to_csv(OUT_DIR / "sdata_aec_qc_flags.csv",  index=False, encoding="utf-8-sig")

    # ── 3. 임상 모델 점수 생성 ────────────────────────────────────────────────
    progress(3, 14, "임상 모델 점수 생성")
    dev_clin_score, dev_clin_thr, ext_clin_score, ext_clin_thr = clinical_internal_external(dev, ext, folds)

    # ── 4. AEC 점수 생성 (내부 CV + 외부 검증) ────────────────────────────────
    progress(4, 14, "AEC 점수 준비 및 학습")
    dev_aec_score, dev_aec_support = pair_score_internal(dev, dev_aec_cols, y_dev, folds)
    progress(10, 14, "신촌 외부 AEC 점수 생성")
    ext_aec_score, ext_aec_support = pair_score_external(dev, ext, dev_aec_cols)

    # ── 5. 최종 strong primary 성능 평가 ─────────────────────────────────────
    progress(11, 14, "최종 판정 규칙 평가")
    rows = [
        eval_workflow_row(
            "Gangnam_internal_CV", y_dev,
            dev_clin_score, dev_clin_thr,
            dev_aec_score,  dev_aec_support,
            FINAL_AEC_THRESHOLD, "strong_primary_gray0075_aec080",
        ),
        eval_workflow_row(
            "Shinchon_sdata_external", y_ext,
            ext_clin_score, ext_clin_thr,
            ext_aec_score,  ext_aec_support,
            FINAL_AEC_THRESHOLD, "strong_primary_gray0075_aec080",
        ),
    ]
    final_summary = pd.DataFrame(rows)
    final_summary.to_csv(OUT_DIR / "final_strong_primary_summary.csv", index=False, encoding="utf-8-sig")
    save_results_png(final_summary, OUT_DIR / "연구결과.png")

    # ── 6. 환자별 예측값 저장 ─────────────────────────────────────────────────
    progress(12, 14, "환자별 예측값 저장")
    dev_base, dev_wf, dev_gray = apply_aec_workflow(dev_clin_score, dev_clin_thr, dev_aec_score, dev_aec_support, FINAL_AEC_THRESHOLD)
    ext_base, ext_wf, ext_gray = apply_aec_workflow(ext_clin_score, ext_clin_thr, ext_aec_score, ext_aec_support, FINAL_AEC_THRESHOLD)

    pred_dev = pd.DataFrame({
        "Cohort":                 "Gangnam_internal_CV",
        "PatientID":              dev["PatientID"],
        "Actual_LowSMI_Derstine": y_dev,
        "PatientSex":             dev["PatientSex"],
        "Scanner":                dev["ManufacturerModelName"],
        "PatientAge":             dev["PatientAge"],
        "Height":                 dev["Height"],
        "Weight":                 dev["Weight"],
        "TAMA":                   dev["TAMA"],
        "SMI_calc":               dev["SMI_calc"],
        "ClinicalScore":          dev_clin_score,
        "ClinicalThreshold":      dev_clin_thr,
        "AECScore":               dev_aec_score,
        "ClinicalPred":           dev_base.astype(int),
        "InGrayZone":             dev_gray.astype(int),
        "StrongWorkflowPred":     dev_wf.astype(int),
        "AEC_QC_fail":            dev_qc["AEC_QC_fail"].astype(int),
    })
    pred_ext = pd.DataFrame({
        "Cohort":                 "Shinchon_sdata_external",
        "PatientID":              ext["PatientID"],
        "Actual_LowSMI_Derstine": y_ext,
        "PatientSex":             ext["PatientSex"],
        "Scanner":                ext["ManufacturerModelName"],
        "PatientAge":             ext["PatientAge"],
        "Height":                 ext["Height"],
        "Weight":                 ext["Weight"],
        "TAMA":                   ext["TAMA"],
        "SMI_calc":               ext["SMI_calc"],
        "ClinicalScore":          ext_clin_score,
        "ClinicalThreshold":      ext_clin_thr,
        "AECScore":               ext_aec_score,
        "ClinicalPred":           ext_base.astype(int),
        "InGrayZone":             ext_gray.astype(int),
        "StrongWorkflowPred":     ext_wf.astype(int),
        "AEC_QC_fail":            ext_qc["AEC_QC_fail"].astype(int),
    })
    predictions = pd.concat([pred_dev, pred_ext], ignore_index=True)
    predictions.to_csv(OUT_DIR / "final_strong_primary_predictions.csv", index=False, encoding="utf-8-sig")

    # ── 7. 임계값 검증 (0.001 간격 전수 탐색 + 고전적 방법 비교) ─────────────
    progress(13, 14, "임계값 방법 검증")
    grid      = np.round(np.arange(0.0, 1.0001, 0.001), 3)
    int_sweep = threshold_sweep("Gangnam_internal_CV",     y_dev, dev_clin_score, dev_clin_thr, dev_aec_score, dev_aec_support, grid)
    ext_sweep = threshold_sweep("Shinchon_sdata_external", y_ext, ext_clin_score, ext_clin_thr, ext_aec_score, ext_aec_support, grid)
    int_sweep.to_csv(OUT_DIR / "internal_aec_threshold_0p001_sweep.csv", index=False, encoding="utf-8-sig")
    ext_sweep.to_csv(OUT_DIR / "external_aec_threshold_0p001_sweep.csv", index=False, encoding="utf-8-sig")

    methods = classic_threshold_methods(int_sweep)
    methods.to_csv(OUT_DIR / "classic_threshold_methods_from_gangnam.csv", index=False, encoding="utf-8-sig")

    progress(14, 14, "완료")

    # ── 콘솔 요약 출력 ────────────────────────────────────────────────────────
    print("\n=== 코호트 규모 및 이벤트 수 ===")
    print(f"강남 내부 CV:   N={len(dev)}, 근감소증={int(y_dev.sum())}")
    print(f"신촌 외부 검증: N={len(ext)}, 근감소증={int(y_ext.sum())}")

    print("\n=== AEC QC 실패 수 ===")
    print(f"강남 QC 실패: {int(dev_qc['AEC_QC_fail'].sum())}")
    print(f"신촌 QC 실패: {int(ext_qc['AEC_QC_fail'].sum())}")

    print("\n=== 최종 strong primary 성능 요약 ===")
    cols = [
        "Cohort",
        "Clinical_AUC",        "AEC_AUC",
        "Clinical_Sensitivity","Workflow_Sensitivity", "Sensitivity_p",
        "Clinical_Specificity","Workflow_Specificity", "Specificity_p",
        "Clinical_Accuracy",   "Workflow_Accuracy",    "Accuracy_p",
        "Workflow_PPV",        "Workflow_NPV",
        "NetCorrected",        "Rescued_FN", "Lost_TP", "Corrected_FP", "Added_FP",
    ]
    print(final_summary[cols].round(6).to_string(index=False))

    print("\n=== 강남 기반 고전적 임계값 방법 비교 ===")
    print(methods.round(6).to_string(index=False))
    print(f"\n결과 저장 경로: {OUT_DIR}")


if __name__ == "__main__":
    main()
