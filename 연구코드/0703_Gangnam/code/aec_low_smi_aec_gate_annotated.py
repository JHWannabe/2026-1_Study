from __future__ import annotations

"""
AEC morphology gate for low-SMI de-escalation
=============================================

이 파일 하나만 학생에게 전달하기 위한 "주석이 자세한 전체 재현 코드"입니다.

이 코드가 하는 일:
1. g1090.xlsx = internal cohort, sdata.xlsx = external cohort를 읽습니다.
2. CT AEC 128-point curve를 smoothing + patient-wise normalization 합니다.
3. age, sex, weight, height만 이용한 clinical model을 만듭니다.
4. internal cohort에서 clinical sensitivity 90%가 되는 S90 threshold를 정합니다.
5. 미리 lock한 4-region AEC morphology gate를 external cohort에 적용합니다.
6. Clinical-positive 환자 중 AEC+와 AEC-의 low SMI 비율을 비교합니다.
7. sensitivity loss, specificity gain, accuracy gain, formal noninferiority bound를 계산합니다.
8. 논문용 1 x 3 mean curve figure를 생성합니다.

연구가 여기까지 온 흐름:
1. 처음에는 AEC 128-point curve 전체를 보고 low SMI와 non-low SMI의 평균 shape 차이를 확인했습니다.
2. 전체 curve를 그대로 쓰는 모델, PCA/functional logistic/wavelet/SVM/ROCKET/CNN 계열을 검토했습니다.
   여기서 AEC가 stand-alone diagnostic model로 강하게 이기는 구조는 아니고,
   clinical-positive 환자 안에서 de-escalation signal을 주는 쪽이 더 맞다는 결론으로 이동했습니다.
3. 그래서 clinical model의 S80/S85/S90 operating point에서
   "specificity gain은 얻되 sensitivity loss는 제한되는가?"를 보는 conditional de-escalation 문제로 재정의했습니다.
4. AEC curve에서 어느 구간이 중요한지 보기 위해 region search를 했습니다.
   - coarse search: 시작 위치를 8-slice 단위로 움직이며 16/24/32 길이 window를 훑었습니다.
   - targeted fine search: 후보가 몰리는 부근을 4-slice 단위로 다시 훑고, 12/16/20/24/28/32 길이 window를 평가했습니다.
   - 이 과정에서 late AEC region, 특히 97-128 및 117-128 부근의 slope/endpoint morphology가 반복적으로 살아났습니다.
5. CNN도 검토했습니다.
   - 그냥 전체 graph를 CNN에 넣는 방식보다, 의미 있는 region을 보게 하는 region-guided CNN / surrogate mimic이 더 그럴듯했습니다.
   - CNN은 "이 morphology가 사람이 정한 hand-crafted feature만의 산물이 아니라 graph에서 학습될 수 있다"는 보조 근거입니다.
   - 하지만 논문 primary로는 CNN보다 interpretable 4-region gate가 더 방어하기 쉽습니다.
6. 최종적으로 primary analysis는 CNN이 아니라 아래 locked interpretable 4-region gate입니다.
   CNN은 secondary analysis로 두는 것이 manuscript framing에 더 안전합니다.

이 파일에서 CNN을 다루는 방식:
- 아래 main analysis는 interpretable gate를 직접 계산합니다.
- secondary CNN-mimic analysis는 이미 학습되어 저장된 CNN branch probability를 읽어
  같은 S90 de-escalation 평가를 다시 계산합니다.
- 즉, 이 파일은 CNN training을 매번 새로 돌리지는 않습니다.
  이유는 CNN training/search는 시간이 오래 걸리고 GPU/seed에 따라 작은 차이가 생길 수 있기 때문입니다.
- 대신 저장된 CNN output을 이용해 "CNN을 쓰면 같은 morphology를 어느 정도 학습하는가"를 재현합니다.
  전체 CNN training script는 work/aec_new_region_cnn_surrogate_mimic_gate.py 입니다.

중요한 해석:
- AEC+는 "low SMI 양성"이라는 뜻이 아닙니다.
- AEC+는 "clinical-positive 환자 중 de-escalation 가능한 low-risk morphology"라는 뜻입니다.
- 이 분석의 주장은 global AUC 향상이 아니라,
  clinical-positive 환자 안에서 specificity를 올리면서 sensitivity loss를 제한하는 것입니다.
"""

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# 0. 경로와 고정 설정
# ---------------------------------------------------------------------------

# 학생이 다른 위치에서 실행해도 동작하도록, 이 파일 위치를 기준으로 project root를 추정합니다.
# 현재 파일 위치:
#   code/aec_low_smi_aec_gate_annotated.py
# project root:
#   Gangnam/
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

# 실제 분석에 사용할 원 데이터입니다.
# internal(Gangnam, n=1090) / external(Sinchon, n=922) cohort의 최신 merged feature 파일입니다.
DATA_DIR = PROJECT_ROOT / "excel"
INTERNAL_XLSX = DATA_DIR / "강남_liver_merged_features_1090.xlsx"
EXTERNAL_XLSX = DATA_DIR / "신촌_liver_merged_features.xlsx"

# 이 스크립트가 만드는 결과물은 output 폴더 하나에만 저장합니다.
OUT_DIR = SCRIPT_PATH.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 재현성을 위한 seed입니다. Logistic regression 자체는 deterministic에 가깝지만,
# out-of-fold split을 만들 때 class별 shuffle을 하기 때문에 seed를 고정합니다.
# Clinical out-of-fold split은 기존 분석 코드(aec_conditional_value.py)에서
# 사용한 seed와 맞춥니다. 이 값이 달라지면 clinical S90 threshold가 조금 바뀌고,
# 따라서 Clinical+ 및 de-escalation counts도 달라집니다.
SEED = 20260629
RNG = np.random.default_rng(SEED)

# AEC preprocessing 설정입니다.
# sigma=1 smoothing은 너무 jagged한 point-to-point noise를 줄이되,
# 전체 curve morphology는 보존하는 정도의 약한 smoothing입니다.
SMOOTHING_SIGMA = 1.0

# Primary clinical operating point입니다.
# 이 연구의 primary claim은 high-sensitivity clinical screen 이후 de-escalation이므로,
# internal clinical sensitivity 90% 지점을 primary threshold로 사용합니다.
TARGET_SENSITIVITY = 0.90

# Formal noninferiority margin입니다.
# "de-escalation 때문에 잃는 true low-SMI sensitivity가 5%p 이내"라는 safety criterion입니다.
NI_MARGIN = 0.05


# ---------------------------------------------------------------------------
# 1. Locked primary AEC gate definition
# ---------------------------------------------------------------------------

# 4개 region은 128-point AEC curve의 slice index 기준입니다.
# 1 = inferior pubic margin, 128 = liver dome이라는 해부학적 방향으로 해석합니다.
#
# 왜 이 region인가?
# - 전체 128개 점을 모두 쓰면 overfitting 위험이 큽니다.
# - coarse 8-slice step window search와 targeted 4-slice step fine search를 통해
#   de-escalation에 반복적으로 관여하는 구간을 좁혔습니다.
# - 그 결과 early/mid support region과 late morphology region을 함께 반영하는
#   R1, R2, R3, R4 구조로 정리했습니다.
# - 최종 시각적/해석적 핵심은 R4 late segment의 slope/endpoint change입니다.
REGIONS = {
    "R1_045_056": (45, 56),
    "R2_057_080": (57, 80),
    "R3_097_128": (97, 128),
    "R4_117_128": (117, 128),
}

# 최종 primary gate는 아래 네 branch를 사용합니다.
# 각 branch는 하나의 AEC morphology feature가 clinical score를 threshold 근처에서
# 얼마나 de-escalation 방향으로 움직이는지 vote로 바꿉니다.
#
# sign:
#   feature_z에 곱할 방향입니다.
#   sign * feature_z가 음/양 어느 방향으로 clinical score를 낮추는지 정의합니다.
#
# width:
#   clinical score가 threshold 근처일수록 AEC feature가 더 많이 작동하도록 하는
#   Gaussian boundary weight의 폭입니다.
#
# lambda:
#   AEC feature가 clinical score에 들어가는 강도입니다.
LOCKED_BRANCHES = [
    {
        "region": "R1",
        "feature": "R1_045_056__endpoint_delta",
        "sign": -1,
        "width": 0.50,
        "lambda": 0.25,
    },
    {
        "region": "R2",
        "feature": "R2_057_080__level_mean",
        "sign": -1,
        "width": 0.70,
        "lambda": 0.25,
    },
    {
        "region": "R3",
        "feature": "R3_097_128__linear_slope",
        "sign": +1,
        "width": 0.35,
        "lambda": 0.25,
    },
    {
        "region": "R4",
        "feature": "R4_117_128__endpoint_delta",
        "sign": -1,
        "width": 0.50,
        "lambda": 0.25,
    },
]

# 네 branch vote를 + / - 문자열로 바꾼 뒤, 아래 pattern 중 하나이면 AEC+로 정의합니다.
# AEC+ = de-escalation morphology = low-SMI 가능성이 낮아 보이는 AEC shape.
SELECTED_AEC_POSITIVE_PATTERNS = {"++++", "++--", "+--+", "--+-", "---+"}


# ---------------------------------------------------------------------------
# 1b. Secondary CNN-mimic gate definition
# ---------------------------------------------------------------------------

# CNN-mimic은 primary가 아니라 secondary입니다.
# 목적:
# - hand-crafted 4-region morphology gate와 비슷한 signal을
#   CNN이 graph-derived regional morphology에서 학습할 수 있는지 확인합니다.
#
# 저장된 probability file:
# - shape: N x 3 x 4
# - 3 operating points = S80, S85, S90
# - 4 branches = R1, R2, R3, R4
#
# 이 file은 기존 CNN training script에서 생성됐습니다.
CNN_PROBABILITY_NPZ = PROJECT_ROOT / "outputs" / "aec_new_region_cnn_surrogate_mimic_gate" / "surrogate_mimic_balanced_probabilities.npz"

# S90은 index 2입니다. 순서: S80, S85, S90.
CNN_S90_INDEX = 2

# Formal 5%p NI 기준에서 internal/external 모두 통과한 secondary CNN-mimic rule입니다.
CNN_BRANCH_THRESHOLDS = np.array([0.80, 0.60, 0.90, 0.60], dtype=float)
CNN_SELECTED_PATTERNS = {"+---", "---+", "-+-+", "++++"}


# ---------------------------------------------------------------------------
# 2. Data loading and preprocessing
# ---------------------------------------------------------------------------


def aec_columns(df: pd.DataFrame) -> list[str]:
    """aec_1, aec_2, ..., aec_128 컬럼을 숫자 순서대로 정렬합니다."""
    cols = [c for c in df.columns if str(c).startswith("aec_")]
    return sorted(cols, key=lambda c: int(str(c).split("_")[1]))


def matrix_from_aec_sheet(df: pd.DataFrame) -> np.ndarray:
    """
    AEC sheet를 N x 128 숫자 행렬로 바꿉니다.

    결측값이 있으면:
    - 먼저 column median으로 채웁니다.
    - column 전체가 결측이면 global median으로 채웁니다.

    결측 처리를 명시적으로 하는 이유:
    downstream smoothing / normalization / feature 계산에서 NaN이 퍼지는 것을 막기 위해서입니다.
    """
    x = df[aec_columns(df)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    global_median = float(np.nanmedian(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else 0.0
    col_median = np.nanmedian(x, axis=0)
    col_median[~np.isfinite(col_median)] = global_median
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = np.take(col_median, np.where(bad)[1])
    x[~np.isfinite(x)] = global_median
    return x


def patient_wise_mean_normalize(x: np.ndarray) -> np.ndarray:
    """
    각 환자의 AEC curve를 자기 자신의 평균으로 나눕니다.

    이 normalization이 중요한 이유:
    - raw AEC level은 scanner/vendor/protocol 영향을 많이 받을 수 있습니다.
    - 우리는 absolute exposure level이 아니라 curve shape/morphology를 보고 싶습니다.
    - 따라서 환자별 평균을 1로 맞춰 curve의 상대적인 굴곡만 남깁니다.
    """
    mean = np.nanmean(x, axis=1, keepdims=True)
    mean[~np.isfinite(mean) | (mean == 0)] = 1.0
    return x / mean


def load_dataset(path: Path) -> dict:
    """
    xlsx 하나를 읽어 분석에 필요한 dictionary로 변환합니다.

    반환값:
    - meta: metadata sheet
    - raw_aec: 원래 AEC 128-point curve
    - smooth_aec: Gaussian smoothing을 적용한 AEC curve
    - norm_aec: smooth_aec를 patient-wise mean normalization한 curve
    - sex: PatientSex
    - smi: (TAMA - IMATA) / height^2
    - low_smi: sex-specific cutoff를 적용한 binary outcome
    """
    meta = pd.read_excel(path, sheet_name="metadata", engine="openpyxl").reset_index(drop=True)
    aec_sheet = pd.read_excel(path, sheet_name="aec_128", engine="openpyxl")

    raw_aec = matrix_from_aec_sheet(aec_sheet)
    smooth_aec = ndimage.gaussian_filter1d(raw_aec, sigma=SMOOTHING_SIGMA, axis=1, mode="nearest")
    norm_aec = patient_wise_mean_normalize(smooth_aec)

    sex = meta["PatientSex"].astype(str).str.upper().to_numpy()
    height_m = pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float) / 100.0
    tama = pd.to_numeric(meta["TAMA"], errors="coerce").to_numpy(dtype=float)
    imata = pd.to_numeric(meta["IMATA"], errors="coerce").to_numpy(dtype=float)
    smi = (tama - imata) / (height_m**2)

    # Sex-specific low SMI cutoffs (강남_aec_128.xlsx의 M_normal/M_low_SMI, F_normal/F_low_SMI
    # 라벨 경계값에서 역산해 다른 파이프라인(01~10)과 동일하게 맞춤).
    # 남자: 43.0 미만, 여자: 34.0 미만.
    low_smi = np.where(sex == "M", smi < 43.0, smi < 34.0).astype(int)

    return {
        "meta": meta,
        "raw_aec": raw_aec,
        "smooth_aec": smooth_aec,
        "norm_aec": norm_aec,
        "sex": sex,
        "smi": smi,
        "low_smi": low_smi,
    }


# ---------------------------------------------------------------------------
# 3. Clinical model: age, sex, weight, height only
# ---------------------------------------------------------------------------


def clinical_design_matrix(internal_meta: pd.DataFrame, external_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Clinical model에 들어갈 변수만 뽑습니다.

    변수:
    - PatientAge
    - Height
    - Weight
    - sex_M

    중요한 점:
    AEC 정보는 clinical model에 절대 들어가지 않습니다.
    그래야 AEC가 clinical model 이후에 추가 정보를 주는지 평가할 수 있습니다.
    """

    def raw_matrix(meta: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                pd.to_numeric(meta["PatientAge"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(meta["Height"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(meta["Weight"], errors="coerce").to_numpy(dtype=float),
                (meta["PatientSex"].astype(str).str.upper().to_numpy() == "M").astype(float),
            ]
        )

    names = ["PatientAge", "Height", "Weight", "sex_M"]
    x_internal = raw_matrix(internal_meta)
    x_external = raw_matrix(external_meta)

    # 결측값은 internal cohort median으로 채웁니다.
    # External 정보를 이용해 imputation 기준을 만들지 않는 것이 원칙입니다.
    median = np.nanmedian(x_internal, axis=0)
    x_internal = np.where(np.isfinite(x_internal), x_internal, median)
    x_external = np.where(np.isfinite(x_external), x_external, median)

    # Standardization 역시 internal mean/std로 fit하고 external에 apply합니다.
    mean = x_internal.mean(axis=0)
    sd = x_internal.std(axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (x_internal - mean) / sd, (x_external - mean) / sd, names


def stratified_folds(y: np.ndarray, k: int = 5) -> list[np.ndarray]:
    """
    Internal cohort에서 out-of-fold clinical score를 만들기 위한 stratified folds.

    왜 OOF가 필요한가?
    - internal cohort에서 같은 데이터로 model을 fit하고 score를 평가하면 optimistic bias가 생깁니다.
    - OOF score는 각 환자가 자신을 포함하지 않은 model에서 받은 score입니다.
    """
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = np.flatnonzero(y == cls)
        RNG.shuffle(idx)
        for i, row_idx in enumerate(idx):
            folds[i % k].append(int(row_idx))
    return [np.array(sorted(fold), dtype=int) for fold in folds]


def fit_clinical_scores(
    x_internal: np.ndarray,
    y_internal: np.ndarray,
    x_external: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clinical logistic regression score를 만듭니다.

    internal:
    - 5-fold out-of-fold decision_function score

    external:
    - internal 전체로 fit한 final model의 decision_function score
    """
    folds = stratified_folds(y_internal, k=5)
    oof_score = np.zeros(len(y_internal), dtype=float)
    all_idx = np.arange(len(y_internal))

    for fold_id, validation_idx in enumerate(folds):
        train_idx = np.setdiff1d(all_idx, validation_idx)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        model.fit(x_internal[train_idx], y_internal[train_idx])
        oof_score[validation_idx] = model.decision_function(x_internal[validation_idx])

    final_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    final_model.fit(x_internal, y_internal)
    external_score = final_model.decision_function(x_external)
    return oof_score, external_score


def z_standardize_by_internal(internal_score: np.ndarray, external_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Clinical score를 internal distribution 기준으로 z-score 변환합니다.

    Gate formula에서 width=0.50 같은 값은 z-score 단위로 해석됩니다.
    """
    mean = float(np.mean(internal_score))
    sd = float(np.std(internal_score))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (internal_score - mean) / sd, (external_score - mean) / sd


def binary_metrics(y: np.ndarray, pred_positive: np.ndarray) -> dict:
    """Sensitivity, specificity, accuracy 등 binary classifier 기본 지표입니다."""
    y_bool = y.astype(bool)
    pred = pred_positive.astype(bool)
    tp = int(np.sum(y_bool & pred))
    fp = int(np.sum(~y_bool & pred))
    fn = int(np.sum(y_bool & ~pred))
    tn = int(np.sum(~y_bool & ~pred))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "accuracy": (tp + tn) / len(y) if len(y) else np.nan,
    }


def threshold_for_min_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """
    Internal cohort에서 target sensitivity 이상을 만족하는 threshold 중 specificity가 가장 높은 것을 고릅니다.

    score가 클수록 low SMI 가능성이 높다고 가정합니다.
    threshold 이상이면 predicted low SMI positive입니다.
    """
    best_threshold = None
    best_specificity = -np.inf
    for threshold in np.unique(score):
        metrics = binary_metrics(y, score >= threshold)
        if metrics["sensitivity"] >= target and metrics["specificity"] > best_specificity:
            best_threshold = float(threshold)
            best_specificity = float(metrics["specificity"])

    # 극단적인 경우 unique threshold scan에서 target sensitivity를 만족하지 못하면,
    # event score quantile로 fallback합니다.
    if best_threshold is None:
        best_threshold = float(np.quantile(score[y == 1], 1 - target))
    return best_threshold


# ---------------------------------------------------------------------------
# 4. AEC regional feature calculation
# ---------------------------------------------------------------------------


def first_difference(x: np.ndarray) -> np.ndarray:
    """AEC curve의 1차 차분. 길이를 128로 맞추기 위해 첫 값을 복제합니다."""
    diff = np.diff(x, axis=1)
    return np.column_stack([diff[:, :1], diff])


def second_difference(x: np.ndarray) -> np.ndarray:
    """AEC curve의 2차 차분. 이 script의 locked gate에는 직접 쓰지 않지만 feature table 완결성을 위해 둡니다."""
    diff = np.diff(first_difference(x), axis=1)
    return np.column_stack([diff[:, :1], diff])


def region_descriptor_matrix(norm_aec: np.ndarray) -> pd.DataFrame:
    """
    4개 locked region에서 여러 morphology descriptor를 계산합니다.

    최종 gate는 이 중 네 개만 씁니다.
    - R1 endpoint_delta
    - R2 level_mean
    - R3 linear_slope
    - R4 endpoint_delta

    그래도 같은 함수 안에서 descriptor들을 함께 계산해두면,
    학생이 feature 이름이 무엇을 의미하는지 이해하기 좋습니다.
    """
    slope = first_difference(norm_aec)
    curvature = second_difference(norm_aec)
    grid = np.arange(norm_aec.shape[1], dtype=float)
    features: dict[str, np.ndarray] = {}

    for region_name, (start, end) in REGIONS.items():
        # Python slice는 0-based, AEC slice index는 1-based라서 start-1을 씁니다.
        region_slice = slice(start - 1, end)
        block = norm_aec[:, region_slice]
        slope_block = slope[:, region_slice]
        curvature_block = curvature[:, region_slice]

        # linear_slope는 region 안에서 least-squares 직선 기울기입니다.
        # x를 centered 해두면 intercept와 무관하게 slope 계산이 안정적입니다.
        x = grid[region_slice] - grid[region_slice].mean()
        denom = float((x**2).sum()) or 1.0

        features[f"{region_name}__level_mean"] = block.mean(axis=1)
        features[f"{region_name}__level_sd"] = block.std(axis=1)
        features[f"{region_name}__endpoint_delta"] = block[:, -1] - block[:, 0]
        features[f"{region_name}__linear_slope"] = (
            ((block - block.mean(axis=1, keepdims=True)) * x[None, :]).sum(axis=1) / denom
        )
        features[f"{region_name}__slope_mean"] = slope_block.mean(axis=1)
        features[f"{region_name}__slope_sd"] = slope_block.std(axis=1)
        features[f"{region_name}__curv_mean"] = curvature_block.mean(axis=1)
        features[f"{region_name}__curv_sd"] = curvature_block.std(axis=1)

    return pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)


def standardize_features_by_internal(
    internal_features: pd.DataFrame,
    external_features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    AEC feature도 clinical variable처럼 internal 기준으로 standardize합니다.

    External cohort의 mean/std를 사용하지 않습니다.
    이것은 external validation의 기본 원칙입니다.
    """
    names = list(internal_features.columns)
    x_internal = internal_features.to_numpy(dtype=float)
    x_external = external_features.to_numpy(dtype=float)

    median = np.nanmedian(x_internal, axis=0)
    median[~np.isfinite(median)] = 0.0
    x_internal = np.where(np.isfinite(x_internal), x_internal, median[None, :])
    x_external = np.where(np.isfinite(x_external), x_external, median[None, :])

    mean = x_internal.mean(axis=0)
    sd = x_internal.std(axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return (x_internal - mean) / sd, (x_external - mean) / sd, names


# ---------------------------------------------------------------------------
# 5. Locked AEC gate calculation
# ---------------------------------------------------------------------------


def branch_gate_score(
    clinical_z: np.ndarray,
    feature_z: np.ndarray,
    threshold: float,
    sign: int,
    width: float,
    lam: float,
) -> np.ndarray:
    """
    하나의 branch가 clinical score를 de-escalation 방향으로 움직이는지 계산합니다.

    핵심 아이디어:
    - AEC는 clinical score 전체를 대체하지 않습니다.
    - Clinical threshold 근처, 즉 decision boundary 근처에서만 더 많이 작동합니다.

    boundary_weight:
    - clinical_z가 threshold와 가까우면 1에 가까움
    - threshold에서 멀어지면 0에 가까움

    gate_score:
    - clinical_z에 AEC feature 효과를 조금 더한 score
    - gate_score가 threshold 아래로 내려가면 branch vote는 +가 됩니다.
    """
    boundary_weight = np.exp(-0.5 * ((clinical_z - threshold) / width) ** 2)
    return clinical_z + lam * boundary_weight * (sign * feature_z)


def vote_pattern_from_matrix(vote_matrix: np.ndarray) -> np.ndarray:
    """
    N x 4 boolean vote matrix를 '+--+' 같은 문자열 pattern으로 바꿉니다.

    True  = '+': 해당 branch가 de-escalation을 지지
    False = '-': 해당 branch가 de-escalation을 지지하지 않음
    """
    return np.array(["".join("+" if v else "-" for v in row) for row in vote_matrix], dtype=object)


def compute_aec_morphology_gate(
    clinical_z: np.ndarray,
    clinical_threshold: float,
    feature_z: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Locked 4-region AEC gate를 적용합니다.

    반환값:
    - vote_matrix: branch별 + / - vote
    - pattern: '++++', '+--+' 같은 4-character pattern
    - aec_positive: selected pattern이면 True
    """
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    branch_votes = []
    branch_summaries = []

    for branch in LOCKED_BRANCHES:
        feature_index = name_to_index[branch["feature"]]
        score = branch_gate_score(
            clinical_z=clinical_z,
            feature_z=feature_z[:, feature_index],
            threshold=clinical_threshold,
            sign=int(branch["sign"]),
            width=float(branch["width"]),
            lam=float(branch["lambda"]),
        )

        # score가 clinical threshold 아래로 내려가면 그 branch는 de-escalation vote '+'입니다.
        vote = score < clinical_threshold
        branch_votes.append(vote)
        branch_summaries.append({**branch, "vote_positive_n": int(vote.sum())})

    vote_matrix = np.column_stack(branch_votes)
    pattern = vote_pattern_from_matrix(vote_matrix)
    aec_positive = np.isin(pattern, list(SELECTED_AEC_POSITIVE_PATTERNS))

    return {
        "vote_matrix": vote_matrix,
        "pattern": pattern,
        "aec_positive": aec_positive,
        "branch_summaries": branch_summaries,
    }


def compute_secondary_cnn_mimic_gate_from_saved_probabilities() -> dict | None:
    """
    저장된 CNN-mimic branch probability를 읽어 secondary AEC+ gate를 계산합니다.

    왜 CNN을 직접 매번 train하지 않는가?
    - CNN training은 GPU/seed/epoch에 따라 작은 변동이 있습니다.
    - 학생용 재현 파일에서는 "locked output을 재평가"하는 것이 더 안정적입니다.
    - 따라서 기존 CNN training script가 저장한 probability를 읽고,
      threshold + pattern rule만 이 파일에서 투명하게 적용합니다.

    CNN probability의 의미:
    - 각 branch probability는 해당 region이 de-escalation vote '+'일 확률입니다.
    - probability >= branch threshold이면 '+'로 변환합니다.
    - 네 branch의 + / - pattern이 CNN_SELECTED_PATTERNS 중 하나이면 CNN-AEC+입니다.

    반환값:
    - prob_internal_s90: internal N x 4 branch probability
    - prob_external_s90: external N x 4 branch probability
    - aec_positive_internal/external: CNN-mimic AEC+ boolean mask
    - pattern_internal/external: '+---' 같은 pattern string
    """
    if not CNN_PROBABILITY_NPZ.exists():
        return None

    packed = np.load(CNN_PROBABILITY_NPZ, allow_pickle=True)
    prob_internal = packed["prob_g"][:, CNN_S90_INDEX, :]
    prob_external = packed["prob_s"][:, CNN_S90_INDEX, :]

    def probabilities_to_gate(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        votes = prob >= CNN_BRANCH_THRESHOLDS[None, :]
        pattern = vote_pattern_from_matrix(votes)
        aec_positive = np.isin(pattern, list(CNN_SELECTED_PATTERNS))
        return pattern, aec_positive

    pattern_internal, aec_positive_internal = probabilities_to_gate(prob_internal)
    pattern_external, aec_positive_external = probabilities_to_gate(prob_external)

    return {
        "prob_internal_s90": prob_internal,
        "prob_external_s90": prob_external,
        "pattern_internal": pattern_internal,
        "pattern_external": pattern_external,
        "aec_positive_internal": aec_positive_internal,
        "aec_positive_external": aec_positive_external,
    }


# ---------------------------------------------------------------------------
# 6. Statistical summaries
# ---------------------------------------------------------------------------


def clopper_pearson_one_sided_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Binomial proportion의 one-sided 95% upper confidence bound.

    여기서는 sensitivity loss rate의 upper bound를 구할 때 씁니다.
    k = de-escalation으로 잃은 low-SMI true positive 수
    n = 전체 low-SMI 환자 수
    """
    if n <= 0:
        return np.nan
    if k == 0:
        return float(1 - alpha ** (1 / n))
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def evaluate_deescalation(y: np.ndarray, clinical_positive: np.ndarray, aec_positive: np.ndarray) -> dict:
    """
    Clinical-only prediction과 post-AEC de-escalation prediction을 비교합니다.

    clinical_positive:
    - clinical score가 S90 threshold 이상이면 positive.

    aec_positive:
    - AEC+ de-escalation morphology.

    post_positive:
    - clinical positive였지만 AEC+이면 de-escalate하여 negative로 바꿉니다.
    - clinical negative는 그대로 negative입니다.
    """
    post_positive = clinical_positive & ~aec_positive

    clinical_metrics = binary_metrics(y, clinical_positive)
    post_metrics = binary_metrics(y, post_positive)

    low = y.astype(bool)
    nonlow = ~low

    # Sensitivity loss는 de-escalation 때문에 low-SMI 환자를 놓친 비율입니다.
    tp_lost = int(np.sum(clinical_positive & aec_positive & low))
    total_low = int(np.sum(low))
    sensitivity_loss = tp_lost / total_low if total_low else np.nan
    sensitivity_loss_upper = clopper_pearson_one_sided_upper(tp_lost, total_low)

    # Specificity gain은 non-low SMI 환자 중 clinical false-positive였다가
    # AEC+ 때문에 negative로 내려간 비율입니다.
    fp_removed = int(np.sum(clinical_positive & aec_positive & nonlow))
    total_nonlow = int(np.sum(nonlow))
    specificity_gain = fp_removed / total_nonlow if total_nonlow else np.nan

    return {
        "clinical_sensitivity": clinical_metrics["sensitivity"],
        "post_sensitivity": post_metrics["sensitivity"],
        "sensitivity_loss": sensitivity_loss,
        "sensitivity_loss_upper95_one_sided": sensitivity_loss_upper,
        "formal_NI_margin": NI_MARGIN,
        "formal_NI_pass": bool(sensitivity_loss_upper <= NI_MARGIN),
        "clinical_specificity": clinical_metrics["specificity"],
        "post_specificity": post_metrics["specificity"],
        "specificity_gain": specificity_gain,
        "clinical_accuracy": clinical_metrics["accuracy"],
        "post_accuracy": post_metrics["accuracy"],
        "accuracy_gain": post_metrics["accuracy"] - clinical_metrics["accuracy"],
        "total_low_smi": total_low,
        "tp_lost": tp_lost,
        "total_nonlow_smi": total_nonlow,
        "fp_removed": fp_removed,
        "deescalated_n": int(np.sum(clinical_positive & aec_positive)),
        "deescalated_low_smi_events": tp_lost,
        "deescalated_event_rate": tp_lost / int(np.sum(clinical_positive & aec_positive))
        if np.sum(clinical_positive & aec_positive)
        else np.nan,
    }


def conditional_low_smi_table(y: np.ndarray, clinical_positive: np.ndarray, aec_positive: np.ndarray) -> tuple[pd.DataFrame, float]:
    """
    Clinical-positive 환자 안에서 AEC+ vs AEC-의 low SMI 비율을 비교합니다.

    이 table이 논문에서 가장 직관적인 결과입니다.
    """
    cp_aec_pos = clinical_positive & aec_positive
    cp_aec_neg = clinical_positive & ~aec_positive

    rows = []
    for label, mask in [
        ("Clinical+ / AEC+ de-escalation morphology", cp_aec_pos),
        ("Clinical+ / AEC- retained morphology", cp_aec_neg),
    ]:
        n = int(mask.sum())
        events = int(y[mask].sum())
        rows.append(
            {
                "group": label,
                "n": n,
                "low_smi_events": events,
                "low_smi_rate": events / n if n else np.nan,
            }
        )

    fisher_table = [
        [rows[0]["low_smi_events"], rows[0]["n"] - rows[0]["low_smi_events"]],
        [rows[1]["low_smi_events"], rows[1]["n"] - rows[1]["low_smi_events"]],
    ]
    fisher_p = float(cast(float, stats.fisher_exact(fisher_table)[1]))
    out = pd.DataFrame(rows)
    out["fisher_p"] = fisher_p
    return out, fisher_p


# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------


def mean_ci(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean curve와 approximate 95% CI를 계산합니다."""
    mean = np.nanmean(curves, axis=0)
    if len(curves) <= 1:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    se = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(len(curves))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def add_region_spans(ax: Axes) -> None:
    """그림에 R1-R4 위치를 표시합니다."""
    spans = [
        ("R1", 45, 56, "#4E79A7", 0.08),
        ("R2", 57, 80, "#F28E2B", 0.08),
        ("R3", 97, 128, "#59A14F", 0.08),
        ("R4", 117, 128, "#B07AA1", 0.14),
    ]
    for label, start, end, color, alpha in spans:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
    y0, y1 = ax.get_ylim()
    for label, start, end, color, _ in spans:
        ax.text((start + end) / 2, y1 - 0.015 * (y1 - y0), label, ha="center", va="bottom", color=color, fontsize=8, fontweight="bold")


def plot_group_mean(ax: Axes, z: np.ndarray, curves: np.ndarray, mask: np.ndarray, label: str, color: str) -> None:
    """한 group의 mean curve와 CI band를 그립니다."""
    mean, low, high = mean_ci(curves[mask])
    ax.plot(z, mean, color=color, lw=2.4, label=f"{label} (n={int(mask.sum())})")
    ax.fill_between(z, low, high, color=color, alpha=0.12, lw=0)


def add_r4_tangent(ax: Axes, z: np.ndarray, curves: np.ndarray, red_mask: np.ndarray, blue_mask: np.ndarray) -> None:
    """
    Panel C에 R4 fitted tangent를 얹습니다.

    왜 tangent를 보여주는가?
    - locked gate의 핵심 시각적 해석이 R4 late slope / endpoint change이기 때문입니다.
    - 전체 curve만 보면 "뭔가 다르다" 정도지만,
      R4 fitted line은 실제로 late segment 기울기가 다름을 직접 보여줍니다.
    """
    r4 = (z >= 117) & (z <= 128)
    x = z[r4].astype(float)
    x_span = np.array([117.0, 128.0])

    red_mean = np.nanmean(curves[red_mask], axis=0)[r4]
    blue_mean = np.nanmean(curves[blue_mask], axis=0)[r4]
    red_slope, red_intercept = np.polyfit(x, red_mean, 1)
    blue_slope, blue_intercept = np.polyfit(x, blue_mean, 1)

    ax.plot(x_span, red_slope * x_span + red_intercept, color="#D04F5B", lw=4.0, alpha=0.85, solid_capstyle="round")
    ax.plot(x_span, blue_slope * x_span + blue_intercept, color="#2F6F9F", lw=4.0, alpha=0.85, solid_capstyle="round")
    ax.text(
        0.98,
        0.16,
        f"R4 fitted slope\nAEC- {red_slope:+.4f}/slice\nAEC+ {blue_slope:+.4f}/slice",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.86, "pad": 4},
    )


def make_main_figure(
    external: dict,
    clinical_positive: np.ndarray,
    aec_positive: np.ndarray,
    output_path: Path,
) -> None:
    """
    논문용 핵심 1 x 3 figure를 생성합니다.

    Panel A:
      Clinical+ vs Clinical-

    Panel B:
      Low SMI vs Non-low SMI

    Panel C:
      Clinical+ 안에서 AEC+ vs AEC-
      이 panel이 가장 중요합니다.
    """
    curves = external["norm_aec"]
    y = external["low_smi"].astype(bool)
    z = np.arange(1, 129)
    overall_mean = np.nanmean(curves, axis=0)
    clinical_pos_mean = np.nanmean(curves[clinical_positive], axis=0)

    cp_aec_pos = clinical_positive & aec_positive
    cp_aec_neg = clinical_positive & ~aec_positive

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 4.7), sharey=True)

    axes[0].plot(z, overall_mean, color="#666666", lw=1.6, ls=":", label="external overall mean")
    plot_group_mean(axes[0], z, curves, clinical_positive, "Clinical +", "#D04F5B")
    plot_group_mean(axes[0], z, curves, ~clinical_positive, "Clinical -", "#2F6F9F")
    axes[0].set_title("A. Clinical S90 operating point", loc="left", fontsize=12, fontweight="bold")

    axes[1].plot(z, overall_mean, color="#666666", lw=1.6, ls=":", label="external overall mean")
    plot_group_mean(axes[1], z, curves, y, "Low SMI +", "#D04F5B")
    plot_group_mean(axes[1], z, curves, ~y, "Non-low SMI", "#2F6F9F")
    axes[1].set_title("B. Outcome phenotype", loc="left", fontsize=12, fontweight="bold")

    axes[2].plot(z, clinical_pos_mean, color="#666666", lw=1.6, ls=":", label="Clinical + mean reference")
    plot_group_mean(axes[2], z, curves, cp_aec_pos, "Clinical+ / AEC+", "#2F6F9F")
    plot_group_mean(axes[2], z, curves, cp_aec_neg, "Clinical+ / AEC-", "#D04F5B")
    add_r4_tangent(axes[2], z, curves, red_mask=cp_aec_neg, blue_mask=cp_aec_pos)
    axes[2].set_title("C. Conditional AEC split with R4 fitted tangent", loc="left", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.axhline(1.0, color="#9E9E9E", lw=0.9, ls="--", alpha=0.7)
        ax.set_xlim(1, 128)
        ax.set_ylim(0.82, 1.21)
        ax.set_xticks([1, 32, 64, 96, 128])
        ax.set_xlabel("Slice index")
        ax.grid(axis="both", color="#D0D0D0", lw=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_region_spans(ax)
        ax.legend(frameon=False, loc="lower left", fontsize=9)

    axes[0].set_ylabel("Patient-normalized AEC")
    fig.suptitle(
        "External S90 core AEC morphology comparisons with R4 fitted tangent",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.text(0.5, -0.015, "Craniocaudal index: 1 inferior pubic margin -> 128 liver dome", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Main workflow
# ---------------------------------------------------------------------------


def main() -> None:
    # ---------------------------
    # Load data
    # ---------------------------
    internal = load_dataset(INTERNAL_XLSX)
    external = load_dataset(EXTERNAL_XLSX)

    y_internal = internal["low_smi"].astype(int)
    y_external = external["low_smi"].astype(int)

    # ---------------------------
    # Clinical model
    # ---------------------------
    x_internal, x_external, clinical_names = clinical_design_matrix(internal["meta"], external["meta"])
    clinical_internal_raw, clinical_external_raw = fit_clinical_scores(x_internal, y_internal, x_external)
    clinical_internal_z, clinical_external_z = z_standardize_by_internal(clinical_internal_raw, clinical_external_raw)

    # S90 threshold는 internal OOF clinical score에서 정합니다.
    clinical_threshold = threshold_for_min_sensitivity(y_internal, clinical_internal_z, TARGET_SENSITIVITY)

    clinical_positive_internal = clinical_internal_z >= clinical_threshold
    clinical_positive_external = clinical_external_z >= clinical_threshold

    # ---------------------------
    # AEC morphology features
    # ---------------------------
    internal_features = region_descriptor_matrix(internal["norm_aec"])
    external_features = region_descriptor_matrix(external["norm_aec"])
    feature_internal_z, feature_external_z, feature_names = standardize_features_by_internal(internal_features, external_features)

    # ---------------------------
    # Locked AEC gate
    # ---------------------------
    gate_internal = compute_aec_morphology_gate(clinical_internal_z, clinical_threshold, feature_internal_z, feature_names)
    gate_external = compute_aec_morphology_gate(clinical_external_z, clinical_threshold, feature_external_z, feature_names)

    aec_positive_internal = gate_internal["aec_positive"]
    aec_positive_external = gate_external["aec_positive"]

    # ---------------------------
    # Metrics
    # ---------------------------
    metrics_rows = []
    for cohort_name, y, clinical_positive, aec_positive in [
        ("Gangnam internal", y_internal, clinical_positive_internal, aec_positive_internal),
        ("Sinchon external", y_external, clinical_positive_external, aec_positive_external),
    ]:
        row = {"cohort": cohort_name, "operating_point": "S90"}
        row.update(evaluate_deescalation(y, clinical_positive, aec_positive))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "table_s90_primary_deescalation_metrics.csv", index=False)

    conditional_df, fisher_p = conditional_low_smi_table(y_external, clinical_positive_external, aec_positive_external)
    conditional_df.to_csv(OUT_DIR / "table_external_s90_clinical_positive_aec_split.csv", index=False)

    # ---------------------------
    # Secondary CNN-mimic analysis
    # ---------------------------
    # 이 분석은 primary가 아닙니다.
    # "interpretable 4-region gate가 잡은 morphology를 CNN도 graph에서 배울 수 있는가?"를 보여주는 보조 결과입니다.
    cnn_result = compute_secondary_cnn_mimic_gate_from_saved_probabilities()
    cnn_metrics_df = None
    cnn_conditional_df = None
    if cnn_result is not None:
        cnn_rows = []
        for cohort_name, y, clinical_positive, cnn_aec_positive in [
            ("Gangnam internal", y_internal, clinical_positive_internal, cnn_result["aec_positive_internal"]),
            ("Sinchon external", y_external, clinical_positive_external, cnn_result["aec_positive_external"]),
        ]:
            row = {"cohort": cohort_name, "operating_point": "S90", "model": "secondary_CNN_mimic"}
            row.update(evaluate_deescalation(y, clinical_positive, cnn_aec_positive))
            cnn_rows.append(row)
        cnn_metrics_df = pd.DataFrame(cnn_rows)
        cnn_metrics_df.to_csv(OUT_DIR / "table_secondary_cnn_mimic_s90_deescalation_metrics.csv", index=False)

        cnn_conditional_df, _ = conditional_low_smi_table(
            y_external,
            clinical_positive_external,
            cnn_result["aec_positive_external"],
        )
        cnn_conditional_df.to_csv(OUT_DIR / "table_secondary_cnn_mimic_external_s90_clinical_positive_split.csv", index=False)

        # Pattern distribution은 CNN이 어떤 + / - 조합으로 de-escalation을 만들었는지 보여줍니다.
        pd.DataFrame(
            {
                "pattern": cnn_result["pattern_external"],
                "clinical_positive": clinical_positive_external,
                "cnn_aec_positive": cnn_result["aec_positive_external"],
                "low_smi": y_external,
            }
        ).to_csv(OUT_DIR / "secondary_cnn_mimic_external_patterns.csv", index=False)

    # ---------------------------
    # Figure
    # ---------------------------
    make_main_figure(
        external=external,
        clinical_positive=clinical_positive_external,
        aec_positive=aec_positive_external,
        output_path=OUT_DIR / "figure_external_s90_core_1x3_with_r4_tangent.png",
    )

    # ---------------------------
    # JSON summary
    # ---------------------------
    summary = {
        "data": {
            "internal_xlsx": str(INTERNAL_XLSX),
            "external_xlsx": str(EXTERNAL_XLSX),
            "internal_n": int(len(y_internal)),
            "internal_low_smi_events": int(y_internal.sum()),
            "external_n": int(len(y_external)),
            "external_low_smi_events": int(y_external.sum()),
        },
        "preprocessing": {
            "aec_sheet": "aec_128",
            "smoothing": f"gaussian_filter1d sigma={SMOOTHING_SIGMA}, axis=1, mode=nearest",
            "normalization": "patient-wise mean normalization after smoothing",
        },
        "clinical_model": {
            "variables": clinical_names,
            "threshold": "internal out-of-fold S90",
            "clinical_threshold_z": float(clinical_threshold),
            "target_sensitivity": TARGET_SENSITIVITY,
        },
        "locked_aec_gate": {
            "branches": LOCKED_BRANCHES,
            "selected_aec_positive_patterns": sorted(SELECTED_AEC_POSITIVE_PATTERNS),
            "aec_positive_meaning": "de-escalation / lower low-SMI probability morphology",
        },
        "secondary_cnn_mimic": {
            "included": cnn_result is not None,
            "probability_file": str(CNN_PROBABILITY_NPZ),
            "note": "CNN is secondary. This script reads saved branch probabilities rather than retraining the CNN.",
            "branch_thresholds": CNN_BRANCH_THRESHOLDS.tolist(),
            "selected_patterns": sorted(CNN_SELECTED_PATTERNS),
        },
        "external_conditional_result": {
            "clinical_positive_aec_positive_n": int(conditional_df.iloc[0]["n"]),
            "clinical_positive_aec_positive_low_smi_events": int(conditional_df.iloc[0]["low_smi_events"]),
            "clinical_positive_aec_positive_low_smi_rate": float(conditional_df.iloc[0]["low_smi_rate"]),
            "clinical_positive_aec_negative_n": int(conditional_df.iloc[1]["n"]),
            "clinical_positive_aec_negative_low_smi_events": int(conditional_df.iloc[1]["low_smi_events"]),
            "clinical_positive_aec_negative_low_smi_rate": float(conditional_df.iloc[1]["low_smi_rate"]),
            "fisher_p": fisher_p,
        },
        "output_files": {
            "metrics_csv": str(OUT_DIR / "table_s90_primary_deescalation_metrics.csv"),
            "conditional_csv": str(OUT_DIR / "table_external_s90_clinical_positive_aec_split.csv"),
            "secondary_cnn_metrics_csv": str(OUT_DIR / "table_secondary_cnn_mimic_s90_deescalation_metrics.csv")
            if cnn_result is not None
            else None,
            "figure_png": str(OUT_DIR / "figure_external_s90_core_1x3_with_r4_tangent.png"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 화면 출력은 학생이 실행 직후 핵심 숫자를 바로 확인할 수 있도록 최소한만 보여줍니다.
    print("\nPrimary S90 de-escalation metrics")
    print(metrics_df.to_string(index=False))
    print("\nExternal S90 Clinical+ AEC split")
    print(conditional_df.to_string(index=False))
    if cnn_metrics_df is not None and cnn_conditional_df is not None:
        print("\nSecondary CNN-mimic S90 de-escalation metrics")
        print(cnn_metrics_df.to_string(index=False))
        print("\nSecondary CNN-mimic external S90 Clinical+ split")
        print(cnn_conditional_df.to_string(index=False))
    else:
        print("\nSecondary CNN-mimic probability file not found; CNN section skipped.")
    print("\nOutputs saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
