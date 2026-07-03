"""
Final reproducible code for the meaning-aware contrastive AEC model.

핵심 목적
---------
이 스크립트는 논문/학생 전달용 최종 분석 코드입니다.
탐색 과정에서 시도했던 여러 모델은 모두 제거하고, 최종적으로 사용할
"meaning-aware contrastive AEC selective correction" 모델 하나만 재현합니다.

연구 질문
---------
Clinical model(age, sex, height, weight)이 low SMI를 예측한 뒤,
clinical score가 애매한 환자에서 AEC curve가 clinical model의 오류를
줄일 수 있는가?

최종 모델 개념
--------------
1. Clinical baseline:
   - age + sex + height + weight로 low SMI를 예측합니다.
   - threshold는 training fold에서 Youden index로 정합니다.

2. Meaning-aware contrastive AEC score:
   - 각 환자마다 training set 안에서 clinical profile이 비슷한 사람을 찾습니다.
   - clinical profile = age, sex, height, weight, BMI.
   - 그중 low SMI 이웃들과 normal 이웃들을 따로 잡습니다.
   - 환자의 AEC signature가 low SMI 이웃에 가까운지, normal 이웃에 가까운지 계산합니다.
   - 즉 "clinical로 비슷한 사람들 사이에서도 AEC가 low SMI 쪽을 닮는가?"를 점수화합니다.

3. Selective correction workflow:
   - clinical positive인데 AEC가 normal 쪽으로 강하면 rule-out합니다.
   - clinical negative인데 AEC가 low SMI 쪽으로 강하면 rule-in합니다.
   - 단, clinical score가 threshold 근처인 gray-zone 안에서만 AEC를 적용합니다.

중요한 방어 논리
----------------
Gangnam internal CV에서만 threshold를 선택하고, Shinchon external에는
선택된 threshold를 그대로 적용합니다. External 결과를 보고 threshold를
조정하지 않습니다.

입력 파일
---------
1. C:/Users/user/OneDrive/1. RESEARCH/radiation/gdata.xlsx
2. C:/Users/user/OneDrive/1. RESEARCH/radiation/sdata.xlsx

필수 sheet/column
-----------------
- metadata sheet:
  PatientID, PatientAge, PatientSex, Height, Weight, TAMA,
  ManufacturerModelName 또는 Manufacturer
- aec_128 sheet:
  PatientID, aec_1 ... aec_128
  가능하면 series_description 포함

Outcome
-------
SMI_calc = TAMA / (height in meter)^2
Derstine low L3 SMI:
  male   < 45.4 cm^2/m^2
  female < 34.4 cm^2/m^2

주요 출력
---------
outputs/final_meaning_aware_contrastive_reproducible/
  - selected_thresholds.csv
  - overall_results.csv
  - subgroup_results.csv
  - subgroup_results_compact.csv
  - workflow_ruleout_rulein_counts.csv
  - patient_level_workflow.csv
  - aec_low_vs_normal_signature_clean.png
"""

from __future__ import annotations

import math
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.fftpack import dct
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Features .* are constant")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


# =============================================================================
# 0. 경로 및 상수 설정
# =============================================================================

GDATA_XLSX = r"연구코드\data\강남_liver_merged_features.xlsx"
SDATA_XLSX = r"연구코드\data\신촌_liver_merged_features.xlsx"

OUT_DIR = Path(r"연구코드\results\study_updated_1090")

SEED = 20260622

# Derstine L3 SMI criteria.
SMI_MALE_CUTOFF = 45.4
SMI_FEMALE_CUTOFF = 34.4

# Clinical model hyperparameter. This is intentionally simple and fixed.
CLINICAL_LOGIT_C = 0.7

# Correction model hyperparameters.
CORRECTION_LOGIT_C = 0.4
CORRECTION_L1_RATIO = 0.25
FEATURE_SELECT_K = 160

# Contrastive score hyperparameter:
# for each patient, use K clinically similar low-SMI and K normal neighbors.
K_NEIGHBORS_PER_CLASS = 20

T0 = time.time()


# =============================================================================
# 1. 보조 유틸리티
# =============================================================================


def progress(step: int, total: int, label: str) -> None:
    """Print a compact progress bar with elapsed seconds."""
    width = 32
    done = int(width * step / max(total, 1))
    bar = "#" * done + "-" * (width - done)
    print(f"[{bar}] {step:02d}/{total:02d} {time.time() - T0:7.1f}s  {label}", flush=True)


def p_fmt(x: float) -> str:
    """Readable p-value formatting for compact tables."""
    if pd.isna(x):
        return ""
    if float(x) < 0.001:
        return "<0.001"
    return f"{float(x):.3f}"


def compact_table(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric tables for reading while preserving the full raw CSV separately."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind in "fc":
            if col.endswith("_p") or col in {"Sensitivity_p", "Specificity_p", "Accuracy_p"}:
                out[col] = out[col].map(p_fmt)
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
    return out


def exact_mcnemar_p(a: int, b: int) -> float:
    """
    Exact two-sided McNemar p-value.

    a and b are paired discordant counts.
    For sensitivity:
      a = rescued false negatives
      b = lost true positives
    For specificity:
      a = corrected false positives
      b = added false positives
    """
    n = int(a + b)
    if n == 0:
        return 1.0
    k = min(int(a), int(b))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def confusion(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Sensitivity, specificity, accuracy, PPV, NPV and confusion counts."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.sum(y & pred))
    tn = int(np.sum(~y & ~pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "Accuracy": (tp + tn) / len(y) if len(y) else np.nan,
        "PPV": tp / (tp + fp) if (tp + fp) else np.nan,
        "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
    }


def paired_stats(y: np.ndarray, clinical_pred: np.ndarray, workflow_pred: np.ndarray) -> Dict[str, float]:
    """Paired clinical vs workflow improvement/worsening counts and p-values."""
    y = np.asarray(y).astype(bool)
    c = np.asarray(clinical_pred).astype(bool)
    w = np.asarray(workflow_pred).astype(bool)

    event = y
    nonevent = ~y
    rescued_fn = int(np.sum(event & ~c & w))
    lost_tp = int(np.sum(event & c & ~w))
    corrected_fp = int(np.sum(nonevent & c & ~w))
    added_fp = int(np.sum(nonevent & ~c & w))

    clinical_ok = c == y
    workflow_ok = w == y
    acc_improved = int(np.sum(~clinical_ok & workflow_ok))
    acc_worsened = int(np.sum(clinical_ok & ~workflow_ok))

    return {
        "Rescued_FN": rescued_fn,
        "Lost_TP": lost_tp,
        "Corrected_FP": corrected_fp,
        "Added_FP": added_fp,
        "Sensitivity_p": exact_mcnemar_p(rescued_fn, lost_tp),
        "Specificity_p": exact_mcnemar_p(corrected_fp, added_fp),
        "Accuracy_p": exact_mcnemar_p(acc_improved, acc_worsened),
        "Accuracy_improved": acc_improved,
        "Accuracy_worsened": acc_worsened,
        "NetCorrected": acc_improved - acc_worsened,
    }


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC with one-class and missing-value protection."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    ok = np.isfinite(score)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], score[ok]))


def youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """Classic threshold: maximize sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(y_true, score)
    valid = np.isfinite(thr)
    idx = int(np.argmax(tpr[valid] - fpr[valid]))
    return float(thr[valid][idx])


# =============================================================================
# 2. 데이터 로드 · 위상 분류 · 아웃컴 정의
# =============================================================================


def find_aec_cols(df: pd.DataFrame) -> List[str]:
    """Find and numerically sort columns named aec_1 ... aec_128."""
    found = []
    for c in df.columns:
        m = re.match(r"^aec[_\- ]?(\d+)$", str(c).strip(), flags=re.IGNORECASE)
        if m:
            found.append((int(m.group(1)), c))
    return [c for _, c in sorted(found)]


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate SMI from TAMA/height^2 and define Derstine low SMI."""
    df = df.copy()
    df["PatientSex"] = df["PatientSex"].astype(str).str.upper().str.strip()
    h_m = df["Height"].astype(float) / 100.0
    df["SMI_calc"] = df["TAMA"].astype(float) / np.maximum(h_m * h_m, 1e-6)
    df["LowSMI_Derstine"] = (
        (df["PatientSex"].eq("M") & (df["SMI_calc"] < SMI_MALE_CUTOFF))
        | (df["PatientSex"].eq("F") & (df["SMI_calc"] < SMI_FEMALE_CUTOFF))
    ).astype(int)
    return df


def classify_phase(text: object) -> str:
    """
    Conservative pre/post classification from series description.

    User-specified rule:
    ABD/body/GU descriptions without explicit contrast evidence are treated as
    pre/noncontrast, not left unknown.
    """
    if pd.isna(text): # type: ignore
        return "indeterminate"
    s = str(text).strip().lower()
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"\s+", " ", s)

    pre_patterns = [
        r"\bpre\b",
        r"\bpre\d",
        r"\bnon\b",
        r"\bnoncon\b",
        r"\bnon contrast\b",
        r"\bwithout\b",
        r"\bw o\b",
        r"\bwo\b",
        r"\bwio\b",
        r"\bplain\b",
        r"\bunenh",
    ]
    if any(re.search(p, s) for p in pre_patterns):
        return "pre_or_noncontrast"

    post_patterns = [
        r"\bwith contrast\b",
        r"\bpost\b",
        r"\bportal\b",
        r"\bport\b",
        r"\bhvp\b",
        r"\barterial\b",
        r"\bdelay\b",
        r"\bdelayed\b",
        r"\bce\b",
        r"\blap\b",
        r"\bcmp\b",
        r"\bcontrast\b",
    ]
    if any(re.search(p, s) for p in post_patterns):
        return "post_or_contrast"

    assumed_pre_patterns = [r"\babd\b", r"\babdomen\b", r"\bbody\b", r"\bgu\b"]
    if any(re.search(p, s) for p in assumed_pre_patterns):
        return "pre_or_noncontrast"

    return "indeterminate"


def read_series_descriptions(path: Path, cohort_key: str) -> pd.DataFrame:
    """
    Read one phase label per PatientID.

    The current workbooks usually have:
      - metadata sheet: Series_Desc
      - aec_128 sheet: series_description
    We prefer aec_128 series_description when available because it is tied to
    the AEC curve.
    """
    meta = pd.read_excel(path, sheet_name="metadata")
    meta_cols = ["PatientID"]
    if "Series_Desc" in meta.columns:
        meta_cols.append("Series_Desc")
    meta = meta[meta_cols].drop_duplicates("PatientID", keep="first")

    try:
        aec_desc = pd.read_excel(
            path,
            sheet_name="aec_128",
            usecols=lambda c: c in {"PatientID", "series_description"},
        )
        if "series_description" not in aec_desc.columns:
            aec_desc = pd.DataFrame({"PatientID": meta["PatientID"], "series_description": np.nan})
    except Exception:
        aec_desc = pd.DataFrame({"PatientID": meta["PatientID"], "series_description": np.nan})
    aec_desc = aec_desc.drop_duplicates("PatientID", keep="first")

    out = meta.merge(aec_desc, on="PatientID", how="outer", validate="one_to_one")
    out = out.rename(columns={"Series_Desc": "SeriesDescription_Metadata", "series_description": "SeriesDescription_AEC128"})
    if "SeriesDescription_Metadata" not in out.columns:
        out["SeriesDescription_Metadata"] = np.nan
    out["SeriesDescription_Used"] = out["SeriesDescription_AEC128"].where(
        out["SeriesDescription_AEC128"].notna(),
        out["SeriesDescription_Metadata"],
    )
    out["PhaseGroup"] = out["SeriesDescription_Used"].map(classify_phase)
    out["CohortSource"] = cohort_key
    return out


def attach_phase(df: pd.DataFrame, path: Path, cohort_key: str) -> pd.DataFrame:
    """Merge PhaseGroup into the patient-level dataframe."""
    phase = read_series_descriptions(path, cohort_key)
    phase["PatientID"] = pd.to_numeric(phase["PatientID"], errors="coerce")
    out = df.merge(
        phase[["PatientID", "SeriesDescription_Used", "PhaseGroup"]],
        on="PatientID",
        how="left",
        validate="many_to_one",
    )
    out["PhaseGroup"] = out["PhaseGroup"].fillna("indeterminate")
    return out


def load_cohort(path: Path, cohort_key: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load metadata + aec_128 sheet, recalculate SMI, and attach scan phase."""
    meta = pd.read_excel(path, sheet_name="metadata")
    if "PatientID" not in meta.columns:
        meta["PatientID"] = np.arange(len(meta))

    if "ManufacturerModelName" not in meta.columns:
        if "Manufacturer" in meta.columns:
            meta["ManufacturerModelName"] = meta["Manufacturer"]
        elif "manufacturer_model" in meta.columns:
            meta["ManufacturerModelName"] = meta["manufacturer_model"]
        else:
            meta["ManufacturerModelName"] = "Unknown"

    aec_cols = find_aec_cols(meta)
    if len(aec_cols) == 0:
        aec = pd.read_excel(path, sheet_name="aec_128")
        aec_cols = find_aec_cols(aec)
        if len(aec_cols) != 128:
            raise ValueError(f"Expected 128 AEC columns in {path}; found {len(aec_cols)}")
        df = meta.merge(aec[["PatientID"] + aec_cols], on="PatientID", how="inner")
    else:
        df = meta.copy()

    required = ["PatientID", "PatientAge", "PatientSex", "Height", "Weight", "TAMA", "ManufacturerModelName"] + aec_cols
    for col in ["PatientID", "PatientAge", "Height", "Weight", "TAMA"] + aec_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy().reset_index(drop=True)
    df["ManufacturerModelName"] = df["ManufacturerModelName"].astype(str).str.strip()
    df = add_outcome(df)
    df = attach_phase(df, path, cohort_key)
    df["Cohort"] = cohort_key
    return df.reset_index(drop=True), aec_cols


# =============================================================================
# 3. AEC 전처리 및 시그니처 추출
# =============================================================================


def smooth_log(df: pd.DataFrame, aec_cols: Sequence[str]) -> np.ndarray:
    """
    AEC values are right-skewed.
    We use log1p transform and Savitzky-Golay smoothing.
    """
    x = np.clip(df[list(aec_cols)].astype(float).to_numpy(), 0.0, None)
    x = np.log1p(x)
    return savgol_filter(x, window_length=9, polyorder=2, axis=1, mode="interp")


def patientwise_robust_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Patient-wise robust z-normalization.

    This emphasizes within-patient curve shape rather than absolute scanner
    dose level. It was part of the final contrastive signature.
    """
    med = np.nanmedian(x, axis=1, keepdims=True)
    q25 = np.nanquantile(x, 0.25, axis=1, keepdims=True)
    q75 = np.nanquantile(x, 0.75, axis=1, keepdims=True)
    return (x - med) / ((q75 - q25) + eps)


def segment_features(x: np.ndarray, bins: int) -> np.ndarray:
    """Multi-resolution summary of curve segments."""
    edges = np.linspace(0, x.shape[1], bins + 1).astype(int)
    vals = []
    for i in range(bins):
        seg = x[:, edges[i] : edges[i + 1]]
        d1 = np.gradient(seg, axis=1) if seg.shape[1] > 1 else np.zeros_like(seg)
        vals.extend(
            [
                seg.mean(axis=1),
                seg.std(axis=1),
                seg.min(axis=1),
                seg.max(axis=1),
                np.ptp(seg, axis=1),
                seg[:, -1] - seg[:, 0],
                np.abs(d1).mean(axis=1),
                np.abs(d1).max(axis=1),
            ]
        )
    return np.column_stack(vals)


def curve_summary(x: np.ndarray) -> np.ndarray:
    """Global summary values for a curve matrix."""
    d1 = np.gradient(x, axis=1)
    d2 = np.gradient(d1, axis=1)
    return np.column_stack(
        [
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            np.ptp(x, axis=1),
            np.percentile(x, 10, axis=1),
            np.percentile(x, 25, axis=1),
            np.percentile(x, 50, axis=1),
            np.percentile(x, 75, axis=1),
            np.percentile(x, 90, axis=1),
            np.trapezoid(x, axis=1),
            x[:, -1] - x[:, 0],
            np.abs(d1).mean(axis=1),
            np.abs(d1).max(axis=1),
            np.abs(d2).mean(axis=1),
        ]
    )


def inverse_features_from_x(x: np.ndarray) -> np.ndarray:
    """Whole-curve and low-order morphology descriptors."""
    n = x.shape[1]
    thirds = np.array_split(np.arange(n), 3)
    quarters = np.array_split(np.arange(n), 4)
    mid = np.arange(max(0, n // 2 - 8), min(n, n // 2 + 8))
    d1 = np.gradient(x, axis=1)
    d2 = np.gradient(d1, axis=1)

    pos = np.arange(n)
    w = np.maximum(x - x.min(axis=1, keepdims=True), 0.0) + 1e-6
    center_of_mass = (w * pos).sum(axis=1) / w.sum(axis=1)

    return np.column_stack(
        [
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            np.ptp(x, axis=1),
            x[:, -1] - x[:, 0],
            x[:, thirds[0]].mean(axis=1),
            x[:, thirds[1]].mean(axis=1),
            x[:, thirds[2]].mean(axis=1),
            x[:, quarters[0]].mean(axis=1),
            x[:, quarters[1]].mean(axis=1),
            x[:, quarters[2]].mean(axis=1),
            x[:, quarters[3]].mean(axis=1),
            x[:, mid].mean(axis=1),
            x[:, mid].min(axis=1),
            x[:, mid].std(axis=1),
            x[:, thirds[1]].mean(axis=1) - 0.5 * (x[:, thirds[0]].mean(axis=1) + x[:, thirds[2]].mean(axis=1)),
            x[:, thirds[2]].mean(axis=1) - x[:, thirds[0]].mean(axis=1),
            center_of_mass,
            np.abs(d1).mean(axis=1),
            np.abs(d1).max(axis=1),
            np.abs(d2).mean(axis=1),
        ]
    )


def dct_features_from_x(x: np.ndarray, n_coeff: int = 16) -> np.ndarray:
    """Low-frequency DCT coefficients: broad whole-curve shape."""
    coeff = dct(x, axis=1, norm="ortho")[:, :n_coeff]
    d1 = np.gradient(x, axis=1)
    d1c = dct(d1, axis=1, norm="ortho")[:, : min(8, n_coeff)]
    return np.column_stack([coeff, d1c])


def morph_from_curve(x: np.ndarray) -> np.ndarray:
    """AEC morphology features used to define the contrastive distance space."""
    return np.column_stack(
        [
            segment_features(x, 2),
            segment_features(x, 4),
            segment_features(x, 8),
            segment_features(x, 16),
            inverse_features_from_x(x),
            dct_features_from_x(x, 16),
            curve_summary(x),
        ]
    )


ZONE_NAMES = ["upper_1", "upper_2", "mid_1", "mid_2", "lower_1", "pelvis"]
ZONE_EDGES = np.array([0, 20, 42, 64, 86, 108, 128])


def zone_features_one(x: np.ndarray) -> np.ndarray:
    """Relative anatomy-zone descriptors across the 128-point AEC curve."""
    vals = []
    means = []
    areas = []
    for lo, hi in zip(ZONE_EDGES[:-1], ZONE_EDGES[1:]):
        seg = x[:, lo:hi]
        d1 = np.gradient(seg, axis=1) if seg.shape[1] > 1 else np.zeros_like(seg)
        m = np.nanmean(seg, axis=1)
        area = np.trapezoid(seg, axis=1)
        means.append(m)
        areas.append(area)
        vals.extend(
            [
                m,
                np.nanstd(seg, axis=1),
                np.nanmin(seg, axis=1),
                np.nanmax(seg, axis=1),
                np.nanmax(seg, axis=1) - np.nanmin(seg, axis=1),
                area,
                seg[:, -1] - seg[:, 0],
                np.nanmean(np.abs(d1), axis=1),
            ]
        )

    means = np.column_stack(means)
    areas = np.column_stack(areas)
    eps = 1e-6
    vals.extend(
        [
            means[:, 0] - means[:, -1],
            means[:, 1] - means[:, 4],
            means[:, 2] - means[:, 3],
            (means[:, 0] - means[:, -1]) / (np.abs(means[:, 0]) + np.abs(means[:, -1]) + eps),
            areas[:, :3].sum(axis=1) - areas[:, 3:].sum(axis=1),
            areas[:, 2:4].sum(axis=1) / (np.abs(areas).sum(axis=1) + eps),
            np.argmax(means, axis=1).astype(float),
            np.argmin(means, axis=1).astype(float),
        ]
    )
    return np.column_stack(vals)


def anatomy_zone_features(raw: np.ndarray, pz: np.ndarray, res_raw: np.ndarray, res_pz: np.ndarray) -> np.ndarray:
    """
    Meaning-aware anatomy-zone features.

    These are not used directly as the final model input here. They help define
    the AEC distance space used for clinical-invariant contrastive matching.
    """
    return np.column_stack(
        [
            zone_features_one(raw),
            zone_features_one(pz),
            zone_features_one(res_raw),
            zone_features_one(res_pz),
        ]
    )


def expected_aec_design(train: pd.DataFrame, apply: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design matrix used to estimate expected AEC.

    Meaning:
    AEC absolute level is strongly influenced by body size, scanner, and phase.
    We estimate expected AEC from those variables and use residual shape as part
    of the contrastive signature.
    """

    def base(d: pd.DataFrame) -> pd.DataFrame:
        h_m = d["Height"].astype(float) / 100.0
        bmi = d["Weight"].astype(float) / np.maximum(h_m * h_m, 1e-6)
        out = pd.DataFrame(
            {
                "age": d["PatientAge"].astype(float),
                "male": (d["PatientSex"].astype(str).str.upper() == "M").astype(float),
                "height": d["Height"].astype(float),
                "weight": d["Weight"].astype(float),
                "bmi": bmi,
                "phase": d["PhaseGroup"].fillna("indeterminate").astype(str),
                "scanner": d["ManufacturerModelName"].fillna("Unknown").astype(str),
            },
            index=d.index,
        )
        return out

    tr = base(train)
    ap = base(apply)
    combined = pd.concat([tr, ap], axis=0)
    encoded = pd.get_dummies(combined, columns=["phase", "scanner"], dummy_na=False)
    x_tr = encoded.iloc[: len(tr)].astype(float).to_numpy()
    x_ap = encoded.iloc[len(tr) :].astype(float).to_numpy()
    return x_tr, x_ap


def fit_expected_residuals(
    train: pd.DataFrame,
    apply: pd.DataFrame,
    train_aec_cols: List[str],
    apply_aec_cols: List[str],
) -> Dict[str, np.ndarray]:
    """Fit expected-AEC model on train only, then apply to train/apply."""
    raw_tr = smooth_log(train, train_aec_cols)
    raw_ap = smooth_log(apply, apply_aec_cols)
    pz_tr = patientwise_robust_z(raw_tr)
    pz_ap = patientwise_robust_z(raw_ap)
    x_tr, x_ap = expected_aec_design(train, apply)

    # Multi-output Ridge predicts the 128-point curve at once.
    raw_model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("ridge", Ridge(alpha=8.0))])
    pz_model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("ridge", Ridge(alpha=12.0))])
    raw_model.fit(x_tr, raw_tr)
    pz_model.fit(x_tr, pz_tr)

    return {
        "raw_train": raw_tr,
        "raw_apply": raw_ap,
        "pz_train": pz_tr,
        "pz_apply": pz_ap,
        "res_raw_train": raw_tr - raw_model.predict(x_tr),
        "res_raw_apply": raw_ap - raw_model.predict(x_ap),
        "res_pz_train": pz_tr - pz_model.predict(x_tr),
        "res_pz_apply": pz_ap - pz_model.predict(x_ap),
    }


# =============================================================================
# 4. 임상 모델 · 대조적 AEC 점수
# =============================================================================


@dataclass
class ClinicalPredictions:
    score: np.ndarray
    threshold: np.ndarray
    pred: np.ndarray


def clinical_matrix(df: pd.DataFrame) -> np.ndarray:
    """Clinical predictors only: age, height, weight, male sex."""
    return np.column_stack(
        [
            df[["PatientAge", "Height", "Weight"]].astype(float).to_numpy(),
            (df["PatientSex"].astype(str).str.upper() == "M").astype(float).to_numpy().reshape(-1, 1),
        ]
    )


def make_clinical_model() -> Pipeline:
    """Logistic regression pipeline for clinical-only prediction."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", C=CLINICAL_LOGIT_C)),
        ]
    )


def clinical_oof(df: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]]) -> ClinicalPredictions:
    """Out-of-fold clinical predictions for internal validation."""
    y = df["LowSMI_Derstine"].astype(int).to_numpy()
    x = clinical_matrix(df)
    score = np.zeros(len(df), dtype=float)
    threshold = np.zeros(len(df), dtype=float)
    for train_idx, test_idx in folds:
        model = make_clinical_model()
        model.fit(x[train_idx], y[train_idx])
        train_score = model.predict_proba(x[train_idx])[:, 1]
        score[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        threshold[test_idx] = youden_threshold(y[train_idx], train_score)
    return ClinicalPredictions(score=score, threshold=threshold, pred=score >= threshold)


def clinical_full_train_external(dev: pd.DataFrame, ext: pd.DataFrame) -> Tuple[ClinicalPredictions, ClinicalPredictions]:
    """Clinical model fitted on all Gangnam, then applied to Shinchon."""
    y = dev["LowSMI_Derstine"].astype(int).to_numpy()
    model = make_clinical_model()
    model.fit(clinical_matrix(dev), y)
    dev_score = model.predict_proba(clinical_matrix(dev))[:, 1]
    ext_score = model.predict_proba(clinical_matrix(ext))[:, 1]
    thr = youden_threshold(y, dev_score)
    return (
        ClinicalPredictions(dev_score, np.full(len(dev), thr), dev_score >= thr),
        ClinicalPredictions(ext_score, np.full(len(ext), thr), ext_score >= thr),
    )


def clinical_match_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Variables used to define clinically similar neighbors.

    Sex is multiplied by 8 to make opposite-sex patients less likely to be
    nearest neighbors, while still allowing the model to work if one class is
    sparse in a fold.
    """
    h_m = df["Height"].astype(float).to_numpy() / 100.0
    bmi = df["Weight"].astype(float).to_numpy() / np.maximum(h_m * h_m, 1e-6)
    return np.column_stack(
        [
            df["PatientAge"].astype(float).to_numpy(),
            (df["PatientSex"].astype(str).str.upper() == "M").astype(float).to_numpy() * 8.0,
            df["Height"].astype(float).to_numpy(),
            df["Weight"].astype(float).to_numpy(),
            bmi,
        ]
    )


def standardize_by_train(train: np.ndarray, apply: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score normalization fitted on train, then applied to both train and apply."""
    mu = np.nanmean(train, axis=0, keepdims=True)
    sd = np.nanstd(train, axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (train - mu) / sd, (apply - mu) / sd


def contrastive_neighbor_features(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    train_y: np.ndarray,
    train_signature: np.ndarray,
    apply_signature: np.ndarray,
    apply_is_train: bool,
    k_per_class: int = K_NEIGHBORS_PER_CLASS,
) -> np.ndarray:
    """
    Compute clinical-invariant contrastive AEC features.

    For each apply patient:
    1. Find clinically similar low-SMI neighbors in the training set.
    2. Find clinically similar normal neighbors in the training set.
    3. Compare AEC signature distance to the two neighbor groups.

    Main score:
      mean distance to normal neighbors - mean distance to low-SMI neighbors

    If positive, the patient's AEC is relatively closer to low-SMI neighbors.
    If negative, the patient's AEC is relatively closer to normal neighbors.
    """
    x_clin_tr = clinical_match_matrix(train_df)
    x_clin_ap = clinical_match_matrix(apply_df)
    x_clin_tr, x_clin_ap = standardize_by_train(x_clin_tr, x_clin_ap)
    sig_tr, sig_ap = standardize_by_train(train_signature, apply_signature)

    y = np.asarray(train_y).astype(int)
    case_idx_all = np.where(y == 1)[0]
    ctrl_idx_all = np.where(y == 0)[0]
    rows = []
    eps = 1e-6

    for i in range(len(apply_df)):
        clin_dist = np.sqrt(np.sum((x_clin_tr - x_clin_ap[i]) ** 2, axis=1))
        if apply_is_train:
            # When scoring the training set itself, exclude the same patient.
            clin_dist[i] = np.inf

        case_idx = case_idx_all[np.argsort(clin_dist[case_idx_all])[:k_per_class]]
        ctrl_idx = ctrl_idx_all[np.argsort(clin_dist[ctrl_idx_all])[:k_per_class]]
        if len(case_idx) == 0 or len(ctrl_idx) == 0:
            rows.append([0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0])
            continue

        dist_to_cases = np.sqrt(np.sum((sig_tr[case_idx] - sig_ap[i]) ** 2, axis=1))
        dist_to_ctrls = np.sqrt(np.sum((sig_tr[ctrl_idx] - sig_ap[i]) ** 2, axis=1))

        # Closer clinical neighbors get more weight.
        case_w = np.exp(-clin_dist[case_idx])
        ctrl_w = np.exp(-clin_dist[ctrl_idx])
        mean_case = float(np.average(dist_to_cases, weights=case_w))
        mean_ctrl = float(np.average(dist_to_ctrls, weights=ctrl_w))
        nearest_case = float(np.min(dist_to_cases))
        nearest_ctrl = float(np.min(dist_to_ctrls))

        rows.append(
            [
                mean_ctrl - mean_case,
                nearest_ctrl - nearest_case,
                mean_case,
                mean_ctrl,
                nearest_case,
                nearest_ctrl,
                float(case_w.sum() / (case_w.sum() + ctrl_w.sum() + eps)),
                float(np.average(clin_dist[case_idx], weights=case_w) - np.average(clin_dist[ctrl_idx], weights=ctrl_w)),
            ]
        )
    return np.asarray(rows, dtype=float)


def build_contrastive_blocks(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    train_y: np.ndarray,
    train_aec_cols: List[str],
    apply_aec_cols: List[str],
    apply_is_train: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return final contrastive feature block for train and apply patients.

    The final correction model uses only these 8 contrastive values, plus the
    clinical score/margin/covariates added later in correction_features().
    """
    curves = fit_expected_residuals(train_df, apply_df, train_aec_cols, apply_aec_cols)
    raw_tr = curves["raw_train"]
    raw_ap = curves["raw_apply"]
    pz_tr = curves["pz_train"]
    pz_ap = curves["pz_apply"]
    rr_tr = curves["res_raw_train"]
    rr_ap = curves["res_raw_apply"]
    rp_tr = curves["res_pz_train"]
    rp_ap = curves["res_pz_apply"]

    zones_tr = anatomy_zone_features(raw_tr, pz_tr, rr_tr, rp_tr)
    zones_ap = anatomy_zone_features(raw_ap, pz_ap, rr_ap, rp_ap)

    # The distance space is deliberately richer than the 8 final contrastive values.
    # It contains AEC shape, expected-AEC residual shape, and anatomy-zone summaries.
    sig_tr = np.column_stack([morph_from_curve(pz_tr), morph_from_curve(rp_tr), zones_tr])
    sig_ap = np.column_stack([morph_from_curve(pz_ap), morph_from_curve(rp_ap), zones_ap])

    contrast_train = contrastive_neighbor_features(
        train_df, train_df, train_y, sig_tr, sig_tr, apply_is_train=True
    )
    contrast_apply = contrastive_neighbor_features(
        train_df, apply_df, train_y, sig_tr, sig_ap, apply_is_train=apply_is_train
    )
    return contrast_train, contrast_apply


# =============================================================================
# 5. 선택적 보정 모델
# =============================================================================


def correction_features(block: np.ndarray, df: pd.DataFrame, clinical: ClinicalPredictions) -> np.ndarray:
    """
    Features for the rule-out/rule-in correction heads.

    We include clinical score and margin because AEC is intended to correct
    clinical errors near the decision boundary, not replace clinical prediction.
    """
    margin = clinical.score - clinical.threshold
    return np.column_stack(
        [
            clinical.score.reshape(-1, 1),
            clinical.threshold.reshape(-1, 1),
            margin.reshape(-1, 1),
            np.abs(margin).reshape(-1, 1),
            clinical.pred.astype(float).reshape(-1, 1),
            clinical_matrix(df),
            block,
        ]
    )


def make_correction_model(n_features: int) -> Pipeline:
    """Sparse ElasticNet logistic regression correction head."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(FEATURE_SELECT_K, n_features))),
            (
                "est",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=CORRECTION_L1_RATIO,
                    C=CORRECTION_LOGIT_C,
                    max_iter=8000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def fit_or_dummy(model: Pipeline, x: np.ndarray, y: np.ndarray):
    """Handle rare folds where a correction target has one class only."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        dummy = DummyClassifier(strategy="constant", constant=int(y[0]) if len(y) else 0)
        dummy.fit(np.zeros((max(1, len(y)), 1)), np.repeat(int(y[0]) if len(y) else 0, max(1, len(y))))
        return dummy, True
    fitted = clone(model)
    fitted.fit(x, y)
    return fitted, False


def predict_positive_probability(model, is_dummy: bool, x: np.ndarray) -> np.ndarray:
    if is_dummy:
        proba = model.predict_proba(np.zeros((len(x), 1)))
    else:
        proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        cls = int(model.classes_[0])
        return np.ones(len(x)) if cls == 1 else np.zeros(len(x))
    idx = int(np.where(model.classes_ == 1)[0][0])
    return proba[:, idx]


def inner_clinical_predictions_for_training(df: pd.DataFrame, train_idx: np.ndarray) -> ClinicalPredictions:
    """
    Inner OOF clinical predictions within each outer training fold.

    This avoids training correction labels using in-sample clinical predictions.
    """
    y = df["LowSMI_Derstine"].astype(int).to_numpy()
    x = clinical_matrix(df)
    inner = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 101).split(train_idx, y[train_idx]))
    score = np.zeros(len(train_idx), dtype=float)
    threshold = np.zeros(len(train_idx), dtype=float)
    for tr_rel, va_rel in inner:
        tr = train_idx[tr_rel]
        va = train_idx[va_rel]
        model = make_clinical_model()
        model.fit(x[tr], y[tr])
        train_score = model.predict_proba(x[tr])[:, 1]
        score[va_rel] = model.predict_proba(x[va])[:, 1]
        threshold[va_rel] = youden_threshold(y[tr], train_score)
    return ClinicalPredictions(score=score, threshold=threshold, pred=score >= threshold)


def train_correction_pair(
    x_train: np.ndarray,
    y_train: np.ndarray,
    clinical_pred_train: np.ndarray,
) -> Tuple[object, bool, object, bool]:
    """
    Train two correction heads.

    Rule-out head:
      Among clinical positives, predict whether the case is actually normal.

    Rule-in head:
      Among clinical negatives, predict whether the case is actually low SMI.
    """
    base_pos = clinical_pred_train.astype(bool)
    base_neg = ~base_pos
    ruleout_y = (y_train[base_pos] == 0).astype(int)
    rulein_y = (y_train[base_neg] == 1).astype(int)
    base_model = make_correction_model(x_train.shape[1])
    ruleout_model, ruleout_dummy = fit_or_dummy(base_model, x_train[base_pos], ruleout_y)
    rulein_model, rulein_dummy = fit_or_dummy(base_model, x_train[base_neg], rulein_y)
    return ruleout_model, ruleout_dummy, rulein_model, rulein_dummy


def selective_scores_oof(
    dev: pd.DataFrame,
    dev_aec_cols: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[ClinicalPredictions, np.ndarray, np.ndarray, np.ndarray]:
    """Gangnam OOF clinical, rule-out, rule-in, and contrastive scores."""
    y = dev["LowSMI_Derstine"].astype(int).to_numpy()
    x_clin = clinical_matrix(dev)
    clin = clinical_oof(dev, folds)
    ruleout_score = np.zeros(len(dev), dtype=float)
    rulein_score = np.zeros(len(dev), dtype=float)
    contrast_block = np.zeros((len(dev), 8), dtype=float)

    for train_idx, test_idx in folds:
        # Clinical predictions for the held-out outer fold.
        outer_clinical = make_clinical_model()
        outer_clinical.fit(x_clin[train_idx], y[train_idx])
        train_score_outer = outer_clinical.predict_proba(x_clin[train_idx])[:, 1]
        test_score = outer_clinical.predict_proba(x_clin[test_idx])[:, 1]
        test_threshold = youden_threshold(y[train_idx], train_score_outer)
        test_clin = ClinicalPredictions(
            score=test_score,
            threshold=np.full(len(test_idx), test_threshold),
            pred=test_score >= test_threshold,
        )

        # Clinical OOF predictions inside the training fold define correction targets.
        inner_clin = inner_clinical_predictions_for_training(dev, train_idx)

        tr_df = dev.iloc[train_idx].copy()
        te_df = dev.iloc[test_idx].copy()
        contrast_train, contrast_test = build_contrastive_blocks(
            tr_df,
            te_df,
            y[train_idx],
            dev_aec_cols,
            dev_aec_cols,
            apply_is_train=False,
        )
        contrast_block[test_idx] = contrast_test

        x_train = correction_features(contrast_train, tr_df, inner_clin)
        x_test = correction_features(contrast_test, te_df, test_clin)

        ro_model, ro_dummy, ri_model, ri_dummy = train_correction_pair(x_train, y[train_idx], inner_clin.pred)
        ruleout_score[test_idx] = predict_positive_probability(ro_model, ro_dummy, x_test)
        rulein_score[test_idx] = predict_positive_probability(ri_model, ri_dummy, x_test)

    return clin, ruleout_score, rulein_score, contrast_block


def selective_scores_external(
    dev: pd.DataFrame,
    ext: pd.DataFrame,
    dev_aec_cols: List[str],
    ext_aec_cols: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    full_ext_clin: ClinicalPredictions,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train correction heads on all Gangnam and apply locked model to Shinchon."""
    y_dev = dev["LowSMI_Derstine"].astype(int).to_numpy()
    train_clin_oof = clinical_oof(dev, folds)
    contrast_train, contrast_ext = build_contrastive_blocks(
        dev,
        ext,
        y_dev,
        dev_aec_cols,
        ext_aec_cols,
        apply_is_train=False,
    )
    x_train = correction_features(contrast_train, dev, train_clin_oof)
    x_ext = correction_features(contrast_ext, ext, full_ext_clin)
    ro_model, ro_dummy, ri_model, ri_dummy = train_correction_pair(x_train, y_dev, train_clin_oof.pred)
    return (
        predict_positive_probability(ro_model, ro_dummy, x_ext),
        predict_positive_probability(ri_model, ri_dummy, x_ext),
        contrast_ext,
    )


def apply_selective(
    clinical: ClinicalPredictions,
    ruleout_score: np.ndarray,
    rulein_score: np.ndarray,
    ruleout_threshold: float,
    rulein_threshold: float,
    max_abs_margin: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the final selective correction rule."""
    base = clinical.pred.astype(bool)
    eligible = np.ones(len(base), dtype=bool)
    if max_abs_margin is not None:
        eligible = np.abs(clinical.score - clinical.threshold) <= float(max_abs_margin)
    ruleout = eligible & base & (ruleout_score >= ruleout_threshold)
    rulein = eligible & (~base) & (rulein_score >= rulein_threshold)
    pred = base.copy()
    pred[ruleout] = False
    pred[rulein] = True
    return pred, ruleout, rulein


def eval_selective(
    df: pd.DataFrame,
    clinical: ClinicalPredictions,
    ruleout_score: np.ndarray,
    rulein_score: np.ndarray,
    ruleout_threshold: float,
    rulein_threshold: float,
    max_abs_margin: Optional[float],
) -> Dict[str, float]:
    """Evaluate clinical baseline versus AEC-assisted workflow."""
    y = df["LowSMI_Derstine"].astype(int).to_numpy()
    workflow_pred, ruleout, rulein = apply_selective(
        clinical, ruleout_score, rulein_score, ruleout_threshold, rulein_threshold, max_abs_margin
    )
    cm = confusion(y, clinical.pred)
    wm = confusion(y, workflow_pred)
    ps = paired_stats(y, clinical.pred, workflow_pred)
    margin = np.abs(clinical.score - clinical.threshold)
    eligible = np.ones(len(y), dtype=bool) if max_abs_margin is None else margin <= max_abs_margin

    return {
        "N": int(len(y)),
        "Events": int(np.sum(y)),
        "MarginGate": "none" if max_abs_margin is None else float(max_abs_margin), # type: ignore
        "EligibleN": int(np.sum(eligible)),
        "RuleOutThreshold": float(ruleout_threshold),
        "RuleInThreshold": float(rulein_threshold),
        "RuleOutN": int(np.sum(ruleout)),
        "RuleInN": int(np.sum(rulein)),
        "Clinical_AUC": safe_auc(y, clinical.score),
        "RuleOut_AUC_in_clinical_positive": safe_auc((y[clinical.pred] == 0).astype(int), ruleout_score[clinical.pred])
        if np.any(clinical.pred)
        else np.nan,
        "RuleIn_AUC_in_clinical_negative": safe_auc((y[~clinical.pred] == 1).astype(int), rulein_score[~clinical.pred])
        if np.any(~clinical.pred)
        else np.nan,
        "Clinical_Sensitivity": cm["Sensitivity"],
        "Workflow_Sensitivity": wm["Sensitivity"],
        "SensitivityDiff": wm["Sensitivity"] - cm["Sensitivity"],
        "Clinical_Specificity": cm["Specificity"],
        "Workflow_Specificity": wm["Specificity"],
        "SpecificityDiff": wm["Specificity"] - cm["Specificity"],
        "Clinical_Accuracy": cm["Accuracy"],
        "Workflow_Accuracy": wm["Accuracy"],
        "AccuracyDiff": wm["Accuracy"] - cm["Accuracy"],
        "Clinical_PPV": cm["PPV"],
        "Workflow_PPV": wm["PPV"],
        "Clinical_NPV": cm["NPV"],
        "Workflow_NPV": wm["NPV"],
        "Workflow_TP": wm["TP"],
        "Workflow_FN": wm["FN"],
        "Workflow_TN": wm["TN"],
        "Workflow_FP": wm["FP"],
        **ps,
    }


def threshold_grid(score: np.ndarray, n: int) -> np.ndarray:
    """Quantile-based threshold grid plus 1.01, which means no cases pass."""
    x = np.asarray(score, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([1.01])
    qs = np.quantile(x, np.linspace(0.05, 0.98, n))
    return np.unique(np.round(np.concatenate([[1.01], qs]), 6))


def sweep_thresholds(df: pd.DataFrame, clinical: ClinicalPredictions, ruleout_score: np.ndarray, rulein_score: np.ndarray) -> pd.DataFrame:
    """Search thresholds using Gangnam internal CV only."""
    rows = []
    margin_gates: List[Optional[float]] = [None, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
    ro_grid = threshold_grid(ruleout_score[clinical.pred], 40)
    ri_grid = threshold_grid(rulein_score[~clinical.pred], 35)
    for margin_gate in margin_gates:
        for ro in ro_grid:
            for ri in ri_grid:
                rows.append(eval_selective(df, clinical, ruleout_score, rulein_score, float(ro), float(ri), margin_gate))
    return pd.DataFrame(rows)


def choose_internal_rules(sweep: pd.DataFrame) -> pd.DataFrame:
    """
    Pick standard internal rules.

    Final primary rule for this study is no_sens_drop:
      - specificity improves significantly
      - accuracy improves significantly
      - sensitivity does not drop in internal CV
    """
    rows = []

    def add(name: str, mask: pd.Series, sort_cols: List[str]) -> None:
        cand = sweep[mask].copy()
        if cand.empty:
            return
        row = cand.sort_values(sort_cols, ascending=False).iloc[0].copy()
        row["Rule"] = name
        rows.append(row)

    useful = (
        (sweep["SpecificityDiff"] > 0)
        & (sweep["AccuracyDiff"] > 0)
        & (sweep["Specificity_p"] < 0.05)
        & (sweep["Accuracy_p"] < 0.05)
    )
    sens_p_guard = ~((sweep["SensitivityDiff"] < 0) & (sweep["Sensitivity_p"] < 0.05))
    add("sens_drop_le_0p02", useful & (sweep["SensitivityDiff"] >= -0.02), ["NetCorrected", "AccuracyDiff", "SpecificityDiff", "SensitivityDiff"])
    add("sens_p_guard", useful & sens_p_guard, ["NetCorrected", "AccuracyDiff", "SpecificityDiff", "SensitivityDiff"])
    add("no_sens_drop", useful & (sweep["SensitivityDiff"] >= 0), ["NetCorrected", "AccuracyDiff", "SpecificityDiff"])
    add("max_net", useful, ["NetCorrected", "AccuracyDiff", "SpecificityDiff"])
    return pd.DataFrame(rows)


# =============================================================================
# 6. 결과 테이블 · 환자 레벨 감사 · 시각화
# =============================================================================


def subgroup_rows(
    df: pd.DataFrame,
    clinical: ClinicalPredictions,
    ruleout_score: np.ndarray,
    rulein_score: np.ndarray,
    selected: pd.Series,
    cohort: str,
) -> pd.DataFrame:
    """Overall, sex, phase, and scanner subgroup table."""
    rows = []
    ro = float(selected["RuleOutThreshold"])
    ri = float(selected["RuleInThreshold"])
    margin_raw = selected["MarginGate"]
    margin = None if str(margin_raw) == "none" else float(margin_raw)

    def subset_eval(group_type: str, group: str, mask: np.ndarray) -> None:
        d = df.loc[mask].copy()
        c = ClinicalPredictions(clinical.score[mask], clinical.threshold[mask], clinical.pred[mask])
        row = eval_selective(d, c, ruleout_score[mask], rulein_score[mask], ro, ri, margin)
        row.update({"Cohort": cohort, "GroupType": group_type, "Group": group}) # type: ignore
        rows.append(row)

    all_mask = np.ones(len(df), dtype=bool)
    subset_eval("Overall", "All", all_mask)
    for sex in ["M", "F"]:
        mask = df["PatientSex"].astype(str).str.upper().to_numpy() == sex
        if np.any(mask):
            subset_eval("Sex", sex, mask)
    for phase, label in [("pre_or_noncontrast", "Pre/noncontrast"), ("post_or_contrast", "Post/contrast")]:
        mask = df["PhaseGroup"].to_numpy() == phase
        if np.any(mask):
            subset_eval("Phase", label, mask)
    for scanner in sorted(df["ManufacturerModelName"].dropna().astype(str).unique()):
        mask = df["ManufacturerModelName"].astype(str).to_numpy() == scanner
        if np.any(mask):
            subset_eval("Scanner", scanner, mask)
    return pd.DataFrame(rows)


def smi_cutoff_array(df: pd.DataFrame) -> np.ndarray:
    """Return the sex-specific Derstine L3 SMI cutoff value for each patient."""
    sex = df["PatientSex"].astype(str).str.upper().to_numpy()
    return np.where(sex == "M", SMI_MALE_CUTOFF, SMI_FEMALE_CUTOFF)


def bmi_array(df: pd.DataFrame) -> np.ndarray:
    """Compute BMI (kg/m²) for each patient from Height (cm) and Weight (kg)."""
    h_m = df["Height"].astype(float).to_numpy() / 100.0
    return df["Weight"].astype(float).to_numpy() / np.maximum(h_m * h_m, 1e-6)


def patient_audit_frame(
    df: pd.DataFrame,
    cohort: str,
    clinical: ClinicalPredictions,
    ruleout_score: np.ndarray,
    rulein_score: np.ndarray,
    contrast_block: np.ndarray,
    selected: pd.Series,
) -> pd.DataFrame:
    """Patient-level table showing who was rule-out/rule-in corrected."""
    ro = float(selected["RuleOutThreshold"])
    ri = float(selected["RuleInThreshold"])
    margin_raw = selected["MarginGate"]
    margin_gate = None if str(margin_raw) == "none" else float(margin_raw)
    y = df["LowSMI_Derstine"].astype(int).to_numpy()
    pred, ruleout, rulein = apply_selective(clinical, ruleout_score, rulein_score, ro, ri, margin_gate)
    margin = clinical.score - clinical.threshold

    out = pd.DataFrame(
        {
            "Cohort": cohort,
            "PatientID": df["PatientID"].to_numpy(),
            "ActualLowSMI": y,
            "ActualLabel": np.where(y == 1, "Low SMI", "Normal"),
            "PatientAge": df["PatientAge"].astype(float).to_numpy(),
            "PatientSex": df["PatientSex"].astype(str).str.upper().to_numpy(),
            "Height": df["Height"].astype(float).to_numpy(),
            "Weight": df["Weight"].astype(float).to_numpy(),
            "BMI": bmi_array(df),
            "TAMA": df["TAMA"].astype(float).to_numpy(),
            "SMI_calc": df["SMI_calc"].astype(float).to_numpy(),
            "SMI_cutoff": smi_cutoff_array(df),
            "SMI_margin": df["SMI_calc"].astype(float).to_numpy() - smi_cutoff_array(df),
            "PhaseGroup": df["PhaseGroup"].astype(str).to_numpy(),
            "Scanner": df["ManufacturerModelName"].astype(str).to_numpy(),
            "ClinicalScore": clinical.score,
            "ClinicalThreshold": clinical.threshold,
            "ClinicalMargin": margin,
            "AbsClinicalMargin": np.abs(margin),
            "ClinicalPred": clinical.pred.astype(int),
            "RuleOutScore": ruleout_score,
            "RuleInScore": rulein_score,
            "ContrastLowLikeScore": contrast_block[:, 0],
            "ContrastNearestLowLikeScore": contrast_block[:, 1],
            "MeanDistanceToLowNeighbors": contrast_block[:, 2],
            "MeanDistanceToNormalNeighbors": contrast_block[:, 3],
            "WorkflowPred": pred.astype(int),
            "EligibleGrayZone": (np.ones(len(df), dtype=bool) if margin_gate is None else (np.abs(margin) <= margin_gate)).astype(int),
            "RuleOutApplied": ruleout.astype(int),
            "RuleInApplied": rulein.astype(int),
        }
    )
    status = np.full(len(out), "unchanged", dtype=object)
    status[(y == 0) & (clinical.pred == 1) & (pred == 0)] = "corrected_FP_ruleout"
    status[(y == 1) & (clinical.pred == 1) & (pred == 0)] = "lost_TP_ruleout"
    status[(y == 1) & (clinical.pred == 0) & (pred == 1)] = "rescued_FN_rulein"
    status[(y == 0) & (clinical.pred == 0) & (pred == 1)] = "added_FP_rulein"
    out["WorkflowStatus"] = status
    return out


def workflow_counts(audit: pd.DataFrame) -> pd.DataFrame:
    """Readable n-counts for clinical-positive rule-out and clinical-negative rule-in."""
    rows = []
    for cohort, d in audit.groupby("Cohort"):
        cp = d[d["ClinicalPred"] == 1]
        cn = d[d["ClinicalPred"] == 0]
        ro = d[d["RuleOutApplied"] == 1]
        ri = d[d["RuleInApplied"] == 1]
        rows.append(
            {
                "Cohort": cohort,
                "N": len(d),
                "Events": int(d["ActualLowSMI"].sum()),
                "ClinicalPositiveN": len(cp),
                "ClinicalPositive_TP": int((cp["ActualLowSMI"] == 1).sum()),
                "ClinicalPositive_FP": int((cp["ActualLowSMI"] == 0).sum()),
                "RuleOutN": len(ro),
                "RuleOut_correctedFP": int((ro["WorkflowStatus"] == "corrected_FP_ruleout").sum()),
                "RuleOut_lostTP": int((ro["WorkflowStatus"] == "lost_TP_ruleout").sum()),
                "ClinicalNegativeN": len(cn),
                "ClinicalNegative_TN": int((cn["ActualLowSMI"] == 0).sum()),
                "ClinicalNegative_FN": int((cn["ActualLowSMI"] == 1).sum()),
                "RuleInN": len(ri),
                "RuleIn_rescuedFN": int((ri["WorkflowStatus"] == "rescued_FN_rulein").sum()),
                "RuleIn_addedFP": int((ri["WorkflowStatus"] == "added_FP_rulein").sum()),
                "NetCorrected": int(
                    d["WorkflowStatus"].isin(["corrected_FP_ruleout", "rescued_FN_rulein"]).sum()
                    - d["WorkflowStatus"].isin(["lost_TP_ruleout", "added_FP_rulein"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def visual_shape_normalize(raw: np.ndarray) -> np.ndarray:
    """Stable plot-only shape normalization."""
    med = np.nanmedian(raw, axis=1, keepdims=True)
    p10 = np.nanquantile(raw, 0.10, axis=1, keepdims=True)
    p90 = np.nanquantile(raw, 0.90, axis=1, keepdims=True)
    denom = np.maximum(p90 - p10, 0.25)
    return np.clip((raw - med) / denom, -6.0, 6.0)


def median_iqr(x: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (median, Q25, Q75) along axis=0 for rows selected by mask."""
    xx = x[mask]
    return np.nanmedian(xx, axis=0), np.nanquantile(xx, 0.25, axis=0), np.nanquantile(xx, 0.75, axis=0)


def add_zone_lines(ax) -> None:
    """Draw light dashed vertical lines at anatomy-zone boundaries on an axes."""
    for edge in ZONE_EDGES[1:-1]:
        ax.axvline(edge + 1, color="#cccccc", lw=0.8, ls="--", zorder=0)


def plot_signature(dev: pd.DataFrame, ext: pd.DataFrame, dev_aec_cols: List[str], ext_aec_cols: List[str]) -> None:
    """Low SMI vs normal AEC signature plot for interpretation."""
    raw_dev = smooth_log(dev, dev_aec_cols)
    raw_ext = smooth_log(ext, ext_aec_cols)
    shape_dev = visual_shape_normalize(raw_dev)
    shape_ext = visual_shape_normalize(raw_ext)
    panels = [
        ("Gangnam smoothed log-AEC", dev, raw_dev, "Smoothed log-AEC"),
        ("Shinchon smoothed log-AEC", ext, raw_ext, "Smoothed log-AEC"),
        ("Gangnam shape-normalized AEC", dev, shape_dev, "Shape value"),
        ("Shinchon shape-normalized AEC", ext, shape_ext, "Shape value"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), sharex=True)
    axes = axes.reshape(-1)
    x = np.arange(1, 129)
    colors = {"Low SMI": "#b2182b", "Normal": "#2166ac"}
    for ax, (title, df, mat, ylabel) in zip(axes, panels):
        y = df["LowSMI_Derstine"].astype(int).to_numpy()
        for val, label in [(1, "Low SMI"), (0, "Normal")]:
            med, q25, q75 = median_iqr(mat, y == val)
            ax.plot(x, med, color=colors[label], lw=2, label=f"{label} (n={(y == val).sum()})")
            ax.fill_between(x, q25, q75, color=colors[label], alpha=0.12, linewidth=0)
        add_zone_lines(ax)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
    for ax in axes[-2:]:
        ax.set_xlabel("Relative z-position: 1 inferior pubic -> 128 liver dome margin")
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aec_low_vs_normal_signature_clean.png", dpi=220)
    plt.close(fig)


def plot_roc_curves(
    dev: pd.DataFrame,
    ext: pd.DataFrame,
    clin_dev: ClinicalPredictions,
    clin_ext: ClinicalPredictions,
    ro_dev: np.ndarray,
    ri_dev: np.ndarray,
    ro_ext: np.ndarray,
    ri_ext: np.ndarray,
    contrast_dev: np.ndarray,
    contrast_ext: np.ndarray,
    final_rule: pd.Series,
) -> None:
    """
    ROC 곡선 시각화 (강남 OOF · 신촌 외부 각각 1개 패널).

    각 패널:
    - 임상 모델 ROC 곡선 및 AUC
    - 대조적 AEC 점수(contrast_block[:, 0]) ROC 곡선 및 AUC
    - 임상 모델 단독 동작점 (○)
    - AEC 보정 워크플로 동작점 (★)
    """
    ro_thr = float(final_rule["RuleOutThreshold"])
    ri_thr = float(final_rule["RuleInThreshold"])
    margin_raw = final_rule["MarginGate"]
    margin_gate = None if str(margin_raw) == "none" else float(margin_raw)

    datasets = [
        ("강남 (내부 OOF)", dev, clin_dev, contrast_dev, ro_dev, ri_dev),
        ("신촌 (외부)", ext, clin_ext, contrast_ext, ro_ext, ri_ext),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, (title, df, clin, contrast, ro_score, ri_score) in zip(axes, datasets):
        y = df["LowSMI_Derstine"].astype(int).to_numpy()

        # 임상 모델 ROC
        fpr_c, tpr_c, _ = roc_curve(y, clin.score)
        auc_c = safe_auc(y, clin.score)
        ax.plot(fpr_c, tpr_c, color="#2166ac", lw=2,
                label=f"임상 모델 (AUC = {auc_c:.3f})")

        # 대조적 AEC 점수 ROC (contrast_block[:, 0] = 정상 이웃 거리 - 저SMI 이웃 거리)
        aec_score = contrast[:, 0]
        auc_aec = safe_auc(y, aec_score)
        if np.isfinite(auc_aec):
            fpr_a, tpr_a, _ = roc_curve(y, aec_score)
            ax.plot(fpr_a, tpr_a, color="#d6604d", lw=2, ls="--",
                    label=f"대조적 AEC 점수 (AUC = {auc_aec:.3f})")

        # 임상 단독 동작점
        cm = confusion(y, clin.pred)
        ax.scatter(
            [1 - cm["Specificity"]], [cm["Sensitivity"]],
            marker="o", s=90, zorder=5, color="#2166ac",
            label=f"임상 동작점  (민감도 {cm['Sensitivity']:.3f} / 특이도 {cm['Specificity']:.3f})",
        )

        # AEC 보정 워크플로 동작점
        wf_pred, _, _ = apply_selective(clin, ro_score, ri_score, ro_thr, ri_thr, margin_gate)
        wm = confusion(y, wf_pred)
        ax.scatter(
            [1 - wm["Specificity"]], [wm["Sensitivity"]],
            marker="*", s=220, zorder=5, color="#1a9641",
            label=f"AEC 워크플로 (민감도 {wm['Sensitivity']:.3f} / 특이도 {wm['Specificity']:.3f})",
        )

        ax.plot([0, 1], [0, 1], color="#cccccc", lw=1, ls="--")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("1 − 특이도 (FPR)")
        ax.set_ylabel("민감도 (TPR)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        ax.grid(alpha=0.2)

    fig.suptitle("ROC 곡선: 임상 모델 vs AEC 보정 워크플로", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_results_table(overall: pd.DataFrame) -> None:
    """
    PNG 결과 테이블: AEC 보정 vs 임상 단독 성능 비교.

    레이아웃:
      - 어두운 헤더 (두 줄: Clinical only / AEC-assisted)
      - 코호트 블록마다 3개 지표 행 (민감도 / 특이도 / 정확도)
      - 코호트 이름 아래 임상 AUC 표시
      - AEC 열: 수치 (±/+/−차이) 유의성 마커, 색상 구분
      - Net NRI 배지
    """
    COHORT_LABELS = {
        "Gangnam_internal_CV": "Gangnam",
        "Shinchon_sdata_external": "Shinchon",
    }
    METRICS = [
        ("Sensitivity", "Clinical_Sensitivity", "Workflow_Sensitivity", "SensitivityDiff", "Sensitivity_p"),
        ("Specificity", "Clinical_Specificity", "Workflow_Specificity", "SpecificityDiff", "Specificity_p"),
        ("Accuracy",    "Clinical_Accuracy",    "Workflow_Accuracy",    "AccuracyDiff",    "Accuracy_p"),
    ]

    # ── y layout (1 = top, 0 = bottom) ───────────────────────────────────────
    TITLE_Y1,  TITLE_Y0  = 1.00, 0.92
    HEADER_Y1, HEADER_Y0 = 0.92, 0.78
    DATA_Y1,   DATA_Y0   = 0.78, 0.06
    FOOTER_Y1, FOOTER_Y0 = 0.06, 0.00

    n_rows = len(overall) * len(METRICS)
    ROW_H  = (DATA_Y1 - DATA_Y0) / n_rows

    # ── columns ───────────────────────────────────────────────────────────────
    # 코호트 | N | Event | 지표 | Clinical only | AEC-assisted | Net NRI
    CW = [0.14, 0.07, 0.07, 0.12, 0.20, 0.29, 0.11]
    XL = [sum(CW[:j]) for j in range(len(CW))]

    # ── palette ───────────────────────────────────────────────────────────────
    C_DARK    = "#2b2b2b"
    C_WHITE   = "#ffffff"
    C_BODY    = "#2a2a2a"
    C_GRAY    = "#888888"
    C_GREEN   = "#1fa86c"
    C_RED     = "#e03a3a"
    C_BADGE   = "#ddeeff"
    C_BADGE_T = "#2255aa"
    C_SEP_L   = "#e0e0e0"

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    def rect(x, y1, w, h, fc, ec=None, lw=0.0, z=1):
        ax.add_patch(Rectangle((x, y1 - h), w, h, facecolor=fc,
                               edgecolor=ec or fc, linewidth=lw,
                               clip_on=False, zorder=z))

    def txt(x, y, s, ha="center", va="center", color=C_BODY, fs=10, fw="normal"):
        ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=fw,
                color=color, clip_on=False, zorder=5)

    # ── Title ─────────────────────────────────────────────────────────────────
    rect(0, TITLE_Y1, 1.0, TITLE_Y1 - TITLE_Y0, C_WHITE)
    txt(0.01, (TITLE_Y1 + TITLE_Y0) / 2,
        "AEC-assisted vs. clinical-only 성능 비교",
        ha="left", fs=13, fw="bold", color="#444444")

    # ── Header ────────────────────────────────────────────────────────────────
    HDR_H = HEADER_Y1 - HEADER_Y0
    rect(0, HEADER_Y1, 1.0, HDR_H, C_DARK)
    hdr_labels = ["코호트", "N",  "Event", "지표", "Clinical only",       "AEC-assisted",          "Net\nNRI"]
    hdr_subs   = ["",       "",   "",      "",     "sens / spec / acc", "sens / spec / acc (p)", ""]
    for j, (lb, sb) in enumerate(zip(hdr_labels, hdr_subs)):
        cx = XL[j] + CW[j] / 2
        if sb:
            txt(cx, HEADER_Y1 - HDR_H * 0.36, lb, fs=10, fw="bold", color=C_WHITE)
            txt(cx, HEADER_Y1 - HDR_H * 0.73, sb, fs=8,             color=C_GRAY)
        else:
            txt(cx, HEADER_Y1 - HDR_H / 2,    lb, fs=10, fw="bold", color=C_WHITE)

    # ── Data background ───────────────────────────────────────────────────────
    rect(0, DATA_Y1, 1.0, DATA_Y1 - DATA_Y0, "#fafafa")

    # ── Data rows ─────────────────────────────────────────────────────────────
    for ci, (_, row) in enumerate(overall.iterrows()):
        label  = COHORT_LABELS.get(str(row["Cohort"]), str(row["Cohort"]))
        N      = int(row["N"])
        Events = int(row["Events"])
        NetNRI = int(row.get("NetCorrected", 0))

        block_y1 = DATA_Y1 - ci * len(METRICS) * ROW_H
        block_h  = len(METRICS) * ROW_H
        block_cy = block_y1 - block_h / 2

        # thick separator between cohort blocks
        if ci > 0:
            ax.axhline(block_y1, color=C_DARK, lw=1.8, clip_on=False, zorder=4)

        # 코호트 이름 + 임상 AUC / N / Events (코호트 블록 전체에 세로 중앙 정렬)
        auc_val = float(row.get("Clinical_AUC", np.nan))
        auc_str = f"AUC {auc_val:.3f}" if np.isfinite(auc_val) else ""
        label_y = block_cy + ROW_H * 0.3 if auc_str else block_cy
        txt(XL[0] + CW[0] / 2, label_y, label, fs=11, fw="bold")
        if auc_str:
            txt(XL[0] + CW[0] / 2, block_cy - ROW_H * 0.3, auc_str, fs=9, color="#1a6e3c")
        txt(XL[1] + CW[1] / 2, block_cy, str(N),      fs=10, color=C_GRAY)
        txt(XL[2] + CW[2] / 2, block_cy, str(Events), fs=10, color=C_GRAY)

        # Net NRI badge (vertically centered in block)
        nri_s = f"+{NetNRI}" if NetNRI >= 0 else str(NetNRI)
        bw, bh = CW[6] * 0.65, ROW_H * 0.55
        bx = XL[6] + (CW[6] - bw) / 2
        rect(bx, block_cy + bh / 2, bw, bh, C_BADGE, ec="#aaccee", lw=0.9, z=2)
        txt(XL[6] + CW[6] / 2, block_cy, nri_s, fs=12, fw="bold", color=C_BADGE_T)

        # per-metric rows
        for mi, (mname, cl_col, wf_col, diff_col, p_col) in enumerate(METRICS):
            row_y1 = block_y1 - mi * ROW_H
            row_cy = row_y1 - ROW_H / 2

            # light separator between metrics (skip first)
            if mi > 0:
                ax.axhline(row_y1, color=C_SEP_L, lw=0.7,
                           xmin=XL[3], xmax=1.0, clip_on=False, zorder=3)

            # 지표 label
            txt(XL[3] + CW[3] / 2, row_cy, mname, fs=10, color=C_GRAY)

            # Clinical value
            cl_v = float(row.get(cl_col, np.nan))
            txt(XL[4] + CW[4] / 2, row_cy,
                f"{cl_v:.3f}" if np.isfinite(cl_v) else "-",
                fs=10, color=C_GRAY)

            # AEC value
            wf_v   = float(row.get(wf_col,   np.nan))
            diff_v = float(row.get(diff_col, np.nan))
            p_v    = float(row.get(p_col,    1.0))

            wf_str = f"{wf_v:.3f}" if np.isfinite(wf_v) else "-"

            # diff annotation and color  (ASCII only — avoid Unicode symbols)
            p_raw   = p_fmt(p_v)
            p_label = f"p{p_raw}" if p_raw.startswith("<") else f"p={p_raw}"
            if np.isfinite(diff_v):
                if abs(diff_v) < 5e-4:
                    ann  = f"(±{abs(diff_v):.3f})"
                    sig  = p_label
                    dcol = C_GRAY
                elif diff_v > 0:
                    ann  = f"(+{diff_v:.3f})"
                    sig  = p_label
                    dcol = C_GREEN if p_v < 0.05 else C_GRAY
                else:
                    ann  = f"(-{abs(diff_v):.3f})"
                    sig  = p_label
                    dcol = C_RED
            else:
                ann, sig, dcol = "", "", C_GRAY

            wf_x  = XL[5] + CW[5] * 0.28
            ann_x = XL[5] + CW[5] * 0.70
            txt(wf_x,  row_cy, wf_str,                  fs=10, color=C_BODY)
            txt(ann_x, row_cy, f"{ann} {sig}".strip(),  fs=9,  color=dcol)

    # ── Footer ────────────────────────────────────────────────────────────────
    txt(0.01, (FOOTER_Y1 + FOOTER_Y0) / 2,
        "* p < 0.05 (유의)    n.s. p ≥ 0.05 (비유의)    Net NRI: AEC 추가 시 순 재분류 개선 환자 수",
        ha="left", fs=8, color=C_GRAY)

    fig.savefig(OUT_DIR / "연구결과.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# =============================================================================
# 7. 메인
# =============================================================================


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    progress(0, 10, "강남·신촌 데이터 로드")
    dev, dev_aec_cols = load_cohort(GDATA_XLSX, "Gangnam_internal_CV") # type: ignore
    ext, ext_aec_cols = load_cohort(SDATA_XLSX, "Shinchon_sdata_external") # type: ignore

    y_dev = dev["LowSMI_Derstine"].astype(int).to_numpy()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(dev, y_dev))

    progress(1, 10, "강남 5-fold 임상 모델 + 대조적 AEC 점수 계산")
    clin_dev, ro_dev, ri_dev, contrast_dev = selective_scores_oof(dev, dev_aec_cols, folds)

    progress(2, 10, "강남 내부 CV에서만 임계값 선택")
    sweep = sweep_thresholds(dev, clin_dev, ro_dev, ri_dev)
    sweep.to_csv(OUT_DIR / "internal_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    selected = choose_internal_rules(sweep)
    selected["Model"] = "Meaning-aware contrastive SEC"
    selected["Cohort"] = "Gangnam_internal_CV"
    selected.to_csv(OUT_DIR / "selected_thresholds.csv", index=False, encoding="utf-8-sig")
    compact_table(selected).to_csv(OUT_DIR / "selected_thresholds_compact.csv", index=False, encoding="utf-8-sig")

    # 최종 기본 규칙: 우선순위 fallback 체인
    _rule_priority = ["no_sens_drop", "sens_p_guard", "sens_drop_le_0p02", "max_net"]
    final_rule = None
    for _rule_name in _rule_priority:
        _cand = selected[selected["Rule"] == _rule_name]
        if not _cand.empty:
            final_rule = _cand.iloc[0]
            print(f"[Rule selected] {_rule_name}")
            break
    if final_rule is None:
        raise RuntimeError("threshold sweep에서 유효한 규칙을 찾지 못했습니다. 하이퍼파라미터를 확인하세요.")

    progress(3, 10, "강남 전체로 학습 후 신촌에 고정 모델 적용")
    _, clin_ext = clinical_full_train_external(dev, ext)
    ro_ext, ri_ext, contrast_ext = selective_scores_external(dev, ext, dev_aec_cols, ext_aec_cols, folds, clin_ext)

    progress(4, 10, "내부·외부 전체 성능 평가")
    ro_thr = float(final_rule["RuleOutThreshold"])
    ri_thr = float(final_rule["RuleInThreshold"])
    margin_gate = None if str(final_rule["MarginGate"]) == "none" else float(final_rule["MarginGate"])
    overall_rows = []
    row_dev = eval_selective(dev, clin_dev, ro_dev, ri_dev, ro_thr, ri_thr, margin_gate)
    row_dev.update({"Cohort": "Gangnam_internal_CV", "Rule": "no_sens_drop"}) # type: ignore
    row_ext = eval_selective(ext, clin_ext, ro_ext, ri_ext, ro_thr, ri_thr, margin_gate)
    row_ext.update({"Cohort": "Shinchon_sdata_external", "Rule": "no_sens_drop"}) # type: ignore
    overall_rows.extend([row_dev, row_ext])
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUT_DIR / "overall_results.csv", index=False, encoding="utf-8-sig")
    compact_table(overall).to_csv(OUT_DIR / "overall_results_compact.csv", index=False, encoding="utf-8-sig")

    progress(5, 10, "성별·위상·스캐너 서브그룹 평가")
    subgroup = pd.concat(
        [
            subgroup_rows(dev, clin_dev, ro_dev, ri_dev, final_rule, "Gangnam_internal_CV"),
            subgroup_rows(ext, clin_ext, ro_ext, ri_ext, final_rule, "Shinchon_sdata_external"),
        ],
        ignore_index=True,
    )
    subgroup.to_csv(OUT_DIR / "subgroup_results.csv", index=False, encoding="utf-8-sig")
    compact_table(subgroup).to_csv(OUT_DIR / "subgroup_results_compact.csv", index=False, encoding="utf-8-sig")

    progress(6, 10, "환자 레벨 감사 및 rule-out/rule-in 건수 저장")
    audit = pd.concat(
        [
            patient_audit_frame(dev, "Gangnam_internal_CV", clin_dev, ro_dev, ri_dev, contrast_dev, final_rule),
            patient_audit_frame(ext, "Shinchon_sdata_external", clin_ext, ro_ext, ri_ext, contrast_ext, final_rule),
        ],
        ignore_index=True,
    )
    audit.to_csv(OUT_DIR / "patient_level_workflow.csv", index=False, encoding="utf-8-sig")
    counts = workflow_counts(audit)
    counts.to_csv(OUT_DIR / "workflow_ruleout_rulein_counts.csv", index=False, encoding="utf-8-sig")

    progress(7, 10, "저SMI vs 정상 AEC 시그니처 플롯 저장")
    plot_signature(dev, ext, dev_aec_cols, ext_aec_cols)

    progress(8, 10, "ROC 곡선 저장")
    plot_roc_curves(
        dev, ext,
        clin_dev, clin_ext,
        ro_dev, ri_dev,
        ro_ext, ri_ext,
        contrast_dev, contrast_ext,
        final_rule,
    )

    progress(9, 10, "연구결과 테이블 이미지 저장")
    plot_results_table(overall)

    progress(10, 10, "완료")

    show_cols = [
        "Cohort",
        "N",
        "Events",
        "Clinical_Sensitivity",
        "Workflow_Sensitivity",
        "Sensitivity_p",
        "Clinical_Specificity",
        "Workflow_Specificity",
        "Specificity_p",
        "Clinical_Accuracy",
        "Workflow_Accuracy",
        "Accuracy_p",
        "NetCorrected",
    ]
    subgroup_show = subgroup[
        subgroup["GroupType"].isin(["Overall", "Sex", "Phase"])
        | ((subgroup["GroupType"] == "Scanner") & (subgroup["N"] >= 100))
    ].copy()
    print("\nFinal selected thresholds")
    print(compact_table(pd.DataFrame([final_rule])).to_string(index=False))
    print("\nOverall results")
    print(compact_table(overall)[show_cols].to_string(index=False))
    print("\nRule-out / Rule-in counts")
    print(counts.to_string(index=False))
    print("\nSubgroup results: overall, sex, phase, and scanners with N >= 100")
    sub_cols = ["Cohort", "GroupType", "Group"] + show_cols[1:]
    print(compact_table(subgroup_show)[sub_cols].to_string(index=False))
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
