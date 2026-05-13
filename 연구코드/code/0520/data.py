"""
0520 데이터 로드·전처리·분할 유틸리티.

0515 data.py의 모든 함수를 포함하며, 아래 함수를 추가한다:
  load_data_with_smi()          — SMI 연속값 회귀용 (M6)
  load_data_sexsplit_aec()      — 성별 분리 데이터 (M5)
  split_data_sexsplit()         — 성별 분리 후 각 sex별 stratified split

공통 전처리 (_load_filtered_meta):
  1. kVp == 100 필터
  2. 소수 제조사 제거 (비율 < MIN_MFR_RATIO)
  3. SMI 이진 레이블 생성
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (DATA_PATH, AEC_SHEET, AEC_LEN, SEED, TEST_SIZE,
                    SMI_THRESH_M, SMI_THRESH_F, AEC_SHUFFLE_SEED, MIN_MFR_RATIO)


def _make_label(row):
    """성별 기준 SMI 임계값 적용 → sarcopenia(1) / normal(0)."""
    thresh = SMI_THRESH_M if row["PatientSex"] == "M" else SMI_THRESH_F
    return 1 if row["SMI"] <= thresh else 0


def _load_filtered_meta():
    """kVp==100 필터 + 소수 제조사 제거된 metadata DataFrame 반환."""
    df = pd.read_excel(DATA_PATH, sheet_name="metadata")
    base_cols = ["PatientID", "PatientAge", "PatientSex", "BMI", "SMI",
                 "kVp", "ManufacturerModelName"]
    optional_cols = [c for c in ["TAMA"] if c in df.columns]
    df = df[base_cols + optional_cols].dropna().reset_index(drop=True)
    df["PatientID"] = df["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    before_kvp = len(df)
    df = df[df["kVp"] == 100].reset_index(drop=True)
    print(f"[Filter] kVp==100: {len(df)} / {before_kvp} samples retained")

    mfr_counts = df["ManufacturerModelName"].value_counts()
    total      = len(df)
    major_mfr  = mfr_counts[mfr_counts / total >= MIN_MFR_RATIO].index
    minor_mfr  = mfr_counts[mfr_counts / total < MIN_MFR_RATIO]
    if len(minor_mfr) > 0:
        removed = total - df["ManufacturerModelName"].isin(major_mfr).sum()
        print(f"[Filter] Minor manufacturer (threshold={MIN_MFR_RATIO:.0%}): "
              f"{removed} samples removed")
        df = df[df["ManufacturerModelName"].isin(major_mfr)].reset_index(drop=True)

    return df


# ── 0515 호환 로드 함수 ────────────────────────────────────────

def load_data():
    """M1용: Clinic 피처(Age, Sex, BMI) + 이진 레이블."""
    df = _load_filtered_meta()
    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)
    X   = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)
    y   = df["label"].values.astype(np.int64)
    sex = df["PatientSex"].values
    return X, y, sex


def split_data(X, y, sex):
    """y 기준 stratified split → (X_cv, y_cv, sex_cv, X_te, y_te, sex_te)."""
    idx = np.arange(len(y))
    cv_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=SEED, stratify=y.tolist())
    return (X[cv_idx], y[cv_idx], sex[cv_idx],
            X[te_idx], y[te_idx], sex[te_idx])


def load_data_with_aec():
    """M2용: Clinic + AEC(256) 매칭 데이터."""
    df_meta = _load_filtered_meta()
    df_aec  = pd.read_excel(DATA_PATH, sheet_name=AEC_SHEET)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    aec_cols = [c for c in df_aec.columns if c != "PatientID"][:AEC_LEN]

    df = pd.merge(df_meta, df_aec[["PatientID"] + aec_cols], on="PatientID", how="inner").reset_index(drop=True)
    flat_mask = df[aec_cols].std(axis=1) == 0
    if flat_mask.any():
        print(f"[Filter] Flat AEC removed: {flat_mask.sum()}")
    df = df[~flat_mask].reset_index(drop=True)

    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)
    X_clin = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)
    X_aec  = df[aec_cols].values.astype(np.float32)
    y      = df["label"].values.astype(np.int64)
    sex    = df["PatientSex"].values
    return X_clin, X_aec, y, sex


def load_data_with_aec_unmatched():
    """M2_2용: AEC 행 순서 셔플(Clinic-AEC unmatching)."""
    X_clin, X_aec, y, sex = load_data_with_aec()
    rng  = np.random.default_rng(AEC_SHUFFLE_SEED)
    perm = rng.permutation(len(y))
    return X_clin, X_aec[perm], y, sex


def split_data_dual(X_clin, X_aec, y, sex):
    """Clinic + AEC → stratified split → 8개 배열."""
    idx = np.arange(len(y))
    cv_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=SEED, stratify=y.tolist())
    return (X_clin[cv_idx], X_aec[cv_idx], y[cv_idx], sex[cv_idx],
            X_clin[te_idx], X_aec[te_idx], y[te_idx], sex[te_idx])


def load_data_with_aec_meta():
    """M3용: Clinic + ManufacturerModelName embedding + AEC."""
    df_meta = _load_filtered_meta()
    df_aec  = pd.read_excel(DATA_PATH, sheet_name=AEC_SHEET)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    aec_cols = [c for c in df_aec.columns if c != "PatientID"][:AEC_LEN]

    df = pd.merge(df_meta, df_aec[["PatientID"] + aec_cols], on="PatientID", how="inner").reset_index(drop=True)
    flat_mask = df[aec_cols].std(axis=1) == 0
    if flat_mask.any():
        df = df[~flat_mask].reset_index(drop=True)

    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    le = LabelEncoder()
    mfr_encoded = np.array(le.fit_transform(df["ManufacturerModelName"]), dtype=int) + 1
    n_manufacturers = len(le.classes_)

    X_clin     = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)
    X_aec      = df[aec_cols].values.astype(np.float32)
    X_scan_mfr = mfr_encoded.astype(np.int64)
    y          = df["label"].values.astype(np.int64)
    sex        = df["PatientSex"].values
    return X_clin, X_aec, X_scan_mfr, y, sex, n_manufacturers


def split_data_quad(X_clin, X_aec, X_scan_mfr, y, sex):
    """Clinic + AEC + Scanner → stratified split → 10개 배열."""
    idx = np.arange(len(y))
    cv_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=SEED, stratify=y.tolist())
    return (X_clin[cv_idx], X_aec[cv_idx], X_scan_mfr[cv_idx], y[cv_idx], sex[cv_idx],
            X_clin[te_idx], X_aec[te_idx], X_scan_mfr[te_idx], y[te_idx], sex[te_idx])


def aec_variant(X_aec: np.ndarray, variant: str):
    """0515 aec_variant와 동일한 7가지 AEC 민감도 변환."""
    N, P = X_aec.shape
    if variant == "len064":
        x_old, x_new = np.linspace(0, 1, P), np.linspace(0, 1, 64)
        return np.array([np.interp(x_new, x_old, row) for row in X_aec], dtype=np.float32), None
    if variant == "len128":
        x_old, x_new = np.linspace(0, 1, P), np.linspace(0, 1, 128)
        return np.array([np.interp(x_new, x_old, row) for row in X_aec], dtype=np.float32), None
    if variant == "len256":
        return X_aec.copy(), None
    if variant == "crop80":
        s, e = int(0.1 * P), int(0.9 * P)
        return X_aec[:, s:e], None
    if variant == "crop60":
        s, e = int(0.2 * P), int(0.8 * P)
        return X_aec[:, s:e], None
    if variant == "norm":
        mu = X_aec.mean(axis=1, keepdims=True)
        sd = X_aec.std(axis=1, keepdims=True) + 1e-8
        return ((X_aec - mu) / sd).astype(np.float32), None
    if variant == "excl_extreme":
        proxy = X_aec.sum(axis=1)
        lo, hi = np.percentile(proxy, [5, 95])
        mask = (proxy >= lo) & (proxy <= hi)
        return X_aec[mask], mask
    raise ValueError(f"Unknown AEC variant: {variant!r}")


# ── 0520 새 로드 함수 ──────────────────────────────────────────

def load_data_with_smi():
    """
    M6(SMI 회귀)용: Clinic + AEC + 연속 SMI 타깃.

    Returns
    -------
    X_clin : (N, 3)   — Age, sex_enc, BMI
    X_aec  : (N, 256) — AEC 커브
    y_smi  : (N,)     — 연속 SMI 값 (회귀 타깃)
    y_bin  : (N,)     — 이진 Sarcopenia 레이블 (평가용)
    sex    : (N,)     — 성별 문자열
    """
    df_meta = _load_filtered_meta()
    df_aec  = pd.read_excel(DATA_PATH, sheet_name=AEC_SHEET)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    aec_cols = [c for c in df_aec.columns if c != "PatientID"][:AEC_LEN]

    df = pd.merge(df_meta, df_aec[["PatientID"] + aec_cols], on="PatientID", how="inner").reset_index(drop=True)
    flat_mask = df[aec_cols].std(axis=1) == 0
    if flat_mask.any():
        print(f"[Filter] Flat AEC removed: {flat_mask.sum()}")
    df = df[~flat_mask].reset_index(drop=True)

    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    X_clin = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)
    X_aec  = df[aec_cols].values.astype(np.float32)
    y_smi  = df["SMI"].values.astype(np.float32)
    y_bin  = df["label"].values.astype(np.int64)
    sex    = df["PatientSex"].values
    return X_clin, X_aec, y_smi, y_bin, sex


def split_data_regression(X_clin, X_aec, y_smi, y_bin, sex):
    """SMI 회귀용 split — y_bin 기준 stratified."""
    idx = np.arange(len(y_bin))
    cv_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=SEED, stratify=y_bin.tolist())
    return (X_clin[cv_idx], X_aec[cv_idx], y_smi[cv_idx], y_bin[cv_idx], sex[cv_idx],
            X_clin[te_idx], X_aec[te_idx], y_smi[te_idx], y_bin[te_idx], sex[te_idx])


def load_data_sexsplit_aec():
    """
    M5(성별 분리)용: Clinic + AEC를 남성/여성 서브셋으로 분리해 반환.

    Returns
    -------
    data_M : (X_clin_M, X_aec_M, y_M, sex_M)
    data_F : (X_clin_F, X_aec_F, y_F, sex_F)
    """
    X_clin, X_aec, y, sex = load_data_with_aec()

    mask_m = sex == "M"
    mask_f = sex == "F"

    data_M = (X_clin[mask_m], X_aec[mask_m], y[mask_m], sex[mask_m])
    data_F = (X_clin[mask_f], X_aec[mask_f], y[mask_f], sex[mask_f])

    print(f"[SexSplit] Male  : {mask_m.sum()} samples "
          f"(Sarco={y[mask_m].sum()}, {y[mask_m].mean()*100:.1f}%)")
    print(f"[SexSplit] Female: {mask_f.sum()} samples "
          f"(Sarco={y[mask_f].sum()}, {y[mask_f].mean()*100:.1f}%)")

    return data_M, data_F


def print_stats(y, sex):
    """샘플 수·성별 분포·sarcopenia 비율 콘솔 출력."""
    print(f"Samples : {len(y)}")
    for s in ["M", "F"]:
        mask = sex == s
        pos  = y[mask].sum()
        print(f"  {s}: {mask.sum()} samples | Positive {pos} ({pos/mask.sum()*100:.1f}%)")
