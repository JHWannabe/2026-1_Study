import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DATA_PATH, AEC_SHEET, AEC_LEN, SEED, TEST_SIZE, SMI_THRESH_M, SMI_THRESH_F, AEC_SHUFFLE_SEED, MIN_MFR_RATIO


def _make_label(row):
    thresh = SMI_THRESH_M if row["PatientSex"] == "M" else SMI_THRESH_F
    return 1 if row["SMI"] <= thresh else 0


def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name="metadata")
    df = df[["PatientAge", "PatientSex", "BMI", "SMI"]].dropna().reset_index(drop=True)

    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    X   = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)
    y   = df["label"].values.astype(np.int64)
    sex = df["PatientSex"].values
    return X, y, sex


def split_data(X, y, sex):
    idx = np.arange(len(y))
    cv_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, stratify=y.tolist()
    )
    return (
        X[cv_idx], y[cv_idx], sex[cv_idx],
        X[te_idx], y[te_idx], sex[te_idx],
    )


def load_data_with_aec():
    """Clinic + AEC 결합 데이터 로드. PatientID 기준 inner join."""
    df_meta = pd.read_excel(DATA_PATH, sheet_name="metadata")
    df_meta = df_meta[["PatientID", "PatientAge", "PatientSex", "BMI", "SMI"]].dropna().reset_index(drop=True)
    df_meta["PatientID"] = df_meta["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    df_aec = pd.read_excel(DATA_PATH, sheet_name=AEC_SHEET)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    aec_cols = [c for c in df_aec.columns if c != "PatientID"][:AEC_LEN]

    df = pd.merge(df_meta, df_aec[["PatientID"] + aec_cols], on="PatientID", how="inner").reset_index(drop=True)

    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    X_clin = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)
    X_aec  = df[aec_cols].values.astype(np.float32)
    y      = df["label"].values.astype(np.int64)
    sex    = df["PatientSex"].values
    return X_clin, X_aec, y, sex


def load_data_with_aec_unmatched():
    """
    Model 2_2용: load_data_with_aec()와 동일한 데이터를 로드하되
    X_aec 행 순서를 AEC_SHUFFLE_SEED로 무작위 permutation하여
    Clinic-AEC 환자 대응을 의도적으로 해제한다.

    label(y)과 sex는 Clinic 데이터(X_clin) 기준으로 유지.
    AEC_SHUFFLE_SEED != SEED 이므로 우연한 재정렬이 발생하지 않는다.
    """
    X_clin, X_aec, y, sex = load_data_with_aec()
    rng  = np.random.default_rng(AEC_SHUFFLE_SEED)
    perm = rng.permutation(len(y))
    return X_clin, X_aec[perm], y, sex


def split_data_dual(X_clin, X_aec, y, sex):
    idx = np.arange(len(y))
    cv_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, stratify=y.tolist()
    )
    return (
        X_clin[cv_idx], X_aec[cv_idx], y[cv_idx], sex[cv_idx],
        X_clin[te_idx], X_aec[te_idx], y[te_idx], sex[te_idx],
    )


def load_data_with_aec_meta():
    """
    Clinic (Age, Sex, BMI) + Scanner (kVp, ManufacturerModelName) + AEC 결합.

    Returns
    -------
    X_clin      : (N, 3) float  — PatientAge, sex_enc, BMI
    X_aec       : (N, 256) float
    X_scan_kvp  : (N,) float    — kVp (연속형 스캐너 파라미터)
    X_scan_mfr  : (N,) int64    — ManufacturerModelName (1-indexed 정수 인코딩)
    y, sex, n_manufacturers
    """
    df_meta = pd.read_excel(DATA_PATH, sheet_name="metadata")
    df_meta = df_meta[
        ["PatientID", "PatientAge", "PatientSex", "BMI", "SMI", "kVp", "ManufacturerModelName"]
    ].dropna().reset_index(drop=True)
    df_meta["PatientID"] = df_meta["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    df_aec = pd.read_excel(DATA_PATH, sheet_name=AEC_SHEET)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    aec_cols = [c for c in df_aec.columns if c != "PatientID"][:AEC_LEN]

    df = pd.merge(df_meta, df_aec[["PatientID"] + aec_cols], on="PatientID", how="inner").reset_index(drop=True)

    # 전체 대비 비율이 MIN_MFR_RATIO 미만인 소수 제조사 제거
    mfr_counts = df["ManufacturerModelName"].value_counts()
    total_before = len(df)
    major_mfr = mfr_counts[mfr_counts / total_before >= MIN_MFR_RATIO].index
    minor_mfr = mfr_counts[mfr_counts / total_before < MIN_MFR_RATIO]
    if len(minor_mfr) > 0:
        removed = total_before - df["ManufacturerModelName"].isin(major_mfr).sum()
        print(f"[Model 3] Minor manufacturer filter (threshold={MIN_MFR_RATIO:.0%}):")
        for mname, cnt in minor_mfr.items():
            print(f"  - '{mname}': {cnt} samples ({cnt/total_before:.1%}) → removed")
        print(f"  Total removed: {removed} samples  ({total_before} → {total_before - removed})")
        df = df[df["ManufacturerModelName"].isin(major_mfr)].reset_index(drop=True)

    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    le = LabelEncoder()
    # 1-indexed: 1, 2, 3, ... n  (0은 Embedding padding 슬롯으로 예약)
    mfr_encoded: np.ndarray = np.array(le.fit_transform(df["ManufacturerModelName"]), dtype=int) + 1
    df["mfr_enc"] = mfr_encoded
    n_manufacturers = len(le.classes_)

    X_clin     = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)  # 환자 임상
    X_aec      = df[aec_cols].values.astype(np.float32)
    X_scan_kvp = df["kVp"].values.astype(np.float32)                             # 스캐너 연속형
    X_scan_mfr = df["mfr_enc"].values.astype(np.int64)                           # 스캐너 범주형
    y          = df["label"].values.astype(np.int64)
    sex        = df["PatientSex"].values
    return X_clin, X_aec, X_scan_kvp, X_scan_mfr, y, sex, n_manufacturers


def split_data_quad(X_clin, X_aec, X_scan_kvp, X_scan_mfr, y, sex):
    idx = np.arange(len(y))
    cv_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, stratify=y.tolist()
    )
    return (
        X_clin[cv_idx], X_aec[cv_idx], X_scan_kvp[cv_idx], X_scan_mfr[cv_idx], y[cv_idx], sex[cv_idx],
        X_clin[te_idx], X_aec[te_idx], X_scan_kvp[te_idx], X_scan_mfr[te_idx], y[te_idx], sex[te_idx],
    )


def print_stats(y, sex):
    print(f"Samples : {len(y)}")
    for s in ["M", "F"]:
        mask = sex == s
        pos  = y[mask].sum()
        print(f"  {s}: {mask.sum()} samples | Positive {pos} ({pos/mask.sum()*100:.1f}%)")
