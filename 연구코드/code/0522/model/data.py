"""
데이터 로드·전처리·분할 유틸리티.

모든 품질 필터(kVp, 소수 제조사, 연령, IQR, AEC 이상치)는
build_dataset.py에서 사전 적용되어 merged_features.xlsx에 저장된다.
이 모듈은 Excel에서 필요한 컬럼을 읽고 레이블을 생성하며 train/test를 분할한다.

모델별 로드 함수:
  Model 1       : load_data()
  Model 2/2_2   : load_data_with_aec() / load_data_with_aec_unmatched()
  Model 3       : load_data_with_aec_meta()

AEC 민감도 분석:
  aec_variant() — 7가지 변환(해상도·범위·정규화·극단치 제외) 적용
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DATA_PATH, AEC_SHEET, AEC_LEN, SEED, TEST_SIZE, SMI_THRESH_M, SMI_THRESH_F, AEC_SHUFFLE_SEED


def _make_label(row):
    """성별 기준 SMI 임계값을 적용해 sarcopenia(1) / normal(0) 이진 레이블을 반환."""
    thresh = SMI_THRESH_M if row["PatientSex"] == "M" else SMI_THRESH_F
    return 1 if row["SMI"] <= thresh else 0


_filtered_meta_cache: pd.DataFrame | None = None


def _load_filtered_meta():
    """metadata 시트를 읽어 모델 학습에 필요한 컬럼을 반환한다.

    필터링은 build_dataset.py에서 사전 적용되므로 컬럼 선택만 수행한다.
    모듈 내 캐시를 사용해 동일 데이터를 반복 로드하지 않는다.
    """
    global _filtered_meta_cache
    if _filtered_meta_cache is not None:
        return _filtered_meta_cache.copy()

    df = pd.read_excel(DATA_PATH, sheet_name="metadata")
    base_cols = ["PatientID", "PatientAge", "PatientSex", "BMI", "SMI",
                 "kVp", "ManufacturerModelName"]
    optional_cols = [c for c in ["TAMA"] if c in df.columns]
    df = df[base_cols + optional_cols].dropna().reset_index(drop=True)
    df["PatientID"] = df["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    _filtered_meta_cache = df
    return df.copy()


def load_data():
    """임상 데이터(Age, Sex, BMI)와 sarcopenia 레이블을 로드해 (X, y, sex)를 반환."""
    df = _load_filtered_meta()

    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    X   = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)
    y   = df["label"].values.astype(np.int64)
    sex = df["PatientSex"].values
    return X, y, sex


def _strat_key(y, sex, age=None, bmi=None, n_bins: int = 3):
    """label × sex × age_bin × bmi_bin 조합 stratification key."""
    key = y * 2 + (sex == "M").astype(int)
    if age is not None:
        age_bin = pd.qcut(age, q=n_bins, labels=False, duplicates="drop").astype(int)
        key = key * n_bins + age_bin
    if bmi is not None:
        bmi_bin = pd.qcut(bmi, q=n_bins, labels=False, duplicates="drop").astype(int)
        key = key * n_bins + bmi_bin
    return key


def _safe_strat_key(y, sex, age, bmi):
    """strata 최소 크기 < 2이면 bin 수를 줄여 재시도. 최종 fallback은 label × sex."""
    for n_bins in (3, 2):
        key = _strat_key(y, sex, age=age, bmi=bmi, n_bins=n_bins)
        if pd.Series(key).value_counts().min() >= 2:
            return key
    key = _strat_key(y, sex)
    return key if pd.Series(key).value_counts().min() >= 2 else y


def split_data(X, y, sex):
    """label × sex × age × bmi stratified train/test split. (X_cv, y_cv, sex_cv, X_te, y_te, sex_te) 반환."""
    idx = np.arange(len(y))
    cv_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED,
        stratify=_safe_strat_key(y, sex, age=X[:, 0], bmi=X[:, 2]),
    )
    return (
        X[cv_idx], y[cv_idx], sex[cv_idx],
        X[te_idx], y[te_idx], sex[te_idx],
    )


def load_data_with_aec(aec_len: int = AEC_LEN, aec_sheet: str = AEC_SHEET):
    """Clinic + AEC 결합 데이터 로드. PatientID 기준 inner join."""
    df_meta = _load_filtered_meta()

    df_aec = pd.read_excel(DATA_PATH, sheet_name=aec_sheet)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    aec_cols = [c for c in df_aec.columns if c != "PatientID"][:aec_len]

    df = pd.merge(df_meta, df_aec[["PatientID"] + aec_cols], on="PatientID", how="inner").reset_index(drop=True)

    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    X_clin = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)
    X_aec  = df[aec_cols].values.astype(np.float32)
    y      = df["label"].values.astype(np.int64)
    sex    = df["PatientSex"].values
    return X_clin, X_aec, y, sex


def load_data_with_aec_unmatched(aec_len: int = AEC_LEN, aec_sheet: str = AEC_SHEET):
    """
    Model 2_2용: load_data_with_aec()와 동일한 데이터를 로드하되
    X_aec 행 순서를 AEC_SHUFFLE_SEED로 무작위 permutation하여
    Clinic-AEC 환자 대응을 의도적으로 해제한다.

    label(y)과 sex는 Clinic 데이터(X_clin) 기준으로 유지.
    AEC_SHUFFLE_SEED != SEED 이므로 우연한 재정렬이 발생하지 않는다.
    """
    X_clin, X_aec, y, sex = load_data_with_aec(aec_len=aec_len, aec_sheet=aec_sheet)
    rng  = np.random.default_rng(AEC_SHUFFLE_SEED)
    perm = rng.permutation(len(y))
    return X_clin, X_aec[perm], y, sex


def split_data_dual(X_clin, X_aec, y, sex):
    """Clinic + AEC 배열을 label × sex × age × bmi stratified split. CV/test 8개 배열을 반환."""
    idx = np.arange(len(y))
    cv_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED,
        stratify=_safe_strat_key(y, sex, age=X_clin[:, 0], bmi=X_clin[:, 2]),
    )
    return (
        X_clin[cv_idx], X_aec[cv_idx], y[cv_idx], sex[cv_idx],
        X_clin[te_idx], X_aec[te_idx], y[te_idx], sex[te_idx],
    )


def load_data_with_aec_meta(aec_len: int = AEC_LEN, aec_sheet: str = AEC_SHEET):
    """
    Clinic (Age, Sex, BMI) + Scanner (kVp, ManufacturerModelName) + AEC 결합.

    Returns
    -------
    X_clin      : (N, 3) float  — PatientAge, sex_enc, BMI
    X_aec       : (N, aec_len) float
    X_scan_mfr  : (N,) int64    — ManufacturerModelName (1-indexed 정수 인코딩)
    y, sex, n_manufacturers
    """
    df_meta = _load_filtered_meta()

    df_aec = pd.read_excel(DATA_PATH, sheet_name=aec_sheet)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    aec_cols = [c for c in df_aec.columns if c != "PatientID"][:aec_len]

    df = pd.merge(df_meta, df_aec[["PatientID"] + aec_cols], on="PatientID", how="inner").reset_index(drop=True)

    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    le = LabelEncoder()
    # 1-indexed: 1, 2, 3, ... n  (0은 Embedding padding 슬롯으로 예약)
    mfr_encoded: np.ndarray = np.array(le.fit_transform(df["ManufacturerModelName"]), dtype=int) + 1
    df["mfr_enc"] = mfr_encoded
    n_manufacturers = len(le.classes_)

    X_clin     = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)
    X_aec      = df[aec_cols].values.astype(np.float32)
    X_scan_mfr = df["mfr_enc"].values.astype(np.int64)
    y          = df["label"].values.astype(np.int64)
    sex        = df["PatientSex"].values
    return X_clin, X_aec, X_scan_mfr, y, sex, n_manufacturers


def split_data_quad(X_clin, X_aec, X_scan_mfr, y, sex):
    """Clinic + AEC + Scanner(mfr) 배열을 label × sex × age × bmi stratified split. CV/test 10개 배열을 반환."""
    idx = np.arange(len(y))
    cv_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED,
        stratify=_safe_strat_key(y, sex, age=X_clin[:, 0], bmi=X_clin[:, 2]),
    )
    return (
        X_clin[cv_idx], X_aec[cv_idx], X_scan_mfr[cv_idx], y[cv_idx], sex[cv_idx],
        X_clin[te_idx], X_aec[te_idx], X_scan_mfr[te_idx], y[te_idx], sex[te_idx],
    )


def aec_variant(X_aec: np.ndarray, variant: str):
    """
    AEC 민감도 분석용 변환.

    Parameters
    ----------
    X_aec   : (N, P) float32
    variant : "len064" | "len128" | "len256" |
              "crop80" | "crop60" | "norm" | "excl_extreme"

    Returns
    -------
    X_out : (N', K) float32  — 변환된 AEC
    mask  : (N,) bool | None — excl_extreme 시 유효 샘플 마스크, 나머지는 None
    """
    N, P = X_aec.shape

    # ── 해상도 변환 (선형 보간) ───────────────────────────────
    # 단순 잘라내기(truncation)가 아닌 선형 보간으로 리샘플링해
    # 원본 곡선 전체 구간을 균등하게 표현한다
    if variant == "len064":
        x_old = np.linspace(0, 1, P)
        x_new = np.linspace(0, 1, 64)
        return np.array([np.interp(x_new, x_old, row) for row in X_aec], dtype=np.float32), None
    if variant == "len128":
        return X_aec.copy(), None   # baseline: 원본 그대로 (AEC_LEN = 128)
    if variant == "len256":
        x_old = np.linspace(0, 1, P)
        x_new = np.linspace(0, 1, 256)
        return np.array([np.interp(x_new, x_old, row) for row in X_aec], dtype=np.float32), None

    # ── 시간 범위 자르기 ─────────────────────────────────────
    # 곡선 양끝은 스캐너 특성(램프업/다운)에 영향받으므로 제거 효과를 확인한다
    if variant == "crop80":
        s, e = int(0.1 * P), int(0.9 * P)
        return X_aec[:, s:e], None
    if variant == "crop60":
        s, e = int(0.2 * P), int(0.8 * P)
        return X_aec[:, s:e], None

    # ── 곡선 내 정규화 ───────────────────────────────────────
    # 스캐너별 절대 선량 수준 차이를 제거하고 곡선 형태만 남긴다
    if variant == "norm":
        mu = X_aec.mean(axis=1, keepdims=True)
        sd = X_aec.std(axis=1,  keepdims=True) + 1e-8
        return ((X_aec - mu) / sd).astype(np.float32), None

    # ── 극단 샘플 제외 ───────────────────────────────────────
    # 곡선 면적(합산값)을 scan length proxy로 사용해 상하위 5% 제거
    # 반환 mask로 X_clin·y·sex 배열도 동기화해야 한다 (main.py 참고)
    if variant == "excl_extreme":
        proxy = X_aec.sum(axis=1)
        lo, hi = np.percentile(proxy, [5, 95])
        mask = (proxy >= lo) & (proxy <= hi)
        return X_aec[mask], mask

    raise ValueError(f"Unknown AEC variant: {variant!r}")


def describe_dataset() -> None:
    """Train/Test split 전 전체 데이터셋의 분포를 출력한다.

    출력 항목:
      - 성비 (전체 / 남 / 여)
      - Age, BMI, SMI, TAMA(있는 경우): mean, SD, P25, Median, P75 by sex
      - Sarcopenia 유병률 by sex
    """
    df = _load_filtered_meta()
    df = df.copy()
    df["sarcopenia"] = df.apply(_make_label, axis=1)

    sep = "=" * 66
    print(f"\n{sep}")
    print("  DATASET DESCRIPTION  (full dataset, before train/test split)")
    print(sep)

    n_total = len(df)
    n_m = (df["PatientSex"] == "M").sum()
    n_f = (df["PatientSex"] == "F").sum()
    print(f"\n  Total  : {n_total}")
    print(f"  Male   : {n_m:>5}  ({n_m / n_total * 100:.1f}%)")
    print(f"  Female : {n_f:>5}  ({n_f / n_total * 100:.1f}%)")

    cont_vars = [
        ("Age   (years)", "PatientAge"),
        ("BMI   (kg/m²)", "BMI"),
        ("SMI   (cm²/m²)", "SMI"),
    ]
    if "TAMA" in df.columns:
        cont_vars.append(("TAMA  (cm²)", "TAMA"))

    hdr = f"  {'':14}  {'N':>5}  {'Mean':>8}  {'SD':>8}  {'P25':>8}  {'Median':>8}  {'P75':>8}"
    div = "  " + "-" * (len(hdr) - 2)

    groups = [("Overall", None), ("Male", "M"), ("Female", "F")]
    for var_label, col in cont_vars:
        print(f"\n  ── {var_label}")
        print(hdr)
        print(div)
        for grp_label, sex_val in groups:
            sub = df[col] if sex_val is None else df[df["PatientSex"] == sex_val][col]
            print(
                f"  {grp_label:<14}  {len(sub):>5}  {sub.mean():>8.2f}  {sub.std():>8.2f}"
                f"  {sub.quantile(0.25):>8.2f}  {sub.median():>8.2f}  {sub.quantile(0.75):>8.2f}"
            )

    print(f"\n  ── Sarcopenia prevalence  (M ≤ {SMI_THRESH_M} cm²/m²,  F ≤ {SMI_THRESH_F} cm²/m²)")
    print(f"  {'':14}  {'N':>5}  {'Positive':>9}  {'Prevalence':>10}")
    print(div)
    for grp_label, sex_val in groups:
        sub = df if sex_val is None else df[df["PatientSex"] == sex_val]
        pos = int(sub["sarcopenia"].sum())
        n   = len(sub)
        print(f"  {grp_label:<14}  {n:>5}  {pos:>9}  {pos / n * 100:>9.1f}%")

    print()


def print_stats(y, sex):
    """데이터셋 통계(샘플 수, 성별 분포, sarcopenia 비율)를 콘솔 출력."""
    print(f"Samples : {len(y)}")
    for s in ["M", "F"]:
        mask = sex == s
        pos  = y[mask].sum()
        print(f"  {s}: {mask.sum()} samples | Positive {pos} ({pos/mask.sum()*100:.1f}%)")
