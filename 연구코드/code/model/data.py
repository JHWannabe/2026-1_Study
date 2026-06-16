"""
데이터 로드·전처리·분할 유틸리티.

모든 품질 필터(kVp, 소수 제조사, 연령, IQR, AEC 이상치)는
build_dataset.py에서 사전 적용되어 merged_features.xlsx에 저장된다.
이 모듈은 Excel에서 필요한 컬럼을 읽고 레이블을 생성하며 train/test를 분할한다.

모델별 로드·피처 추출 함수:
  Model 1       : load_data()
  Model 2       : load_data_with_aec()
  Model 3       : load_data_with_aec() + extract_aec_features_m3()  → (N, 54)
  Model 4       : load_data_with_aec() + extract_aec_features_m4()  → (N, 98)
  Model 5       : load_data_with_aec() + extract_aec_features_m5()  → (N, 282)

AEC 민감도 분석:
  aec_variant() — 3가지 변환(raw·norm·global_zscore) 적용
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DATA_PATH, AEC_SHEET, AEC_LEN, SEED, TEST_SIZE, SMI_THRESH_M, SMI_THRESH_F, AEC_SHUFFLE_SEED

def _apply_crop(X_aec: np.ndarray, aec_len: int, crop_points: int | None) -> np.ndarray:
    """중앙 crop_points개 포인트를 추출. None이거나 aec_len 이상이면 그대로 반환."""
    if crop_points is None or crop_points >= aec_len:
        return X_aec
    start = (aec_len - crop_points) // 2
    return X_aec[:, start:start + crop_points]

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

def _aec_global_stats(X_aec: np.ndarray) -> np.ndarray:
    """전체 신호 글로벌 통계 26개. (N, 26) float32.

    mean, std, max, min, AUC, skew, kurt
    p10, p25, p50, p75, p90, IQR
    range, CV, RMS
    first_val, last_val, peak_pos, valley_pos
    autocorr_lag1, autocorr_lag2
    fft_low, fft_mid, fft_high, fft_centroid
    """
    from scipy.stats import skew as _skew, kurtosis as _kurt
    _, T = X_aec.shape

    mean_vals   = X_aec.mean(axis=1)
    std_vals    = X_aec.std(axis=1)
    max_vals    = X_aec.max(axis=1)
    min_vals    = X_aec.min(axis=1)
    peak_idxs   = X_aec.argmax(axis=1)
    valley_idxs = X_aec.argmin(axis=1)
    first_vals  = X_aec[:, 0]
    last_vals   = X_aec[:, -1]

    p10 = np.percentile(X_aec, 10, axis=1)
    p25 = np.percentile(X_aec, 25, axis=1)
    p50 = np.percentile(X_aec, 50, axis=1)
    p75 = np.percentile(X_aec, 75, axis=1)
    p90 = np.percentile(X_aec, 90, axis=1)

    var_vals  = std_vals ** 2 + 1e-8
    x_c       = X_aec - mean_vals[:, None]
    autocorr1 = (x_c[:, :-1] * x_c[:, 1:]).mean(axis=1) / var_vals
    autocorr2 = (x_c[:, :-2] * x_c[:, 2:]).mean(axis=1) / var_vals

    fft_mag   = np.abs(np.fft.rfft(X_aec, axis=1))
    n_fft     = fft_mag.shape[1]
    lb1, lb2  = max(1, n_fft // 3), max(1, 2 * n_fft // 3)
    power     = fft_mag ** 2
    total_p   = power[:, 1:].sum(axis=1) + 1e-8
    fft_low   = power[:, 1:lb1].sum(axis=1) / total_p
    fft_mid   = power[:, lb1:lb2].sum(axis=1) / total_p
    fft_high  = power[:, lb2:].sum(axis=1) / total_p
    spec_cent = (np.arange(n_fft, dtype=float) * fft_mag).sum(axis=1) / (fft_mag.sum(axis=1) + 1e-8)

    return np.column_stack([
        mean_vals, std_vals, max_vals, min_vals,
        np.trapezoid(X_aec, axis=1),
        _skew(X_aec, axis=1), _kurt(X_aec, axis=1),
        p10, p25, p50, p75, p90, p75 - p25,
        max_vals - min_vals,
        std_vals / (np.abs(mean_vals) + 1e-8),
        np.sqrt((X_aec ** 2).mean(axis=1)),
        first_vals, last_vals,
        peak_idxs.astype(float) / T,
        valley_idxs.astype(float) / T,
        autocorr1, autocorr2,
        fft_low, fft_mid, fft_high,
        spec_cent / (n_fft + 1e-8),
    ]).astype(np.float32)


def _aec_seg_stats(X_aec: np.ndarray, k: int) -> np.ndarray:
    """X_aec (N, T)를 k등분해 구간별 통계를 반환.
    k=4  → mean×4 + std×4 + max×4 + min×4  = (N, 16)
    k=8  → mean×8 + std×8                   = (N, 16)
    k=16 → mean×16                           = (N, 16)
    """
    _, T  = X_aec.shape
    seg   = T // k
    segs  = [X_aec[:, i * seg:(i + 1) * seg] for i in range(k)]
    means = np.column_stack([s.mean(axis=1) for s in segs])
    stds  = np.column_stack([s.std(axis=1)  for s in segs])
    if k == 4:
        maxes = np.column_stack([s.max(axis=1) for s in segs])
        mins  = np.column_stack([s.min(axis=1) for s in segs])
        return np.column_stack([means, stds, maxes, mins]).astype(np.float32)
    if k == 8:
        return np.column_stack([means, stds]).astype(np.float32)
    return means.astype(np.float32)  # k == 16


def _aec_seg_means(X_aec: np.ndarray, k: int) -> np.ndarray:
    """X_aec (N, T)를 k등분해 구간 mean만 (N, k) float32로 반환."""
    _, T = X_aec.shape
    seg  = T // k
    return np.column_stack(
        [X_aec[:, i * seg:(i + 1) * seg].mean(axis=1) for i in range(k)]
    ).astype(np.float32)


def _aec_pairwise(seg_means: np.ndarray) -> np.ndarray:
    """(N, k) 구간 평균에서 모든 pair (i < j)의 ratio + diff를 (N, k*(k-1)) 배열로 반환.
    인접한 pair뿐 아니라 전체 C(k,2) 쌍의 관계를 모두 포함한다.
    """
    _, k  = seg_means.shape
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    ratios = np.column_stack([seg_means[:, i] / (seg_means[:, j] + 1e-8) for i, j in pairs])
    diffs  = np.column_stack([seg_means[:, i] -  seg_means[:, j]          for i, j in pairs])
    return np.column_stack([ratios, diffs]).astype(np.float32)


def extract_aec_features_m3(X_aec: np.ndarray) -> np.ndarray:
    """Model 3: 전체 글로벌 통계(26) + 4등분 구간통계(16) + 4등분 전체 pairwise(12) → (N, 54).

    4등분 구간통계: mean×4 + std×4 + max×4 + min×4 = 16
    4등분 pairwise: C(4,2)=6 pairs × (ratio+diff)   = 12
    """
    return np.column_stack([
        _aec_global_stats(X_aec),
        _aec_seg_stats(X_aec, 4),
        _aec_pairwise(_aec_seg_means(X_aec, 4)),
    ]).astype(np.float32)


def extract_aec_features_m4(X_aec: np.ndarray) -> np.ndarray:
    """Model 4: 전체 글로벌 통계(26) + 8등분 구간통계(16) + 8등분 전체 pairwise(56) → (N, 98).

    8등분 구간통계: mean×8 + std×8                   = 16
    8등분 pairwise: C(8,2)=28 pairs × (ratio+diff)   = 56
    """
    return np.column_stack([
        _aec_global_stats(X_aec),
        _aec_seg_stats(X_aec, 8),
        _aec_pairwise(_aec_seg_means(X_aec, 8)),
    ]).astype(np.float32)


def extract_aec_features_m5(X_aec: np.ndarray) -> np.ndarray:
    """Model 5: 전체 글로벌 통계(26) + 16등분 구간통계(16) + 16등분 전체 pairwise(240) → (N, 282).

    16등분 구간통계: mean×16                             = 16
    16등분 pairwise: C(16,2)=120 pairs × (ratio+diff)   = 240
    """
    return np.column_stack([
        _aec_global_stats(X_aec),
        _aec_seg_stats(X_aec, 16),
        _aec_pairwise(_aec_seg_means(X_aec, 16)),
    ]).astype(np.float32)


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

def load_data_with_aec(aec_len: int = AEC_LEN, aec_sheet: str = AEC_SHEET,
                       crop_points: int | None = None):
    """Clinic + AEC 결합 데이터 로드. PatientID 기준 inner join."""
    df_meta = _load_filtered_meta()

    df_aec = pd.read_excel(DATA_PATH, sheet_name=aec_sheet)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    aec_cols = sorted(
        [c for c in df_aec.columns if str(c).startswith("aec_")],
        key=lambda x: int(str(x).split("_")[1]),
    )[:aec_len]

    df = pd.merge(df_meta, df_aec[["PatientID"] + aec_cols], on="PatientID", how="inner").reset_index(drop=True)

    df["label"]   = df.apply(_make_label, axis=1)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    X_clin = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)
    X_aec  = _apply_crop(df[aec_cols].values.astype(np.float32), aec_len, crop_points)
    y      = df["label"].values.astype(np.int64)
    sex    = df["PatientSex"].values
    return X_clin, X_aec, y, sex


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

def aec_variant(X_aec: np.ndarray, variant: str):
    """
    AEC 민감도 분석용 변환.

    Parameters
    ----------
    X_aec   : (N, P) float32
    variant : "raw" | "std_scaled" | "norm" | "global_zscore"

    Returns
    -------
    X_out      : (N, P) float32  — 변환된 AEC
    mask       : None             — 샘플 필터링 없음 (하위 호환용)
    scale_mode : str              — "none" | "column" | "global"
                                   호출자가 이 모드에 따라 AEC에 추가 스케일링을 적용한다.
                                   "column" → StandardScaler (열 방향, train fold에서 fit)
                                   "global" → Train set 전체의 단일 mean/std로 정규화
                                   "none"   → 추가 스케일링 없음
    """
    if variant == "raw":
        return X_aec.copy(), None, "none"

    if variant == "norm":
        # 행 방향 z-score (환자별 절대 선량 수준 제거, 곡선 형태 보존)
        mu = X_aec.mean(axis=1, keepdims=True)
        sd = X_aec.std(axis=1,  keepdims=True) + 1e-8
        return ((X_aec - mu) / sd).astype(np.float32), None, "none"

    if variant == "global_zscore":
        # Train set 전체의 단일 mean/std로 정규화 — cross_val/evaluate에서 적용
        return X_aec.copy(), None, "global"

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