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
  aec_variant() — 4가지 변환(raw·std_scaled·norm·global_zscore) 적용
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

def extract_aec_features_batch(X_aec: np.ndarray) -> np.ndarray:
    """AEC 시퀀스 배치 (N, T)에서 통계 피처 60개를 추출해 (N, 60) float32 배열로 반환.

    피처 순서:
      ── 기본 통계 ──
      [ 0] mean           — 전체 평균
      [ 1] std            — 전체 표준편차
      [ 2] max            — 최댓값 (peak)
      [ 3] min            — 최솟값
      [ 4] peak_pos       — argmax / T
      [ 5] auc            — trapz 면적
      [ 6] skew           — 왜도
      [ 7] kurt           — 첨도
      ── 구간 평균 (1/3) ──
      [ 8] early_mean     — 앞 1/3
      [ 9] mid_mean       — 중간 1/3
      [10] late_mean      — 뒤 1/3
      ── 구간 평균 (1/4) ──
      [11] q1_mean        — 앞 1/4
      [12] q2_mean        — 2/4
      [13] q3_mean        — 3/4
      [14] q4_mean        — 뒤 1/4
      ── 백분위 ──
      [15] p10
      [16] p25
      [17] p50  (median)
      [18] p75
      [19] p90
      [20] iqr            — p75 - p25
      ── 형태·변동성 ──
      [21] range          — max - min
      [22] cv             — std / (|mean| + 1e-8)
      [23] rms            — root mean square
      [24] energy         — sum(x²) / T
      ── 위치 ──
      [25] valley_pos     — argmin / T
      [26] first_val      — 시작점 값
      [27] last_val       — 종료점 값
      ── 기울기·면적 분할 ──
      [28] slope_rise     — (max - first) / (peak_idx + 1)
      [29] slope_fall     — (last - max) / (T - peak_idx)
      [30] rise_auc       — peak 이전 trapz
      [31] fall_auc       — peak 이후 trapz
      ── 1차 차분 ──
      [32] diff_abs_mean  — mean(|Δx|)
      [33] diff_std       — std(Δx)
      [34] diff_abs_max   — max(|Δx|)
      ── 비율·임계 ──
      [35] above_mean_ratio — (x > mean) 비율
      ── 확장: 백분위 추가 ──
      [36] p5
      [37] p95
      ── 확장: 2차 차분 ──
      [38] diff2_abs_mean — mean(|Δ²x|)
      [39] diff2_std      — std(Δ²x)
      [40] diff2_abs_max  — max(|Δ²x|)
      ── 확장: 자기상관 ──
      [41] autocorr_lag1  — lag-1 정규화 자기상관
      [42] autocorr_lag2  — lag-2 정규화 자기상관
      ── 확장: FFT 주파수 분석 ──
      [43] fft_power_low  — 저주파 대역 파워
      [44] fft_power_mid  — 중주파 대역 파워
      [45] fft_power_high — 고주파 대역 파워
      [46] spectral_centroid — FFT 스펙트럼 무게중심
      ── 확장: 구간 표준편차 ──
      [47] early_std      — 앞 1/3 표준편차
      [48] mid_std        — 중간 1/3 표준편차
      [49] late_std       — 뒤 1/3 표준편차
      ── 확장: 임계 비율 ──
      [50] above_p75_ratio — (x > p75) 비율
      [51] below_p25_ratio — (x < p25) 비율
      ── 확장: 파생 비율 ──
      [52] auc_ratio      — rise_auc / (fall_auc + 1e-8)
      [53] symmetry_index — |peak_pos - 0.5|
      [54] mean_to_max    — mean / (max + 1e-8)
      [55] late_to_early  — late_mean / (early_mean + 1e-8)
      [56] start_to_end   — first_val / (last_val + 1e-8)
      ── 확장: 곡선 형태 ──
      [57] peak_half_dur  — (x > 0.5*max) 비율 (반최댓값 폭)
      [58] valley_depth   — min / (max + 1e-8)
      [59] tail_mean      — 뒤 10% 구간 평균
    """
    from scipy.stats import skew as _skew, kurtosis as _kurt
    N, T = X_aec.shape
    seg  = T // 3
    qrt  = T // 4
    tail = max(1, int(T * 0.9))

    peak_idxs   = X_aec.argmax(axis=1)
    valley_idxs = X_aec.argmin(axis=1)
    peak_vals   = X_aec.max(axis=1)
    min_vals    = X_aec.min(axis=1)
    mean_vals   = X_aec.mean(axis=1)
    std_vals    = X_aec.std(axis=1)
    first_vals  = X_aec[:, 0]
    last_vals   = X_aec[:, -1]

    slope_rise = (peak_vals - first_vals) / (peak_idxs.astype(float) + 1)
    slope_fall = (last_vals - peak_vals) / (T - peak_idxs.astype(float))

    rise_aucs = np.array([np.trapz(X_aec[i, :peak_idxs[i] + 1]) for i in range(N)], dtype=np.float32)
    fall_aucs = np.array([np.trapz(X_aec[i, peak_idxs[i]:])     for i in range(N)], dtype=np.float32)

    diffs  = np.diff(X_aec, axis=1)          # (N, T-1)
    diffs2 = np.diff(diffs,  axis=1)          # (N, T-2)

    p5  = np.percentile(X_aec, 5,  axis=1)
    p10 = np.percentile(X_aec, 10, axis=1)
    p25 = np.percentile(X_aec, 25, axis=1)
    p50 = np.percentile(X_aec, 50, axis=1)
    p75 = np.percentile(X_aec, 75, axis=1)
    p90 = np.percentile(X_aec, 90, axis=1)
    p95 = np.percentile(X_aec, 95, axis=1)

    var_vals = std_vals ** 2 + 1e-8
    x_c = X_aec - mean_vals[:, None]
    autocorr1 = (x_c[:, :-1] * x_c[:, 1:]).mean(axis=1) / var_vals
    autocorr2 = (x_c[:, :-2] * x_c[:, 2:]).mean(axis=1) / var_vals

    fft_mag  = np.abs(np.fft.rfft(X_aec, axis=1))   # (N, T//2+1)
    n_fft    = fft_mag.shape[1]
    lb1, lb2 = max(1, n_fft // 3), max(1, 2 * n_fft // 3)
    power    = fft_mag ** 2
    total_p  = power[:, 1:].sum(axis=1) + 1e-8
    fft_power_low  = power[:, 1:lb1].sum(axis=1) / total_p
    fft_power_mid  = power[:, lb1:lb2].sum(axis=1) / total_p
    fft_power_high = power[:, lb2:].sum(axis=1) / total_p
    freq_idx = np.arange(n_fft, dtype=float)
    spec_cent = (freq_idx * fft_mag).sum(axis=1) / (fft_mag.sum(axis=1) + 1e-8)

    early_mean = X_aec[:, :seg].mean(axis=1)
    late_mean  = X_aec[:, 2 * seg:].mean(axis=1)

    return np.column_stack([
        # 0-7: 기본 통계
        mean_vals,
        std_vals,
        peak_vals,
        min_vals,
        peak_idxs.astype(float) / T,
        np.trapezoid(X_aec, axis=1),
        _skew(X_aec, axis=1),
        _kurt(X_aec, axis=1),
        # 8-10: 구간 평균 (1/3)
        early_mean,
        X_aec[:, seg:2 * seg].mean(axis=1),
        late_mean,
        # 11-14: 구간 평균 (1/4)
        X_aec[:, :qrt].mean(axis=1),
        X_aec[:, qrt:2 * qrt].mean(axis=1),
        X_aec[:, 2 * qrt:3 * qrt].mean(axis=1),
        X_aec[:, 3 * qrt:].mean(axis=1),
        # 15-20: 백분위
        p10, p25, p50, p75, p90,
        p75 - p25,
        # 21-24: 형태·변동성
        peak_vals - min_vals,
        std_vals / (np.abs(mean_vals) + 1e-8),
        np.sqrt((X_aec ** 2).mean(axis=1)),
        (X_aec ** 2).sum(axis=1) / T,
        # 25-27: 위치
        valley_idxs.astype(float) / T,
        first_vals,
        last_vals,
        # 28-31: 기울기·면적 분할
        slope_rise,
        slope_fall,
        rise_aucs,
        fall_aucs,
        # 32-34: 1차 차분
        np.abs(diffs).mean(axis=1),
        diffs.std(axis=1),
        np.abs(diffs).max(axis=1),
        # 35: 비율
        (X_aec > mean_vals[:, None]).sum(axis=1).astype(float) / T,
        # 36-37: 확장 백분위
        p5, p95,
        # 38-40: 2차 차분
        np.abs(diffs2).mean(axis=1),
        diffs2.std(axis=1),
        np.abs(diffs2).max(axis=1),
        # 41-42: 자기상관
        autocorr1,
        autocorr2,
        # 43-46: FFT 주파수 분석
        fft_power_low,
        fft_power_mid,
        fft_power_high,
        spec_cent / (n_fft + 1e-8),
        # 47-49: 구간 표준편차
        X_aec[:, :seg].std(axis=1),
        X_aec[:, seg:2 * seg].std(axis=1),
        X_aec[:, 2 * seg:].std(axis=1),
        # 50-51: 임계 비율
        (X_aec > p75[:, None]).sum(axis=1).astype(float) / T,
        (X_aec < p25[:, None]).sum(axis=1).astype(float) / T,
        # 52-56: 파생 비율
        rise_aucs / (fall_aucs + 1e-8),
        np.abs(peak_idxs.astype(float) / T - 0.5),
        mean_vals / (peak_vals + 1e-8),
        late_mean / (early_mean + 1e-8),
        first_vals / (last_vals + 1e-8),
        # 57-59: 곡선 형태
        (X_aec > 0.5 * peak_vals[:, None]).sum(axis=1).astype(float) / T,
        min_vals / (peak_vals + 1e-8),
        X_aec[:, tail:].mean(axis=1),
    ]).astype(np.float32)


def load_data_with_aec_features(aec_len: int = AEC_LEN, aec_sheet: str = AEC_SHEET):
    """Clinic(3) + AEC hand-crafted 통계 피처(11) = 총 14개 결합 벡터를 반환.

    열 구성:
      [0] PatientAge  [1] sex_enc  [2] BMI   ← 임상 피처 (M1과 동일)
      [3..13]         AEC 통계 피처 11개      ← extract_aec_features_batch 순서

    Returns
    -------
    X_combined : (N, 14) float32
    y          : (N,) int64
    sex        : (N,) str
    """
    X_clin, X_aec, y, sex = load_data_with_aec(aec_len=aec_len, aec_sheet=aec_sheet)
    X_aec_feats = extract_aec_features_batch(X_aec)
    return np.concatenate([X_clin, X_aec_feats], axis=1), y, sex


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