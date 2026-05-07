from pathlib import Path
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.stats as stats
import pywt

FILE_PATH = Path(__file__).parents[2] / "data" / "강남_merged_features.xlsx"
META_SHEET = "metadata-bmi_add"
RAW_SHEET  = "aec-raw"
N_INTERP   = 256
FS         = 1.0
BAND_EDGES = [0, FS / 8, FS / 4, FS * 3 / 8, FS / 2]
META_COLS  = ["PatientAge", "PatientSex", "BMI", "TAMA", "SMI"]


def interpolate_signal(values: np.ndarray, n: int = N_INTERP) -> np.ndarray:
    valid = values[~np.isnan(values)]
    x_old = np.linspace(0, 1, len(valid))
    x_new = np.linspace(0, 1, n)
    f = interp.interp1d(x_old, valid, kind="linear")
    return f(x_new)


def extract_features(x: np.ndarray, fs: float = 1.0) -> dict:
    x = np.asarray(x, dtype=float)
    N = len(x)
    feats = {}

    feats["signal_length"]      = N
    feats["mean"]               = np.mean(x)
    feats["std"]                = np.std(x, ddof=1)
    feats["min"]                = np.min(x)
    feats["max"]                = np.max(x)
    feats["range"]              = np.max(x) - np.min(x)
    feats["median"]             = np.median(x)
    feats["IQR"]                = np.percentile(x, 75) - np.percentile(x, 25)
    feats["skewness"]           = stats.skew(x) if np.std(x) > 1e-10 else np.nan
    feats["kurtosis"]           = stats.kurtosis(x) if np.std(x) > 1e-10 else np.nan
    feats["p5"]                 = np.percentile(x, 5)
    feats["p10"]                = np.percentile(x, 10)
    feats["p25"]                = np.percentile(x, 25)
    feats["p75"]                = np.percentile(x, 75)
    feats["p90"]                = np.percentile(x, 90)
    feats["p95"]                = np.percentile(x, 95)
    feats["p90_p10_ratio"]      = feats["p90"] / feats["p10"] if feats["p10"] != 0 else np.nan
    feats["CV"]                 = feats["std"] / feats["mean"] if feats["mean"] != 0 else np.nan
    feats["RMSE"]               = np.sqrt(np.mean(x ** 2))
    feats["signal_energy"]      = np.sum(x ** 2)
    feats["mean_abs_deviation"] = np.mean(np.abs(x - feats["mean"]))

    peaks, props = sig.find_peaks(x, height=feats["mean"])
    peak_heights = props["peak_heights"] if len(peaks) > 0 else np.array([])

    feats["peak_count"]       = len(peaks)
    feats["peak_max_height"]  = np.max(peak_heights)  if len(peaks) > 0 else np.nan
    feats["peak_mean_height"] = np.mean(peak_heights) if len(peaks) > 0 else np.nan
    feats["peak_std_height"]  = np.std(peak_heights)  if len(peaks) > 1 else 0.0
    feats["peak_first_pos"]   = int(peaks[0])          if len(peaks) > 0 else np.nan
    feats["peak_last_pos"]    = int(peaks[-1])          if len(peaks) > 0 else np.nan
    feats["peak_main_pos"]    = int(peaks[np.argmax(peak_heights)]) if len(peaks) > 0 else np.nan

    if len(peaks) > 0:
        widths = sig.peak_widths(x, peaks, rel_height=0.5)[0]
        feats["peak_mean_width"] = np.mean(widths)
        feats["peak_max_width"]  = np.max(widths)
    else:
        feats["peak_mean_width"] = np.nan
        feats["peak_max_width"]  = np.nan

    valleys, _ = sig.find_peaks(-x)
    feats["valley_count"] = len(valleys)

    slopes = np.diff(x)
    feats["slope_mean"]     = np.mean(slopes)
    feats["slope_std"]      = np.std(slopes, ddof=1)
    feats["slope_max"]      = np.max(slopes)
    feats["slope_min"]      = np.min(slopes)
    feats["slope_abs_mean"] = np.mean(np.abs(slopes))

    x_centered = x - feats["mean"]
    zero_crossings = np.where(np.diff(np.sign(x_centered)))[0]
    feats["zero_crossing_rate"] = len(zero_crossings) / N

    t = np.arange(N) / fs
    feats["AUC"]            = np.trapezoid(x, t)
    feats["AUC_normalized"] = feats["AUC"] / (N / fs)

    high_idx = np.where(x > feats["p75"])[0]
    feats["first_high_pos"] = int(high_idx[0]) if len(high_idx) > 0 else np.nan

    fft_mags = np.abs(np.fft.rfft(x))
    freqs    = np.fft.rfftfreq(N, d=1.0 / fs)

    feats["fft_mag_mean"]  = np.mean(fft_mags)
    feats["fft_mag_max"]   = np.max(fft_mags)
    feats["fft_mag_std"]   = np.std(fft_mags)
    feats["dominant_freq"] = freqs[np.argmax(fft_mags)]

    sum_mags         = np.sum(fft_mags)
    total_fft_energy = np.sum(fft_mags ** 2)
    feats["spectral_energy"]   = total_fft_energy
    feats["spectral_centroid"] = np.sum(freqs * fft_mags) / sum_mags if sum_mags != 0 else np.nan
    feats["spectral_spread"]   = np.sqrt(
        np.sum(((freqs - feats["spectral_centroid"]) ** 2) * fft_mags) / sum_mags
    ) if sum_mags != 0 else np.nan

    cumulative_energy = np.cumsum(fft_mags ** 2)
    rolloff_idx = np.searchsorted(cumulative_energy, 0.85 * total_fft_energy)
    feats["spectral_rolloff"] = freqs[min(rolloff_idx, len(freqs) - 1)]

    for i in range(4):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        band_mask = (freqs >= lo) & (freqs < hi)
        band_e = np.sum(fft_mags[band_mask] ** 2)
        feats[f"band{i+1}_energy"]       = band_e
        feats[f"band{i+1}_energy_ratio"] = band_e / total_fft_energy if total_fft_energy != 0 else np.nan

    coeffs = pywt.wavedec(x, wavelet="db4", level=3)
    cA, cD3, cD2, cD1 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

    feats["wavelet_cA_energy"]  = np.sum(cA  ** 2)
    feats["wavelet_cA_std"]     = np.std(cA)
    feats["wavelet_cD3_energy"] = np.sum(cD3 ** 2)
    feats["wavelet_cD3_std"]    = np.std(cD3)
    feats["wavelet_cD2_energy"] = np.sum(cD2 ** 2)
    feats["wavelet_cD2_std"]    = np.std(cD2)
    feats["wavelet_cD1_energy"] = np.sum(cD1 ** 2)
    feats["wavelet_cD1_std"]    = np.std(cD1)

    total_wavelet_energy = (
        feats["wavelet_cA_energy"] + feats["wavelet_cD3_energy"] +
        feats["wavelet_cD2_energy"] + feats["wavelet_cD1_energy"]
    )
    feats["wavelet_energy_ratio_D1"] = (
        feats["wavelet_cD1_energy"] / total_wavelet_energy
        if total_wavelet_energy != 0 else np.nan
    )

    return feats


FEATURE_DESC = [
    ("signal_length",           "Signal Info",         "신호 길이 (샘플 수)"),
    ("mean",                    "Basic Statistics",    "평균"),
    ("std",                     "Basic Statistics",    "표준편차"),
    ("min",                     "Basic Statistics",    "최솟값"),
    ("max",                     "Basic Statistics",    "최댓값"),
    ("range",                   "Basic Statistics",    "범위 (max − min)"),
    ("median",                  "Basic Statistics",    "중앙값"),
    ("IQR",                     "Basic Statistics",    "사분위 범위 (Q3 − Q1)"),
    ("skewness",                "Basic Statistics",    "왜도"),
    ("kurtosis",                "Basic Statistics",    "첨도"),
    ("p5",                      "Basic Statistics",    "5 백분위수"),
    ("p10",                     "Basic Statistics",    "10 백분위수"),
    ("p25",                     "Basic Statistics",    "25 백분위수 (Q1)"),
    ("p75",                     "Basic Statistics",    "75 백분위수 (Q3)"),
    ("p90",                     "Basic Statistics",    "90 백분위수"),
    ("p95",                     "Basic Statistics",    "95 백분위수"),
    ("p90_p10_ratio",           "Basic Statistics",    "p90 / p10 비율"),
    ("CV",                      "Basic Statistics",    "변동계수 (std / mean)"),
    ("RMSE",                    "Basic Statistics",    "제곱평균제곱근"),
    ("signal_energy",           "Basic Statistics",    "신호 에너지 (Σx²)"),
    ("mean_abs_deviation",      "Basic Statistics",    "평균 절대 편차"),
    ("peak_count",              "Shape",               "피크 개수"),
    ("peak_max_height",         "Shape",               "피크 최대 높이"),
    ("peak_mean_height",        "Shape",               "피크 평균 높이"),
    ("peak_std_height",         "Shape",               "피크 높이 표준편차"),
    ("peak_first_pos",          "Shape",               "첫 번째 피크 위치 (인덱스)"),
    ("peak_last_pos",           "Shape",               "마지막 피크 위치 (인덱스)"),
    ("peak_main_pos",           "Shape",               "최대 높이 피크 위치 (인덱스)"),
    ("peak_mean_width",         "Shape",               "피크 평균 너비 (반높이 기준)"),
    ("peak_max_width",          "Shape",               "피크 최대 너비 (반높이 기준)"),
    ("valley_count",            "Shape",               "밸리 개수"),
    ("slope_mean",              "Shape",               "기울기 (1차 차분) 평균"),
    ("slope_std",               "Shape",               "기울기 표준편차"),
    ("slope_max",               "Shape",               "기울기 최댓값"),
    ("slope_min",               "Shape",               "기울기 최솟값"),
    ("slope_abs_mean",          "Shape",               "기울기 절대값 평균"),
    ("zero_crossing_rate",      "Shape",               "영점 교차율"),
    ("AUC",                     "Shape",               "곡선 아래 면적 (사다리꼴 적분)"),
    ("AUC_normalized",          "Shape",               "정규화 AUC (신호 길이 기준)"),
    ("first_high_pos",          "Shape",               "75th percentile 첫 초과 위치"),
    ("fft_mag_mean",            "Frequency & Wavelet", "FFT 크기 평균"),
    ("fft_mag_max",             "Frequency & Wavelet", "FFT 크기 최댓값"),
    ("fft_mag_std",             "Frequency & Wavelet", "FFT 크기 표준편차"),
    ("dominant_freq",           "Frequency & Wavelet", "주요 주파수 (FFT 크기 최대 지점)"),
    ("spectral_energy",         "Frequency & Wavelet", "스펙트럼 에너지"),
    ("spectral_centroid",       "Frequency & Wavelet", "스펙트럼 무게중심 주파수"),
    ("spectral_spread",         "Frequency & Wavelet", "스펙트럼 확산도"),
    ("spectral_rolloff",        "Frequency & Wavelet", "85% 에너지 기준 롤오프 주파수"),
    ("band1_energy",            "Frequency & Wavelet", "주파수 밴드 1 에너지"),
    ("band1_energy_ratio",      "Frequency & Wavelet", "주파수 밴드 1 에너지 비율"),
    ("band2_energy",            "Frequency & Wavelet", "주파수 밴드 2 에너지"),
    ("band2_energy_ratio",      "Frequency & Wavelet", "주파수 밴드 2 에너지 비율"),
    ("band3_energy",            "Frequency & Wavelet", "주파수 밴드 3 에너지"),
    ("band3_energy_ratio",      "Frequency & Wavelet", "주파수 밴드 3 에너지 비율"),
    ("band4_energy",            "Frequency & Wavelet", "주파수 밴드 4 에너지"),
    ("band4_energy_ratio",      "Frequency & Wavelet", "주파수 밴드 4 에너지 비율"),
    ("wavelet_cA_energy",       "Frequency & Wavelet", "웨이블릿 근사 계수(cA3) 에너지"),
    ("wavelet_cA_std",          "Frequency & Wavelet", "웨이블릿 근사 계수(cA3) 표준편차"),
    ("wavelet_cD3_energy",      "Frequency & Wavelet", "웨이블릿 세부 계수 D3 에너지"),
    ("wavelet_cD3_std",         "Frequency & Wavelet", "웨이블릿 세부 계수 D3 표준편차"),
    ("wavelet_cD2_energy",      "Frequency & Wavelet", "웨이블릿 세부 계수 D2 에너지"),
    ("wavelet_cD2_std",         "Frequency & Wavelet", "웨이블릿 세부 계수 D2 표준편차"),
    ("wavelet_cD1_energy",      "Frequency & Wavelet", "웨이블릿 세부 계수 D1 에너지"),
    ("wavelet_cD1_std",         "Frequency & Wavelet", "웨이블릿 세부 계수 D1 표준편차"),
    ("wavelet_energy_ratio_D1", "Frequency & Wavelet", "D1 계수 에너지 비율 (전체 웨이블릿 에너지 대비)"),
]


def main():
    print("Excel 파일 읽는 중...")
    meta_df = pd.read_excel(FILE_PATH, sheet_name=META_SHEET)
    raw_df  = pd.read_excel(FILE_PATH, sheet_name=RAW_SHEET)

    meta_sub = meta_df[["PatientID"] + META_COLS].copy()
    slice_cols = [c for c in raw_df.columns if c != "PatientID"]
    print(f"  환자 수: {len(raw_df)}, 원시 슬라이스 열: {len(slice_cols)}")

    # ── Interpolation ────────────────────────────────────────────────────────
    print(f"  {N_INTERP}점으로 보간 중...")
    interp_rows = []
    skipped = []
    for _, row in raw_df.iterrows():
        pid    = row["PatientID"]
        values = np.array(row[slice_cols], dtype=float)
        valid  = values[~np.isnan(values)]
        if len(valid) < 2:
            skipped.append(pid)
            continue
        interp_signal = interpolate_signal(values, N_INTERP)
        rec = {"PatientID": pid}
        for i, v in enumerate(interp_signal, start=1):
            rec[f"aec{i}"] = v
        interp_rows.append(rec)

    if skipped:
        print(f"  보간 건너뜀 (유효 슬라이스 < 2): {skipped}")

    interp_df = pd.DataFrame(interp_rows)
    interp_df = interp_df.merge(meta_sub, on="PatientID", how="left")
    aec_cols  = [f"aec{i}" for i in range(1, N_INTERP + 1)]
    interp_df = interp_df[["PatientID"] + META_COLS + aec_cols]
    print(f"  보간 완료: {len(interp_df)}명")

    # ── BMI/SMI 결측치 제외 ──────────────────────────────────────────────────
    interp_filtered_df = interp_df.dropna(subset=["BMI", "SMI"]).reset_index(drop=True)
    n_bmi_smi_dropped  = len(interp_df) - len(interp_filtered_df)
    print(f"  BMI/SMI 결측치 제외: {len(interp_filtered_df)}명 (제거: {n_bmi_smi_dropped}명)")

    # ── Feature Extraction ───────────────────────────────────────────────────
    print("피처 추출 중...")
    feat_rows = []
    for _, row in interp_filtered_df.iterrows():
        signal = np.array(row[aec_cols], dtype=float)
        feats  = extract_features(signal, fs=FS)
        feats["PatientID"] = row["PatientID"]
        for c in META_COLS:
            feats[c] = row[c]
        feat_rows.append(feats)

    feat_df   = pd.DataFrame(feat_rows)
    feat_cols = [c for c in feat_df.columns if c not in {"PatientID"} | set(META_COLS)]
    feat_df   = feat_df[["PatientID"] + META_COLS + feat_cols]

    # ── Filtering ────────────────────────────────────────────────────────────
    n_total        = len(feat_df)
    range_zero     = feat_df["range"] == 0
    n_range_zero   = int(range_zero.sum())
    filtered_df    = feat_df[~range_zero].dropna(subset=feat_cols).reset_index(drop=True)
    print(f"  필터링 후: {len(filtered_df)}명 (제거: {n_total - len(filtered_df)}명)")

    # ── aec_feature_filtered 환자의 보간 데이터 ─────────────────────────────
    valid_ids       = set(filtered_df["PatientID"])
    interp_feat_df  = interp_df[interp_df["PatientID"].isin(valid_ids)].reset_index(drop=True)

    # ── feature_dist ─────────────────────────────────────────────────────────
    missing_df = pd.DataFrame({
        "feature":         feat_cols,
        "missing_count":   [feat_df[c].isna().sum() for c in feat_cols],
        "total_count":     n_total,
        "missing_rate(%)": [round(feat_df[c].isna().sum() / n_total * 100, 2) for c in feat_cols],
    })
    desc_df     = pd.DataFrame(FEATURE_DESC, columns=["feature", "category", "description"])
    combined_df = desc_df.merge(missing_df, on="feature", how="left")
    combined_df = combined_df[["category", "feature", "description", "missing_count", "total_count", "missing_rate(%)"]]

    range_zero_summary = pd.DataFrame([{
        "category":        "",
        "feature":         "[range=0 rows]",
        "description":     "range=0인 행 수 (신호가 상수인 환자)",
        "missing_count":   n_range_zero,
        "total_count":     n_total,
        "missing_rate(%)": round(n_range_zero / n_total * 100, 2),
    }])
    dist_df = pd.concat([combined_df, pd.DataFrame([{}]), range_zero_summary], ignore_index=True)

    # ── 저장 ─────────────────────────────────────────────────────────────────
    print("Excel에 저장 중...")
    with pd.ExcelWriter(FILE_PATH, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        interp_df.to_excel(writer,          sheet_name="aec_interpolation",          index=False)
        interp_filtered_df.to_excel(writer, sheet_name="aec_interpolation_filtered", index=False)
        feat_df.to_excel(writer,            sheet_name="aec_feature",                index=False)
        filtered_df.to_excel(writer,        sheet_name="aec_feature_filtered",       index=False)
        interp_feat_df.to_excel(writer,     sheet_name="aec_interpolation_final",    index=False)
        dist_df.to_excel(writer,            sheet_name="feature_dist",               index=False)

    print(f"완료! '{FILE_PATH}'")
    print(f"  aec_interpolation          : {len(interp_df)}행 × {len(interp_df.columns)}열")
    print(f"  aec_interpolation_filtered : {len(interp_filtered_df)}행 × {len(interp_filtered_df.columns)}열")
    print(f"  aec_feature                : {len(feat_df)}행 × {len(feat_df.columns)}열")
    print(f"  aec_feature_filtered       : {len(filtered_df)}행 × {len(filtered_df.columns)}열")
    print(f"  aec_interpolation_final    : {len(interp_feat_df)}행 × {len(interp_feat_df.columns)}열")
    print(f"  feature_dist               : {len(dist_df)}행")


if __name__ == "__main__":
    main()
