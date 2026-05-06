from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as sig
import scipy.stats as stats
import pywt

# ── 설정 ─────────────────────────────────────────────────────────────────────
FILE_PATH = Path(__file__).parents[2] / "data" / "강남_merged_features.xlsx"
SHEET_IN  = "merged"
SHEET_OUT = "aec_feature"

SIGNAL_COLS_PREFIX = "aec_"

# 실제 샘플링 주파수(Hz)로 교체하세요
FS = 1.0

# Hz 단위 주파수 밴드 경계 (FS 기준 4등분 — 도메인에 맞게 조정)
BAND_EDGES = [0, FS / 8, FS / 4, FS * 3 / 8, FS / 2]


def extract_features(signal: np.ndarray, fs: float = 1.0) -> dict:
    x = np.asarray(signal, dtype=float)
    N = len(x)

    feats = {}

    # ── 기본 통계 ──────────────────────────────────────────────────────────
    feats["signal_length"]      = N
    feats["mean"]               = np.mean(x)
    feats["std"]                = np.std(x, ddof=1)
    feats["min"]                = np.min(x)
    feats["max"]                = np.max(x)
    feats["range"]              = np.max(x) - np.min(x)
    feats["median"]             = np.median(x)
    feats["IQR"]                = np.percentile(x, 75) - np.percentile(x, 25)
    feats["skewness"]           = stats.skew(x)
    feats["kurtosis"]           = stats.kurtosis(x)
    feats["p5"]                 = np.percentile(x, 5)
    feats["p10"]                = np.percentile(x, 10)
    feats["p25"]                = np.percentile(x, 25)
    feats["p75"]                = np.percentile(x, 75)
    feats["p90"]                = np.percentile(x, 90)
    feats["p95"]                = np.percentile(x, 95)
    feats["p90_p10_ratio"]      = feats["p90"] / feats["p10"] if feats["p10"] != 0 else np.nan
    feats["CV"]                 = feats["std"] / feats["mean"] if feats["mean"] != 0 else np.nan
    feats["signal_energy"]      = np.sum(x ** 2)
    feats["mean_abs_deviation"] = np.mean(np.abs(x - feats["mean"]))

    # ── 피크 ──────────────────────────────────────────────────────────────
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

    # ── 밸리 ──────────────────────────────────────────────────────────────
    valleys, _ = sig.find_peaks(-x)
    feats["valley_count"] = len(valleys)

    # ── 기울기 (1차 차분) ─────────────────────────────────────────────────
    slopes = np.diff(x)
    feats["slope_mean"]     = np.mean(slopes)
    feats["slope_std"]      = np.std(slopes, ddof=1)
    feats["slope_max"]      = np.max(slopes)
    feats["slope_min"]      = np.min(slopes)
    feats["slope_abs_mean"] = np.mean(np.abs(slopes))

    # ── 기타 시간 영역 ────────────────────────────────────────────────────
    x_centered = x - feats["mean"]
    zero_crossings = np.where(np.diff(np.sign(x_centered)))[0]
    feats["zero_crossing_rate"] = len(zero_crossings) / N

    t = np.arange(N) / fs
    feats["AUC"]            = np.trapezoid(x, t)
    feats["AUC_normalized"] = feats["AUC"] / (N / fs)

    # 신호가 처음으로 75th percentile을 초과하는 위치
    high_idx = np.where(x > feats["p75"])[0]
    feats["first_high_pos"] = int(high_idx[0]) if len(high_idx) > 0 else np.nan

    # ── FFT 주파수 영역 ───────────────────────────────────────────────────
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

    # 누적 에너지 85% 지점의 주파수
    cumulative_energy = np.cumsum(fft_mags ** 2)
    rolloff_idx = np.searchsorted(cumulative_energy, 0.85 * total_fft_energy)
    feats["spectral_rolloff"] = freqs[min(rolloff_idx, len(freqs) - 1)]

    for i in range(4):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        band_mask = (freqs >= lo) & (freqs < hi)
        band_e = np.sum(fft_mags[band_mask] ** 2)
        feats[f"band{i+1}_energy"]       = band_e
        feats[f"band{i+1}_energy_ratio"] = band_e / total_fft_energy if total_fft_energy != 0 else np.nan

    # ── 웨이블릿 (3레벨 분해: cA3, cD3, cD2, cD1) ────────────────────────
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


def main():
    print("Excel 파일 읽는 중...")
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_IN)

    signal_cols = [c for c in df.columns if str(c).startswith(SIGNAL_COLS_PREFIX)]
    print(f"  환자 수: {len(df)}, 신호 열 수: {len(signal_cols)}")

    rows = []
    for idx, row in df.iterrows():
        signal = row[signal_cols].values.astype(float)
        feats  = extract_features(signal, fs=FS)
        feats["PatientID"] = row["PatientID"]
        rows.append(feats)

        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(df)} 처리 완료")

    feat_df = pd.DataFrame(rows)
    cols = ["PatientID"] + [c for c in feat_df.columns if c != "PatientID"]
    feat_df = feat_df[cols]

    print(f"\n피처 수: {len(feat_df.columns) - 1}개")
    print("aec_feature 시트에 저장 중...")

    with pd.ExcelWriter(FILE_PATH, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        feat_df.to_excel(writer, sheet_name=SHEET_OUT, index=False)

    print(f"완료! '{FILE_PATH}' → '{SHEET_OUT}' 시트 저장됨")


if __name__ == "__main__":
    main()