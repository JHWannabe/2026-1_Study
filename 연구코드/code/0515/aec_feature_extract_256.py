from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as sig
import scipy.stats as stats
import pywt

SITE = "신촌"
INPUT_PATH  = Path(__file__).parents[2] / "data" / SITE / f"{SITE}_aec_256_interpolation.xlsx"
OUTPUT_PATH = Path(__file__).parents[2] / "data" / SITE / f"{SITE}_aec_256_features.xlsx"
FS = 1.0
BAND_EDGES = [0, FS / 8, FS / 4, FS * 3 / 8, FS / 2]


def extract_features(x: np.ndarray, fs: float = 1.0) -> dict:
    x = np.asarray(x, dtype=float)
    N = len(x)
    f = {}

    f["signal_length"]      = N
    f["mean"]               = np.mean(x)
    f["std"]                = np.std(x, ddof=1)
    f["min"]                = np.min(x)
    f["max"]                = np.max(x)
    f["range"]              = np.max(x) - np.min(x)
    f["median"]             = np.median(x)
    f["IQR"]                = np.percentile(x, 75) - np.percentile(x, 25)
    f["skewness"]           = stats.skew(x) if np.std(x) > 1e-10 else np.nan
    f["kurtosis"]           = stats.kurtosis(x) if np.std(x) > 1e-10 else np.nan
    f["p5"]                 = np.percentile(x, 5)
    f["p10"]                = np.percentile(x, 10)
    f["p25"]                = np.percentile(x, 25)
    f["p75"]                = np.percentile(x, 75)
    f["p90"]                = np.percentile(x, 90)
    f["p95"]                = np.percentile(x, 95)
    f["p90_p10_ratio"]      = f["p90"] / f["p10"] if f["p10"] != 0 else np.nan
    f["CV"]                 = f["std"] / f["mean"] if f["mean"] != 0 else np.nan
    f["RMSE"]               = np.sqrt(np.mean(x ** 2))
    f["signal_energy"]      = np.sum(x ** 2)
    f["mean_abs_deviation"] = np.mean(np.abs(x - f["mean"]))

    peaks, props = sig.find_peaks(x, height=f["mean"])
    peak_heights = props["peak_heights"] if len(peaks) > 0 else np.array([])

    f["peak_count"]       = len(peaks)
    f["peak_max_height"]  = np.max(peak_heights)  if len(peaks) > 0 else np.nan
    f["peak_mean_height"] = np.mean(peak_heights) if len(peaks) > 0 else np.nan
    f["peak_std_height"]  = np.std(peak_heights)  if len(peaks) > 1 else 0.0
    f["peak_first_pos"]   = int(peaks[0])          if len(peaks) > 0 else np.nan
    f["peak_last_pos"]    = int(peaks[-1])          if len(peaks) > 0 else np.nan
    f["peak_main_pos"]    = int(peaks[np.argmax(peak_heights)]) if len(peaks) > 0 else np.nan

    if len(peaks) > 0:
        widths = sig.peak_widths(x, peaks, rel_height=0.5)[0]
        f["peak_mean_width"] = np.mean(widths)
        f["peak_max_width"]  = np.max(widths)
    else:
        f["peak_mean_width"] = np.nan
        f["peak_max_width"]  = np.nan

    valleys, _ = sig.find_peaks(-x)
    f["valley_count"] = len(valleys)

    slopes = np.diff(x)
    f["slope_mean"]     = np.mean(slopes)
    f["slope_std"]      = np.std(slopes, ddof=1)
    f["slope_max"]      = np.max(slopes)
    f["slope_min"]      = np.min(slopes)
    f["slope_abs_mean"] = np.mean(np.abs(slopes))

    x_centered = x - f["mean"]
    zero_crossings = np.where(np.diff(np.sign(x_centered)))[0]
    f["zero_crossing_rate"] = len(zero_crossings) / N

    t = np.arange(N) / fs
    f["AUC"]            = np.trapz(x, t)
    f["AUC_normalized"] = f["AUC"] / (N / fs)

    high_idx = np.where(x > f["p75"])[0]
    f["first_high_pos"] = int(high_idx[0]) if len(high_idx) > 0 else np.nan

    fft_mags = np.abs(np.fft.rfft(x))
    freqs    = np.fft.rfftfreq(N, d=1.0 / fs)

    f["fft_mag_mean"]  = np.mean(fft_mags)
    f["fft_mag_max"]   = np.max(fft_mags)
    f["fft_mag_std"]   = np.std(fft_mags)
    f["dominant_freq"] = freqs[np.argmax(fft_mags)]

    sum_mags         = np.sum(fft_mags)
    total_fft_energy = np.sum(fft_mags ** 2)
    f["spectral_energy"]   = total_fft_energy
    f["spectral_centroid"] = np.sum(freqs * fft_mags) / sum_mags if sum_mags != 0 else np.nan
    f["spectral_spread"]   = np.sqrt(
        np.sum(((freqs - f["spectral_centroid"]) ** 2) * fft_mags) / sum_mags
    ) if sum_mags != 0 else np.nan

    cumulative_energy = np.cumsum(fft_mags ** 2)
    rolloff_idx = np.searchsorted(cumulative_energy, 0.85 * total_fft_energy)
    f["spectral_rolloff"] = freqs[min(rolloff_idx, len(freqs) - 1)]

    for i in range(4):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        band_mask = (freqs >= lo) & (freqs < hi)
        band_e = np.sum(fft_mags[band_mask] ** 2)
        f[f"band{i+1}_energy"]       = band_e
        f[f"band{i+1}_energy_ratio"] = band_e / total_fft_energy if total_fft_energy != 0 else np.nan

    coeffs = pywt.wavedec(x, wavelet="db4", level=3)
    cA, cD3, cD2, cD1 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

    f["wavelet_cA_energy"]  = np.sum(cA  ** 2)
    f["wavelet_cA_std"]     = np.std(cA)
    f["wavelet_cD3_energy"] = np.sum(cD3 ** 2)
    f["wavelet_cD3_std"]    = np.std(cD3)
    f["wavelet_cD2_energy"] = np.sum(cD2 ** 2)
    f["wavelet_cD2_std"]    = np.std(cD2)
    f["wavelet_cD1_energy"] = np.sum(cD1 ** 2)
    f["wavelet_cD1_std"]    = np.std(cD1)

    total_wavelet_energy = (
        f["wavelet_cA_energy"] + f["wavelet_cD3_energy"] +
        f["wavelet_cD2_energy"] + f["wavelet_cD1_energy"]
    )
    f["wavelet_energy_ratio_D1"] = (
        f["wavelet_cD1_energy"] / total_wavelet_energy
        if total_wavelet_energy != 0 else np.nan
    )

    return f


def main():
    print(f"읽는 중: {INPUT_PATH}")
    df = pd.read_excel(INPUT_PATH)
    aec_cols = [c for c in df.columns if c.startswith("aec_")]
    print(f"  환자 수: {len(df)}, AEC 열: {len(aec_cols)}개")

    feat_rows = []
    for _, row in df.iterrows():
        signal = np.array(row[aec_cols], dtype=float)
        feats  = extract_features(signal, fs=FS)
        feats["PatientID"] = row["PatientID"]
        feat_rows.append(feats)

    feat_cols = [c for _, c in enumerate([
        "signal_length", "mean", "std", "min", "max", "range", "median", "IQR",
        "skewness", "kurtosis", "p5", "p10", "p25", "p75", "p90", "p95",
        "p90_p10_ratio", "CV", "RMSE", "signal_energy", "mean_abs_deviation",
        "peak_count", "peak_max_height", "peak_mean_height", "peak_std_height",
        "peak_first_pos", "peak_last_pos", "peak_main_pos",
        "peak_mean_width", "peak_max_width", "valley_count",
        "slope_mean", "slope_std", "slope_max", "slope_min", "slope_abs_mean",
        "zero_crossing_rate", "AUC", "AUC_normalized", "first_high_pos",
        "fft_mag_mean", "fft_mag_max", "fft_mag_std", "dominant_freq",
        "spectral_centroid", "spectral_spread", "spectral_energy", "spectral_rolloff",
        "band1_energy", "band1_energy_ratio", "band2_energy", "band2_energy_ratio",
        "band3_energy", "band3_energy_ratio", "band4_energy", "band4_energy_ratio",
        "wavelet_cA_energy", "wavelet_cA_std", "wavelet_cD3_energy", "wavelet_cD3_std",
        "wavelet_cD2_energy", "wavelet_cD2_std", "wavelet_cD1_energy", "wavelet_cD1_std",
        "wavelet_energy_ratio_D1",
    ])]

    feat_df = pd.DataFrame(feat_rows)
    feat_df = feat_df[["PatientID"] + feat_cols]

    print(f"저장 중: {OUTPUT_PATH}")
    feat_df.to_excel(OUTPUT_PATH, index=False)
    print(f"완료: {len(feat_df)}행 × {len(feat_df.columns)}열")


if __name__ == "__main__":
    main()
