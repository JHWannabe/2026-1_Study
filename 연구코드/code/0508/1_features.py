import numpy as np
import pandas as pd
from scipy import stats, signal
import pywt

data_path = r'연구코드\data\강남_merged_features.xlsx'

merged = pd.read_excel(data_path, sheet_name='merged')

aec_cols = [c for c in merged.columns if c.startswith('aec_')]
meta_cols = [c for c in merged.columns if not c.startswith('aec_')]


def extract_features(sig):
    n = len(sig)
    f = {}

    # ── 기초 통계 ──────────────────────────────────────────────
    f['signal_length'] = n
    f['mean']   = np.mean(sig)
    f['std']    = np.std(sig, ddof=1)
    f['min']    = np.min(sig)
    f['max']    = np.max(sig)
    f['range']  = f['max'] - f['min']
    f['median'] = np.median(sig)
    f['IQR']    = np.percentile(sig, 75) - np.percentile(sig, 25)
    f['skewness'] = stats.skew(sig)
    f['kurtosis'] = stats.kurtosis(sig)

    # ── 백분위 ────────────────────────────────────────────────
    for p in [5, 10, 25, 75, 90, 95]:
        f[f'p{p}'] = np.percentile(sig, p)

    f['p90_p10_ratio'] = f['p90'] / f['p10'] if f['p10'] != 0 else np.nan
    f['CV']            = f['std'] / f['mean'] if f['mean'] != 0 else np.nan
    f['RMSE']          = np.sqrt(np.mean(sig ** 2))
    f['signal_energy'] = np.sum(sig ** 2)
    f['mean_abs_deviation'] = np.mean(np.abs(sig - f['mean']))

    # ── 피크 ─────────────────────────────────────────────────
    peaks, _ = signal.find_peaks(sig, height=0)
    if len(peaks) > 0:
        heights = sig[peaks]
        widths  = signal.peak_widths(sig, peaks, rel_height=0.5)[0]
        f['peak_count']       = len(peaks)
        f['peak_max_height']  = np.max(heights)
        f['peak_mean_height'] = np.mean(heights)
        f['peak_std_height']  = np.std(heights, ddof=1) if len(heights) > 1 else 0.0
        f['peak_first_pos']   = peaks[0] / (n - 1)
        f['peak_last_pos']    = peaks[-1] / (n - 1)
        f['peak_main_pos']    = peaks[np.argmax(heights)] / (n - 1)
        f['peak_mean_width']  = np.mean(widths)
        f['peak_max_width']   = np.max(widths)
    else:
        f['peak_count'] = 0
        for k in ['peak_max_height', 'peak_mean_height', 'peak_std_height',
                  'peak_first_pos', 'peak_last_pos', 'peak_main_pos',
                  'peak_mean_width', 'peak_max_width']:
            f[k] = np.nan

    # ── 골짜기 ────────────────────────────────────────────────
    valleys, _ = signal.find_peaks(-sig)
    f['valley_count'] = len(valleys)

    # ── 기울기 ────────────────────────────────────────────────
    slopes = np.diff(sig)
    f['slope_mean']     = np.mean(slopes)
    f['slope_std']      = np.std(slopes, ddof=1)
    f['slope_max']      = np.max(slopes)
    f['slope_min']      = np.min(slopes)
    f['slope_abs_mean'] = np.mean(np.abs(slopes))

    # ── 기타 시계열 ───────────────────────────────────────────
    centered = sig - f['mean']
    f['zero_crossing_rate'] = np.sum(np.diff(np.sign(centered)) != 0) / (n - 1)
    f['AUC']            = np.trapz(sig)
    f['AUC_normalized'] = f['AUC'] / n

    high_idx = np.where(sig > f['p75'])[0]
    f['first_high_pos'] = high_idx[0] / (n - 1) if len(high_idx) > 0 else np.nan

    # ── FFT ─────────────────────────────────────────────────
    fft_mag = np.abs(np.fft.rfft(sig))
    freqs   = np.fft.rfftfreq(n)

    f['fft_mag_mean'] = np.mean(fft_mag)
    f['fft_mag_max']  = np.max(fft_mag)
    f['fft_mag_std']  = np.std(fft_mag, ddof=1)
    f['dominant_freq'] = freqs[np.argmax(fft_mag)]

    # DC 성분 제외한 스펙트럼 통계
    mag  = fft_mag[1:]
    freq = freqs[1:]
    mag_sq     = mag ** 2
    total_sq   = np.sum(mag_sq)

    if total_sq > 0:
        f['spectral_centroid'] = np.sum(freq * mag_sq) / total_sq
        f['spectral_spread']   = np.sqrt(np.sum(((freq - f['spectral_centroid']) ** 2) * mag_sq) / total_sq)
        cumsum = np.cumsum(mag_sq)
        rolloff_idx = np.where(cumsum >= 0.85 * total_sq)[0]
        f['spectral_rolloff'] = freq[rolloff_idx[0]] if len(rolloff_idx) > 0 else np.nan
    else:
        f['spectral_centroid'] = np.nan
        f['spectral_spread']   = np.nan
        f['spectral_rolloff']  = np.nan

    f['spectral_energy'] = total_sq

    # ── 주파수 밴드 에너지 (4등분) ───────────────────────────
    n_fft      = len(fft_mag)
    band_size  = n_fft // 4
    total_e    = np.sum(fft_mag ** 2)
    for b in range(1, 5):
        start = (b - 1) * band_size
        end   = b * band_size if b < 4 else n_fft
        band_e = np.sum(fft_mag[start:end] ** 2)
        f[f'band{b}_energy']       = band_e
        f[f'band{b}_energy_ratio'] = band_e / total_e if total_e > 0 else np.nan

    # ── 웨이블릿 (db4, level=3) ──────────────────────────────
    cA, cD3, cD2, cD1 = pywt.wavedec(sig, 'db4', level=3)

    f['wavelet_cA_energy']  = np.sum(cA  ** 2)
    f['wavelet_cA_std']     = np.std(cA,  ddof=1)
    f['wavelet_cD3_energy'] = np.sum(cD3 ** 2)
    f['wavelet_cD3_std']    = np.std(cD3, ddof=1)
    f['wavelet_cD2_energy'] = np.sum(cD2 ** 2)
    f['wavelet_cD2_std']    = np.std(cD2, ddof=1)
    f['wavelet_cD1_energy'] = np.sum(cD1 ** 2)
    f['wavelet_cD1_std']    = np.std(cD1, ddof=1)

    total_wav = f['wavelet_cA_energy'] + f['wavelet_cD3_energy'] + f['wavelet_cD2_energy'] + f['wavelet_cD1_energy']
    f['wavelet_energy_ratio_D1'] = f['wavelet_cD1_energy'] / total_wav if total_wav > 0 else np.nan

    return f


print(f"merged 시트 로드: {len(merged)}행, AEC 칼럼 {len(aec_cols)}개")

rows = []
for _, row in merged.iterrows():
    sig  = row[aec_cols].values.astype(float)
    meta = {c: row[c] for c in meta_cols}
    feat = extract_features(sig)
    rows.append({**meta, **feat})

features_df = pd.DataFrame(rows)

with pd.ExcelWriter(data_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    features_df.to_excel(writer, sheet_name='features', index=False)

print(f"저장 완료: 'features' 시트 ({len(features_df)}행 x {len(features_df.columns)}열)")
print(f"  메타데이터 칼럼: {len(meta_cols)}개, 피처 칼럼: {len(feat)}개")
