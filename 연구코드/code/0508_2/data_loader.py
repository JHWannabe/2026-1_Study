"""
공유 데이터 로더
- 강남 train (aec_interpolation_final, aec_feature_filtered)
- 신촌 external validation (추후 추가)
- Low SMI label: sex-stratified 25th percentile
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'data'))

GANGNAM_FILE  = os.path.join(_DATA_DIR, '강남_최종_정리본.xlsx')
SHINCHON_FILE = os.path.join(_DATA_DIR, '신촌_merged_features.xlsx')

AEC_COLS  = [f'aec{i}' for i in range(1, 257)]          # 강남: aec1~aec256
CLIN_COLS = ['PatientAge', 'PatientSex', 'BMI']

# 강남 feature_filtered 에 있는 handcrafted AEC feature 목록
HC_COLS = [
    'mean', 'std', 'min', 'max', 'range', 'median', 'IQR',
    'skewness', 'kurtosis', 'p5', 'p10', 'p25', 'p75', 'p90', 'p95',
    'p90_p10_ratio', 'CV', 'RMSE', 'signal_energy', 'mean_abs_deviation',
    'peak_count', 'peak_max_height', 'peak_mean_height', 'peak_std_height',
    'peak_first_pos', 'peak_last_pos', 'peak_main_pos',
    'peak_mean_width', 'peak_max_width', 'valley_count',
    'slope_mean', 'slope_std', 'slope_max', 'slope_min', 'slope_abs_mean',
    'zero_crossing_rate', 'AUC', 'AUC_normalized', 'first_high_pos',
    'fft_mag_mean', 'fft_mag_max', 'fft_mag_std', 'dominant_freq',
    'spectral_energy', 'spectral_centroid', 'spectral_spread', 'spectral_rolloff',
    'band1_energy', 'band1_energy_ratio', 'band2_energy', 'band2_energy_ratio',
    'band3_energy', 'band3_energy_ratio', 'band4_energy', 'band4_energy_ratio',
    'wavelet_cA_energy', 'wavelet_cA_std',
    'wavelet_cD3_energy', 'wavelet_cD3_std',
    'wavelet_cD2_energy', 'wavelet_cD2_std',
    'wavelet_cD1_energy', 'wavelet_cD1_std',
    'wavelet_energy_ratio_D1',
]

SEED = 42


def _encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df['PatientSex'].dtype == object:
        df['PatientSex'] = df['PatientSex'].map({'M': 0, 'F': 1})
    return df


def _add_binary_label(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()
    df[f'{target}_bin'] = df.groupby('PatientSex')[target].transform(
        lambda x: (x <= x.quantile(0.25)).astype(int)
    )
    return df


def _print_label_stats(df: pd.DataFrame, target: str, name: str = ''):
    tag = f'[{name}] ' if name else ''
    print(f'{tag}n={len(df)}, Low-{target}: {df[f"{target}_bin"].sum()} '
          f'({df[f"{target}_bin"].mean()*100:.1f}%)')
    for code, label in [(0, 'Male'), (1, 'Female')]:
        thr = df[df['PatientSex'] == code][target].quantile(0.25)
        n   = int((df[df['PatientSex'] == code][f'{target}_bin'] == 1).sum())
        print(f'  {label}: threshold={thr:.4f}, n_low={n}')


# ── 강남 (AEC interp + clinical) ──────────────────────────
def load_gangnam(target: str = 'SMI') -> pd.DataFrame:
    """강남 데이터: aec_interpolation_final 시트 (AEC 256pt + clinical)"""
    df = pd.read_excel(GANGNAM_FILE, sheet_name='aec_interpolation_final')
    df = _encode_sex(df)
    df = df.dropna(subset=AEC_COLS + CLIN_COLS + [target]).reset_index(drop=True)
    df = _add_binary_label(df, target)
    _print_label_stats(df, target, '강남 AEC')
    return df


# ── 강남 (handcrafted features) ───────────────────────────
def load_gangnam_handcrafted(target: str = 'SMI') -> pd.DataFrame:
    """강남 데이터: aec_feature_filtered 시트 (handcrafted AEC + clinical)"""
    df = pd.read_excel(GANGNAM_FILE, sheet_name='aec_feature_filtered')
    df = _encode_sex(df)
    need = CLIN_COLS + HC_COLS + [target]
    df = df.dropna(subset=need).reset_index(drop=True)
    df = _add_binary_label(df, target)
    _print_label_stats(df, target, '강남 HC')
    return df


# ── Train/Test split ──────────────────────────────────────
def train_test_idx(df: pd.DataFrame, target: str = 'SMI', test_size: float = 0.2):
    """Stratified 80/20 split, 강남 내부 검증용"""
    y   = df[f'{target}_bin'].values
    idx = np.arange(len(df))
    return train_test_split(idx, test_size=test_size, random_state=SEED, stratify=y)
