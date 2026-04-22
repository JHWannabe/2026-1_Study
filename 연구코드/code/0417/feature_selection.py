"""
feature_selection.py - AEC 특징 변수 상관분석 및 선택 가이드

목적:
  - 66개 AEC summary feature 각각과 TAMA의 Pearson/Spearman 상관계수 계산
  - VIF(분산팽창인수) 계산으로 다중공선성 확인
  - 결과를 results/feature_selection_report.xlsx 로 저장
  - 이 결과를 보고 config.py의 SELECTED_AEC_FEATURES 를 결정할 것

실행:
  python feature_selection.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

import config
import data_loader

os.makedirs(config.RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# AEC 특징 컬럼 목록 (PatientID 제외)
# ─────────────────────────────────────────────────────────────────────────────
AEC_FEATURE_COLS = [
    'signal_length', 'mean', 'std', 'min', 'max', 'range', 'median', 'IQR',
    'skewness', 'kurtosis', 'p5', 'p10', 'p25', 'p75', 'p90', 'p95',
    'p90_p10_ratio', 'CV', 'RMSE', 'signal_energy', 'mean_abs_deviation',
    'slope_mean', 'slope_std', 'slope_max', 'slope_min', 'slope_abs_mean',
    'zero_crossing_rate', 'peak_count', 'peak_max_height', 'peak_mean_height',
    'peak_std_height', 'peak_first_pos', 'peak_last_pos', 'peak_main_pos',
    'peak_mean_width', 'peak_max_width', 'valley_count',
    'AUC', 'AUC_normalized', 'first_high_pos',
    'fft_mag_mean', 'fft_mag_max', 'fft_mag_std', 'dominant_freq',
    'spectral_centroid', 'spectral_spread', 'spectral_energy', 'spectral_rolloff',
    'band1_energy', 'band1_energy_ratio', 'band2_energy', 'band2_energy_ratio',
    'band3_energy', 'band3_energy_ratio', 'band4_energy', 'band4_energy_ratio',
    'wavelet_cA_energy', 'wavelet_cA_std', 'wavelet_cD3_energy', 'wavelet_cD3_std',
    'wavelet_cD2_energy', 'wavelet_cD2_std', 'wavelet_cD1_energy', 'wavelet_cD1_std',
    'wavelet_energy_ratio_D1',
]


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """각 AEC feature와 TAMA의 Pearson / Spearman 상관계수 계산."""
    tama = df['TAMA'].values
    records = []

    for col in AEC_FEATURE_COLS:
        if col not in df.columns:
            continue
        x = df[col].dropna()
        common = df[[col, 'TAMA']].dropna()
        if len(common) < 10:
            continue

        x_vals = common[col].values
        y_vals = common['TAMA'].values

        pearson_r, pearson_p   = stats.pearsonr(x_vals, y_vals)
        spearman_r, spearman_p = stats.spearmanr(x_vals, y_vals)

        records.append({
            'Feature':    col,
            'Pearson_r':  f"{pearson_r:.4e}",
            'Pearson_p':  f"{pearson_p:.4e}",
            'Spearman_r': f"{spearman_r:.4e}",
            'Spearman_p': f"{spearman_p:.4e}",
            'N':          len(common),
        })

    corr_df = pd.DataFrame(records)
    # 정렬용 절대값 컬럼 (문자열 → float 변환)
    corr_df['|Pearson_r|'] = corr_df['Pearson_r'].apply(lambda x: abs(float(x)))
    corr_df = corr_df.sort_values('|Pearson_r|', ascending=False).reset_index(drop=True)
    corr_df.insert(0, 'Rank', corr_df.index + 1)
    return corr_df


def compute_vif(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    선택 feature들 간 VIF 계산 (Age, Sex 포함한 전체 예측변수 공간 기준).
    VIF < 5: 낮음, 5–10: 중간, >10: 높음 (다중공선성 주의)
    """
    subset = df[features].dropna()
    # 상수항 추가
    X = np.column_stack([np.ones(len(subset)), subset.values])
    col_names = ['const'] + features

    records = []
    for i, name in enumerate(col_names):
        if name == 'const':
            continue
        vif = variance_inflation_factor(X, i)
        records.append({'Feature': name, 'VIF': round(vif, 3)})

    return pd.DataFrame(records).sort_values('VIF', ascending=False)


def tama_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """성별별 TAMA 기술통계 (logistic regression 임계값 설정 참고용)."""
    rows = []
    for sex in ['M', 'F']:
        sub = df[df['PatientSex'] == sex]['TAMA']
        rows.append({
            'Sex':    sex,
            'N':      len(sub),
            'Mean':   round(sub.mean(), 2),
            'SD':     round(sub.std(),  2),
            'Min':    round(sub.min(),  2),
            'P25':    round(sub.quantile(0.25), 2),
            'Median': round(sub.median(), 2),
            'P75':    round(sub.quantile(0.75), 2),
            'Max':    round(sub.max(),  2),
            'P10':    round(sub.quantile(0.10), 2),
            'P33':    round(sub.quantile(0.33), 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Feature Selection - AEC 상관분석")
    print("=" * 60)

    # 데이터 로드 (전처리 없이 원시 데이터 사용)
    df = data_loader.load_raw_data()

    # 1. 상관계수 계산
    print("\n[1] TAMA와 AEC feature 상관계수 계산 중...")
    corr_df = compute_correlations(df)
    print(f"  완료: {len(corr_df)}개 feature 분석")
    print("\n  상위 10개 feature (|Pearson r| 기준):")
    print(corr_df[['Rank', 'Feature', 'Pearson_r', 'Pearson_p',
                    'Spearman_r', 'Spearman_p']].head(10).to_string(index=False))

    # 2. 현재 SELECTED_AEC_FEATURES VIF 계산
    print(f"\n[2] 현재 SELECTED_AEC_FEATURES VIF 계산: {config.SELECTED_AEC_FEATURES}")
    vif_cols = config.SELECTED_AEC_FEATURES
    available = [c for c in vif_cols if c in df.columns]
    if len(available) >= 2:
        # Age, Sex 포함한 전체 예측변수에서 VIF 계산
        vif_input_cols = ['PatientAge'] + available
        df_vif = df[vif_input_cols].dropna()
        # 성별 인코딩
        df_enc = data_loader.encode_sex(df.copy())
        vif_input_cols2 = ['PatientAge', 'Sex'] + available
        df_vif2 = df_enc[vif_input_cols2].dropna()
        vif_df = compute_vif(df_vif2, vif_input_cols2)
        print(vif_df.to_string(index=False))
        vif_alert = vif_df[vif_df['VIF'] > 10]
        if not vif_alert.empty:
            print(f"\n  ⚠ VIF > 10인 feature (다중공선성 주의): {vif_alert['Feature'].tolist()}")
    else:
        vif_df = pd.DataFrame()
        print("  VIF 계산 생략 (feature 수 부족)")

    # 3. 성별 TAMA 분포 (임계값 설정 참고)
    print("\n[3] 성별 TAMA 분포 (임계값 설정 참고):")
    tama_dist = tama_distribution_summary(df)
    print(tama_dist.to_string(index=False))
    print(f"\n  현재 설정된 임계값: M < {config.TAMA_THRESHOLD_MALE} cm², "
          f"F < {config.TAMA_THRESHOLD_FEMALE} cm²")
    # 현재 임계값 적용 시 양성 비율 미리보기
    df_enc2 = data_loader.encode_sex(df.copy())
    df_bin  = data_loader.add_tama_binary(df_enc2)

    # 4. Excel 저장
    out_path = os.path.join(config.RESULTS_DIR, 'feature_selection_report.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        corr_df.to_excel(writer, sheet_name='상관계수_전체', index=False)
        corr_df.head(20).to_excel(writer, sheet_name='상관계수_Top20', index=False)
        if not vif_df.empty:
            vif_df.to_excel(writer, sheet_name='VIF_선택feature', index=False)
        tama_dist.to_excel(writer, sheet_name='TAMA_분포_성별', index=False)

    print(f"\n[저장] {out_path}")
    print("\n" + "=" * 60)
    print("  ※ 다음 단계: 상관계수 Top 결과를 확인하고")
    print("     config.py 의 SELECTED_AEC_FEATURES 를 업데이트하세요.")
    print("=" * 60)


if __name__ == '__main__':
    main()
