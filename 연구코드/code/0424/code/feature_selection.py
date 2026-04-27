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

import config as config
import data_loader as data_loader

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


def compute_correlation_matrix(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    선택 feature들 간 Pearson 상관행렬 계산.
    |r| > 0.8 이면 다중공선성 위험 신호.
    """
    available = [f for f in features if f in df.columns]
    subset = df[available].dropna()
    return subset.corr(method='pearson').round(4)


def scanner_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """CT 스캐너 모델별 환자 수 및 비율 집계."""
    counts = df['ManufacturerModelName'].value_counts()
    pct    = counts / counts.sum() * 100
    return pd.DataFrame({
        'Model': counts.index,
        'N':     counts.values,
        'Pct':   pct.round(1).values,
    }).reset_index(drop=True)


def kvp_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """kVp 값별 환자 수 및 비율 집계. KVP / kvp / kVp 컬럼 자동 탐지."""
    col = next((c for c in ['KVP', 'kvp', 'kVp'] if c in df.columns), None)
    if col is None:
        return pd.DataFrame()
    counts = df[col].value_counts().sort_index()
    pct    = counts / counts.sum() * 100
    return pd.DataFrame({
        'kVp': counts.index,
        'N':   counts.values,
        'Pct': pct.round(1).values,
    }).reset_index(drop=True)


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
    df, _ = data_loader.load_raw_data()

    # 1. 상관계수 계산
    print("\n[1] TAMA와 AEC feature 상관계수 계산 중...")
    corr_df = compute_correlations(df)
    print(f"  완료: {len(corr_df)}개 feature 분석")
    print("\n  상위 10개 feature (|Pearson r| 기준):")
    print(corr_df[['Rank', 'Feature', 'Pearson_r', 'Pearson_p',
                    'Spearman_r', 'Spearman_p']].head(10).to_string(index=False))

    # 2. VIF 계산 — 현재 선택, mean/AUC_normalized 부분 제거 비교
    print(f"\n[2] VIF 계산 (현재 + 부분 제거 비교): {config.SELECTED_AEC_FEATURES}")
    df_enc = data_loader.encode_sex(df.copy())
    base_demo = ['PatientAge', 'Sex']

    base_aec = config.SELECTED_AEC_FEATURES
    available = [c for c in base_aec if c in df.columns]

    # mean / AUC_normalized 부분 제거 비교 집합 구성
    without_mean     = [f for f in available if f != 'mean'] + (['AUC_normalized'] if 'AUC_normalized' in df.columns else [])
    without_auc_norm = [f for f in available if f != 'AUC_normalized'] + (['mean'] if 'mean' in df.columns else [])
    both_included    = available + [f for f in ['mean', 'AUC_normalized'] if f in df.columns and f not in available]

    vif_scenarios = [
        ('현재 선택 (SELECTED_AEC_FEATURES)', available),
        ('mean 제외, AUC_normalized 포함',    without_mean),
        ('AUC_normalized 제외, mean 포함',    without_auc_norm),
        ('mean + AUC_normalized 둘 다 포함',  both_included),
    ]

    vif_df = pd.DataFrame()
    vif_partial_rows = []
    for label, feats in vif_scenarios:
        feats_avail = [f for f in feats if f in df_enc.columns]
        if len(feats_avail) < 2:
            continue
        cols = base_demo + feats_avail
        vdf  = compute_vif(df_enc[cols].dropna(), cols)
        print(f"\n  VIF - {label}:")
        print(vdf.to_string(index=False))
        alert = vdf[vdf['VIF'] > 10]
        if not alert.empty:
            print(f"  [!] VIF > 10: {alert['Feature'].tolist()}")
        vdf['Scenario'] = label
        vif_partial_rows.append(vdf)
        if label.startswith('현재'):
            vif_df = vdf

    vif_partial_df = pd.concat(vif_partial_rows, ignore_index=True) if vif_partial_rows else pd.DataFrame()

    # 3. 선택 feature 간 Correlation Matrix (다중공선성 확인)
    print(f"\n[3] 선택 feature 간 Correlation Matrix (|r| > 0.8 주의):")
    all_model_feats = base_demo + [f for f in available if f in df_enc.columns]
    corr_matrix = compute_correlation_matrix(df_enc, all_model_feats)
    print(corr_matrix.to_string())
    # 쌍별 고상관 feature 출력
    high_corr_pairs = []
    cols_cm = corr_matrix.columns.tolist()
    for i in range(len(cols_cm)):
        for j in range(i + 1, len(cols_cm)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.8:
                high_corr_pairs.append({
                    'Feature_A': cols_cm[i],
                    'Feature_B': cols_cm[j],
                    'Pearson_r': round(r, 4),
                })
    high_corr_df = (pd.DataFrame(high_corr_pairs)
                    .sort_values('Pearson_r', key=abs, ascending=False)
                    if high_corr_pairs else pd.DataFrame(columns=['Feature_A', 'Feature_B', 'Pearson_r']))
    if not high_corr_df.empty:
        print("\n  [!] |r| > 0.8 쌍 (다중공선성 주의):")
        print(high_corr_df.to_string(index=False))
    else:
        print("  -> |r| > 0.8 쌍 없음")

    # 4. 성별 TAMA 분포 (임계값 설정 참고)
    print("\n[4] 성별 TAMA 분포 (임계값 설정 참고):")
    tama_dist = tama_distribution_summary(df)
    print(tama_dist.to_string(index=False))
    print(f"\n  현재 설정된 임계값: M < {config.TAMA_THRESHOLD_MALE} cm², "
          f"F < {config.TAMA_THRESHOLD_FEMALE} cm²")
    df_bin = data_loader.add_tama_binary(df_enc.copy())

    # 5. CT 스캐너 분포
    print("\n[5] CT 스캐너 분포:")
    scanner_df = scanner_distribution(df)
    print(scanner_df.head(10).to_string(index=False))
    print(f"  총 {len(scanner_df)}종 스캐너, 상위 스캐너: {scanner_df.iloc[0]['Model']} "
          f"({scanner_df.iloc[0]['N']}명, {scanner_df.iloc[0]['Pct']}%)")

    # 6. kVp 분포
    print("\n[6] kVp 분포:")
    kvp_df = kvp_distribution(df)
    if not kvp_df.empty:
        print(kvp_df.to_string(index=False))
        dominant = kvp_df.loc[kvp_df['N'].idxmax()]
        print(f"  주요 kVp: {dominant['kVp']} ({dominant['N']}명, {dominant['Pct']}%)")
    else:
        print("  kVp 컬럼 없음 (KVP / kvp)")

    # 7. Excel 저장
    out_path = os.path.join(config.RESULTS_DIR, 'feature_selection_report.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        corr_df.to_excel(writer, sheet_name='상관계수_전체', index=False)
        corr_df.head(20).to_excel(writer, sheet_name='상관계수_Top20', index=False)
        if not vif_df.empty:
            vif_df.to_excel(writer, sheet_name='VIF_선택feature', index=False)
        if not vif_partial_df.empty:
            vif_partial_df.to_excel(writer, sheet_name='VIF_부분제거_비교', index=False)
        corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
        if not high_corr_df.empty:
            high_corr_df.to_excel(writer, sheet_name='고상관쌍_r0.8이상', index=False)
        tama_dist.to_excel(writer, sheet_name='TAMA_분포_성별', index=False)
        scanner_df.to_excel(writer, sheet_name='CT_스캐너_분포', index=False)
        if not kvp_df.empty:
            kvp_df.to_excel(writer, sheet_name='kVp_분포', index=False)

    print(f"\n[저장] {out_path}")
    print("\n" + "=" * 60)
    print("  ※ 다음 단계: 상관계수 Top 결과를 확인하고")
    print("     config.py 의 SELECTED_AEC_FEATURES 를 업데이트하세요.")
    print("=" * 60)


if __name__ == '__main__':
    main()
