# ResNet1D Report — TAMA

**Model:** ResNet1D_AECFeature_scaleX

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 20.0422 | 0.2741 | 0.6682 | 0.5958 | 14 |
| 2 | 18.2886 | 0.2591 | 0.7182 | 0.5488 | 13 |
| 3 | 17.5221 | 0.2221 | 0.7045 | 0.5861 | 13 |
| 4 | 20.2415 | 0.1889 | 0.6986 | 0.7026 | 13 |
| 5 | 16.9944 | 0.3192 | 0.6986 | 0.5834 | 13 |
| **Mean** | **18.6178** | **0.2527** | **0.6976** | **0.6033** | **13.2** |
| **Std** | **1.3122** | **0.0446** | **0.0164** | **0.0521** | |

## Test Set 성능 (Test 20%)

Test R² = **0.2435**
Test MAE = **19.4757**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.5181 (p=2.705e-20) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.3248 |
| 이진화 ACC (성별 기준) | 0.7018 |
| 이진화 AUC (성별 기준) | 0.5727 |
| 이진화 AUPRC (성별 기준) | 0.7983 |
| 이진화 Brier Score (성별 기준) | 0.4232 |

## 피처 선택 목록 (Fold별)

### Fold 1 (14개)

skewness, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, peak_max_width, valley_count, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

### Fold 2 (13개)

skewness, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, peak_max_width, valley_count, slope_mean, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

### Fold 3 (13개)

skewness, peak_count, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

### Fold 4 (13개)

skewness, peak_count, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

### Fold 5 (13개)

skewness, peak_count, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

### 최종 모델 (Train 전체, 14개)

skewness, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, peak_max_width, valley_count, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

