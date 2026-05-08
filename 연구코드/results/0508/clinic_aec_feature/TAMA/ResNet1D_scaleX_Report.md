# ResNet1D Report — TAMA

**Model:** ResNet1D_ClinicAECFeature_scaleX

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 13.1073 | 0.6304 | 0.7682 | 0.6425 | 16 |
| 2 | 13.3043 | 0.5995 | 0.7682 | 0.5608 | 14 |
| 3 | 11.5864 | 0.7068 | 0.7955 | 0.6106 | 16 |
| 4 | 13.7312 | 0.5461 | 0.7763 | 0.6771 | 16 |
| 5 | 13.1466 | 0.6220 | 0.7580 | 0.6553 | 16 |
| **Mean** | **12.9752** | **0.6210** | **0.7732** | **0.6293** | **15.6** |
| **Std** | **0.7287** | **0.0520** | **0.0125** | **0.0405** | |

## Test Set 성능 (Test 20%)

Test R² = **0.6688**
Test MAE = **13.6120**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.8246 (p=1.692e-69) |
| Shapiro-Wilk p | 0.0067 |
| Bias t-test p | 0.0185 |
| 이진화 ACC (성별 기준) | 0.7709 |
| 이진화 AUC (성별 기준) | 0.6045 |
| 이진화 AUPRC (성별 기준) | 0.8406 |
| 이진화 Brier Score (성별 기준) | 0.3896 |

## 피처 선택 목록 (Fold별)

### Fold 1 (16개)

PatientSex, PatientAge, BMI, skewness, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, peak_max_width, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

### Fold 2 (14개)

PatientSex, PatientAge, BMI, skewness, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, slope_mean, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

### Fold 3 (16개)

PatientSex, PatientAge, BMI, skewness, peak_count, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

### Fold 4 (16개)

PatientSex, PatientAge, BMI, skewness, peak_count, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

### Fold 5 (16개)

PatientSex, PatientAge, BMI, skewness, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, peak_max_width, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

### 최종 모델 (Train 전체, 16개)

PatientSex, PatientAge, BMI, skewness, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, peak_max_width, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

