# ResNet1D Report — TAMA

**Model:** ResNet1D_ClinicAECFeature_scaleXY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 13.4344 | 0.6044 | 0.7500 | 0.6175 | 16 |
| 2 | 12.5491 | 0.6483 | 0.7864 | 0.5760 | 14 |
| 3 | 12.7568 | 0.6352 | 0.7682 | 0.5435 | 16 |
| 4 | 14.3106 | 0.5772 | 0.7260 | 0.6530 | 16 |
| 5 | 13.3979 | 0.6295 | 0.7443 | 0.6537 | 16 |
| **Mean** | **13.2897** | **0.6189** | **0.7550** | **0.6087** | **15.6** |
| **Std** | **0.6176** | **0.0253** | **0.0207** | **0.0433** | |

## Test Set 성능 (Test 20%)

Test R² = **0.6652**
Test MAE = **13.6134**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.8176 (p=2.126e-67) |
| Shapiro-Wilk p | 0.0320 |
| Bias t-test p | 0.3720 |
| 이진화 ACC (성별 기준) | 0.7636 |
| 이진화 AUC (성별 기준) | 0.6388 |
| 이진화 AUPRC (성별 기준) | 0.8486 |
| 이진화 Brier Score (성별 기준) | 0.3534 |

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

