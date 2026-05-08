# ResNet1D Report — SMI

**Model:** ResNet1D_ClinicAECFeature_scaleY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 5.2126 | 0.3521 | 0.7682 | 0.6911 | 16 |
| 2 | 5.0486 | 0.3462 | 0.6955 | 0.6588 | 14 |
| 3 | 4.4243 | 0.5299 | 0.7636 | 0.7310 | 16 |
| 4 | 4.7688 | 0.3534 | 0.6941 | 0.6544 | 16 |
| 5 | 5.0655 | 0.3317 | 0.7580 | 0.6060 | 16 |
| **Mean** | **4.9040** | **0.3827** | **0.7359** | **0.6683** | **15.6** |
| **Std** | **0.2795** | **0.0740** | **0.0337** | **0.0415** | |

## Test Set 성능 (Test 20%)

Test R² = **0.2608**
Test MAE = **5.6503**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.5248 (p=7.294e-21) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.0340 |
| 이진화 ACC (성별 기준) | 0.7164 |
| 이진화 AUC (성별 기준) | 0.7019 |
| 이진화 AUPRC (성별 기준) | 0.8702 |
| 이진화 Brier Score (성별 기준) | 0.4381 |

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

