# ResNet1D Report — TAMA

**Model:** ResNet1D_ClinicAECFeature

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 14.1029 | 0.6124 | 0.7091 | 0.6026 | 16 |
| 2 | 13.0111 | 0.6173 | 0.7955 | 0.5776 | 14 |
| 3 | 12.7488 | 0.6561 | 0.7955 | 0.5799 | 16 |
| 4 | 13.7698 | 0.5315 | 0.7352 | 0.6405 | 16 |
| 5 | 13.4480 | 0.6170 | 0.7443 | 0.6296 | 16 |
| **Mean** | **13.4161** | **0.6069** | **0.7559** | **0.6060** | **15.6** |
| **Std** | **0.4913** | **0.0409** | **0.0343** | **0.0255** | |

## Test Set 성능 (Test 20%)

Test R² = **0.6662**
Test MAE = **13.8909**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.8166 (p=4.178e-67) |
| Shapiro-Wilk p | 0.0525 |
| Bias t-test p | 0.9380 |
| 이진화 ACC (성별 기준) | 0.7709 |
| 이진화 AUC (성별 기준) | 0.6293 |

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

