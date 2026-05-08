# ResNet1D Report — SMI

**Model:** ResNet1D_ClinicAECFeature_scaleX

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 4.8803 | 0.4177 | 0.7455 | 0.6734 | 16 |
| 2 | 4.7092 | 0.4097 | 0.8182 | 0.6712 | 14 |
| 3 | 4.7817 | 0.4497 | 0.7182 | 0.6581 | 16 |
| 4 | 4.5139 | 0.4187 | 0.7671 | 0.6537 | 16 |
| 5 | 5.0085 | 0.3878 | 0.7489 | 0.6718 | 16 |
| **Mean** | **4.7787** | **0.4167** | **0.7596** | **0.6656** | **15.6** |
| **Std** | **0.1662** | **0.0199** | **0.0332** | **0.0081** | |

## Test Set 성능 (Test 20%)

Test R² = **0.4223**
Test MAE = **4.9157**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.6652 (p=1.640e-36) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.0040 |
| 이진화 ACC (성별 기준) | 0.7782 |
| 이진화 AUC (성별 기준) | 0.6817 |
| 이진화 AUPRC (성별 기준) | 0.8570 |
| 이진화 Brier Score (성별 기준) | 0.4117 |

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

