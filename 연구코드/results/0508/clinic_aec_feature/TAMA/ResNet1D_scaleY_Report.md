# ResNet1D Report — TAMA

**Model:** ResNet1D_ClinicAECFeature_scaleY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 19.4528 | 0.3570 | 0.6591 | 0.7236 | 16 |
| 2 | 19.0391 | 0.1809 | 0.6818 | 0.6331 | 14 |
| 3 | 13.2888 | 0.5724 | 0.7500 | 0.5518 | 16 |
| 4 | 21.2484 | 0.1882 | 0.6073 | 0.7368 | 16 |
| 5 | 18.6118 | 0.2445 | 0.6484 | 0.5880 | 16 |
| **Mean** | **18.3282** | **0.3086** | **0.6693** | **0.6467** | **15.6** |
| **Std** | **2.6748** | **0.1462** | **0.0470** | **0.0730** | |

## Test Set 성능 (Test 20%)

Test R² = **0.3263**
Test MAE = **18.9000**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.5742 (p=1.621e-25) |
| Shapiro-Wilk p | 0.0009 |
| Bias t-test p | 0.2622 |
| 이진화 ACC (성별 기준) | 0.6909 |
| 이진화 AUC (성별 기준) | 0.5890 |
| 이진화 AUPRC (성별 기준) | 0.8341 |
| 이진화 Brier Score (성별 기준) | 0.4580 |

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

