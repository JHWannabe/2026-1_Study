# ResNet1D Report — SMI

**Model:** ResNet1D_ClinicAECFeature_scaleXY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 4.9596 | 0.4156 | 0.7409 | 0.6989 | 16 |
| 2 | 4.8549 | 0.4098 | 0.7545 | 0.6436 | 14 |
| 3 | 4.4745 | 0.4783 | 0.7364 | 0.6667 | 16 |
| 4 | 4.6652 | 0.4054 | 0.7352 | 0.6371 | 16 |
| 5 | 5.2242 | 0.3428 | 0.7397 | 0.6034 | 16 |
| **Mean** | **4.8357** | **0.4104** | **0.7413** | **0.6500** | **15.6** |
| **Std** | **0.2554** | **0.0430** | **0.0069** | **0.0318** | |

## Test Set 성능 (Test 20%)

Test R² = **0.3967**
Test MAE = **5.2002**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.6374 (p=9.271e-33) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.2525 |
| 이진화 ACC (성별 기준) | 0.7818 |
| 이진화 AUC (성별 기준) | 0.6393 |
| 이진화 AUPRC (성별 기준) | 0.8441 |
| 이진화 Brier Score (성별 기준) | 0.3076 |

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

