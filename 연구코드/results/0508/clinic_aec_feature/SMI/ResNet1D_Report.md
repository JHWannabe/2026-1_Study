# ResNet1D Report — SMI

**Model:** ResNet1D_ClinicAECFeature_noScale

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 5.0988 | 0.4006 | 0.7682 | 0.7521 | 16 |
| 2 | 4.8629 | 0.4005 | 0.7364 | 0.6574 | 14 |
| 3 | 4.6144 | 0.4522 | 0.7409 | 0.7247 | 16 |
| 4 | 4.9991 | 0.3068 | 0.7169 | 0.6890 | 16 |
| 5 | 5.2097 | 0.2923 | 0.7215 | 0.5952 | 16 |
| **Mean** | **4.9570** | **0.3705** | **0.7368** | **0.6837** | **15.6** |
| **Std** | **0.2059** | **0.0611** | **0.0181** | **0.0546** | |

## Test Set 성능 (Test 20%)

Test R² = **0.2995**
Test MAE = **5.6656**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.5578 (p=6.945e-24) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.2324 |
| 이진화 ACC (성별 기준) | 0.7382 |
| 이진화 AUC (성별 기준) | 0.7283 |
| 이진화 AUPRC (성별 기준) | 0.8786 |
| 이진화 Brier Score (성별 기준) | 0.2562 |

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

