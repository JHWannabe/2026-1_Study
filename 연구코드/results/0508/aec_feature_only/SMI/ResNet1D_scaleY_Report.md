# ResNet1D Report — SMI

**Model:** ResNet1D_AECFeature_scaleY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 6.3060 | 0.1574 | 0.7000 | 0.6192 | 14 |
| 2 | 6.0601 | 0.0980 | 0.6500 | 0.4886 | 13 |
| 3 | 6.1638 | 0.1271 | 0.6364 | 0.5533 | 13 |
| 4 | 6.0455 | 0.1061 | 0.6667 | 0.6166 | 13 |
| 5 | 6.1232 | 0.1063 | 0.6986 | 0.5891 | 13 |
| **Mean** | **6.1397** | **0.1190** | **0.6703** | **0.5734** | **13.2** |
| **Std** | **0.0935** | **0.0215** | **0.0255** | **0.0486** | |

## Test Set 성능 (Test 20%)

Test R² = **0.0790**
Test MAE = **6.5673**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.3277 (p=2.643e-08) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.2532 |
| 이진화 ACC (성별 기준) | 0.7055 |
| 이진화 AUC (성별 기준) | 0.5341 |
| 이진화 AUPRC (성별 기준) | 0.7791 |
| 이진화 Brier Score (성별 기준) | 0.4336 |

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

