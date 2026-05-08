# ResNet1D Report — SMI

**Model:** ResNet1D_AECFeature_scaleXY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 6.0980 | 0.1711 | 0.6727 | 0.6154 | 14 |
| 2 | 5.6966 | 0.1374 | 0.7091 | 0.5720 | 13 |
| 3 | 5.8182 | 0.2126 | 0.6682 | 0.6349 | 13 |
| 4 | 5.7886 | 0.1344 | 0.6895 | 0.6273 | 13 |
| 5 | 5.6913 | 0.1956 | 0.7489 | 0.5616 | 13 |
| **Mean** | **5.8185** | **0.1702** | **0.6977** | **0.6023** | **13.2** |
| **Std** | **0.1484** | **0.0310** | **0.0294** | **0.0298** | |

## Test Set 성능 (Test 20%)

Test R² = **0.2084**
Test MAE = **6.0770**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.4593 (p=9.295e-16) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.4007 |
| 이진화 ACC (성별 기준) | 0.7018 |
| 이진화 AUC (성별 기준) | 0.6015 |
| 이진화 AUPRC (성별 기준) | 0.8215 |
| 이진화 Brier Score (성별 기준) | 0.2964 |

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

