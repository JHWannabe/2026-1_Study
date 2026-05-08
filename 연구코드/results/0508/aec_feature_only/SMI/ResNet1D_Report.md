# ResNet1D Report — SMI

**Model:** ResNet1D_AECFeature_noScale

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 6.2170 | 0.1674 | 0.6727 | 0.6449 | 14 |
| 2 | 5.9501 | 0.1432 | 0.6955 | 0.5263 | 13 |
| 3 | 6.1794 | 0.1379 | 0.6136 | 0.5679 | 13 |
| 4 | 5.9820 | 0.1052 | 0.7123 | 0.6283 | 13 |
| 5 | 5.9708 | 0.1118 | 0.7123 | 0.5632 | 13 |
| **Mean** | **6.0599** | **0.1331** | **0.6813** | **0.5861** | **13.2** |
| **Std** | **0.1140** | **0.0225** | **0.0368** | **0.0440** | |

## Test Set 성능 (Test 20%)

Test R² = **0.1605**
Test MAE = **6.1585**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.4109 (p=1.261e-12) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.1129 |
| 이진화 ACC (성별 기준) | 0.6873 |
| 이진화 AUC (성별 기준) | 0.5574 |
| 이진화 AUPRC (성별 기준) | 0.7885 |
| 이진화 Brier Score (성별 기준) | 0.3916 |

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

