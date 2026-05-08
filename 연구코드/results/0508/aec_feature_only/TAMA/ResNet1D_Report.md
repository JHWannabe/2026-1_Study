# ResNet1D Report — TAMA

**Model:** ResNet1D_AECFeature

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 19.6177 | 0.2763 | 0.6773 | 0.6277 | 14 |
| 2 | 17.7422 | 0.2389 | 0.7409 | 0.5624 | 13 |
| 3 | 17.7557 | 0.2666 | 0.7227 | 0.6044 | 13 |
| 4 | 19.8498 | 0.1705 | 0.6438 | 0.6691 | 13 |
| 5 | 17.5210 | 0.3143 | 0.7123 | 0.6008 | 13 |
| **Mean** | **18.4973** | **0.2533** | **0.6994** | **0.6129** | **13.2** |
| **Std** | **1.0157** | **0.0479** | **0.0347** | **0.0351** | |

## Test Set 성능 (Test 20%)

Test R² = **0.2547**
Test MAE = **19.5603**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.5146 (p=5.422e-20) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.8322 |
| 이진화 ACC (성별 기준) | 0.6909 |
| 이진화 AUC (성별 기준) | 0.5751 |

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

