# ResNet1D Report — TAMA

**Model:** ResNet1D_AECFeature_scaleY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 20.6364 | 0.2334 | 0.6364 | 0.6169 | 14 |
| 2 | 19.0971 | 0.1663 | 0.6500 | 0.4929 | 13 |
| 3 | 18.9943 | 0.1305 | 0.6727 | 0.5047 | 13 |
| 4 | 21.9180 | 0.1189 | 0.6210 | 0.6607 | 13 |
| 5 | 19.8490 | 0.1936 | 0.6758 | 0.5946 | 13 |
| **Mean** | **20.0990** | **0.1686** | **0.6512** | **0.5739** | **13.2** |
| **Std** | **1.0848** | **0.0418** | **0.0210** | **0.0650** | |

## Test Set 성능 (Test 20%)

Test R² = **0.1557**
Test MAE = **20.4614**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.4193 (p=3.894e-13) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.0183 |
| 이진화 ACC (성별 기준) | 0.6655 |
| 이진화 AUC (성별 기준) | 0.5162 |
| 이진화 AUPRC (성별 기준) | 0.7789 |
| 이진화 Brier Score (성별 기준) | 0.4683 |

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

