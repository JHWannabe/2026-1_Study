# ResNet1D Report — SMI

**Model:** ResNet1D_AECFeature_scaleX

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 6.0824 | 0.1534 | 0.7045 | 0.6106 | 14 |
| 2 | 5.8715 | 0.1675 | 0.7318 | 0.5751 | 13 |
| 3 | 5.6780 | 0.2226 | 0.7136 | 0.6146 | 13 |
| 4 | 6.0043 | 0.0886 | 0.6667 | 0.6246 | 13 |
| 5 | 5.7487 | 0.1664 | 0.7260 | 0.5354 | 13 |
| **Mean** | **5.8770** | **0.1597** | **0.7085** | **0.5921** | **13.2** |
| **Std** | **0.1513** | **0.0428** | **0.0230** | **0.0329** | |

## Test Set 성능 (Test 20%)

Test R² = **0.1743**
Test MAE = **6.2172**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.4205 (p=3.313e-13) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.3647 |
| 이진화 ACC (성별 기준) | 0.7018 |
| 이진화 AUC (성별 기준) | 0.5893 |
| 이진화 AUPRC (성별 기준) | 0.7972 |
| 이진화 Brier Score (성별 기준) | 0.3914 |

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

