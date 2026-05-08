# ResNet1D Report — TAMA

**Model:** ResNet1D_AECFeature_scaleXY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 19.4458 | 0.2773 | 0.7091 | 0.6243 | 14 |
| 2 | 17.4711 | 0.2582 | 0.7500 | 0.5354 | 13 |
| 3 | 17.7290 | 0.2384 | 0.7091 | 0.5458 | 13 |
| 4 | 19.9422 | 0.2103 | 0.6804 | 0.7143 | 13 |
| 5 | 17.9227 | 0.2951 | 0.6895 | 0.5882 | 13 |
| **Mean** | **18.5021** | **0.2559** | **0.7076** | **0.6016** | **13.2** |
| **Std** | **0.9961** | **0.0296** | **0.0240** | **0.0646** | |

## Test Set 성능 (Test 20%)

Test R² = **0.2970**
Test MAE = **19.3902**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.5482 (p=5.677e-23) |
| Shapiro-Wilk p | 0.0001 |
| Bias t-test p | 0.3106 |
| 이진화 ACC (성별 기준) | 0.7055 |
| 이진화 AUC (성별 기준) | 0.5326 |
| 이진화 AUPRC (성별 기준) | 0.7962 |
| 이진화 Brier Score (성별 기준) | 0.4347 |

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

