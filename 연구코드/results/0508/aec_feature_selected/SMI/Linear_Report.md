# Linear Regression Report — SMI

**Input features:** AEC 선택 피처 (|r|<0.8, VIF<10.0) — Train 기준 14개, 폴드 평균 13.2개

## 피처 선택 결과 (폴드별)

| Fold | 선택된 AEC 피처 수 |
|---|---|
| 1 | 14 |
| 2 | 13 |
| 3 | 13 |
| 4 | 13 |
| 5 | 13 |
| **Mean** | **13.2** |

## Train 기준 최종 선택 피처

총 **14개**: skewness, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, peak_max_width, valley_count, slope_mean, slope_max, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

## 성별 데이터 분포

### SMI

| 통계 | Male | Female |
|---|---|---|
| Count | 514 | 859 |
| Mean | 51.7489 | 42.3286 |
| Std | 8.7129 | 6.2294 |
| Min | 5.4688 | 14.0547 |
| Q25 | 46.6563 | 38.4583 |
| Median | 52.1349 | 41.6419 |
| Q75 | 57.1952 | 45.7010 |
| Max | 87.4614 | 87.6616 |

### BMI

| 통계 | Male | Female |
|---|---|---|
| Count | 514 | 859 |
| Mean | 24.1897 | 23.0643 |
| Std | 3.2702 | 3.3946 |
| Min | 14.4795 | 14.4000 |
| Q25 | 22.1389 | 20.7541 |
| Median | 24.1632 | 22.7444 |
| Q75 | 25.9021 | 25.0042 |
| Max | 36.7570 | 39.4905 |

### PatientAge

| 통계 | Male | Female |
|---|---|---|
| Count | 514 | 859 |
| Mean | 59.6459 | 55.5716 |
| Std | 12.5255 | 12.1897 |
| Min | 18.0000 | 11.0000 |
| Q25 | 53.0000 | 47.0000 |
| Median | 60.0000 | 55.0000 |
| Q75 | 68.0000 | 64.0000 |
| Max | 89.0000 | 91.0000 |

## 5-Fold CV 성능 (Train 80%)

| Fold | MSE | RMSE | R² |
|---|---|---|---|
| 1 | 74.5299 | 8.6331 | 0.1057 |
| 2 | 59.4795 | 7.7123 | 0.0768 |
| 3 | 66.4040 | 8.1489 | 0.0445 |
| 4 | 70.7149 | 8.4092 | -0.0544 |
| 5 | 61.8879 | 7.8669 | 0.0726 |
| **Mean** | **66.6032** | **8.1541** | **0.0490** |
| **Std** | **5.5274** | **0.3383** | **0.0552** |

## Test Set 성능 (Test 20%)

| MSE | RMSE | R² |
|---|---|---|
| **83.1012** | **9.1160** | **0.0186** |

## 상위 20 계수 (Train 학습)

| Feature | Coefficient | P-value |
|---|---|---|
| zero_crossing_rate | 17265.305518 | 0.0000 |
| wavelet_energy_ratio_D1 | -17221.211597 | 0.0000 |
| skewness | -71.588722 | 0.0001 |
| slope_mean | 61.133205 | 0.0442 |
| valley_count | -17.309322 | 0.0000 |
| slope_max | -12.292824 | 0.3225 |
| peak_std_height | -3.866993 | 0.0212 |
| peak_first_pos | -1.287326 | 0.1475 |
| first_high_pos | 1.233032 | 0.1655 |
| peak_mean_width | -1.097651 | 0.2599 |
| peak_main_pos | 1.082371 | 0.0920 |
| peak_max_width | -0.449758 | 0.5422 |
| wavelet_cD2_energy | 0.348196 | 0.4902 |
| wavelet_cD3_energy | 0.001873 | 0.9977 |
| Intercept | 46.7546 |
