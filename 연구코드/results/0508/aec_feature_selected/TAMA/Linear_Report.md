# Linear Regression Report — TAMA

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

### TAMA

| 통계 | Male | Female |
|---|---|---|
| Count | 514 | 859 |
| Mean | 148.2276 | 104.9418 |
| Std | 26.3471 | 15.3114 |
| Min | 14.0000 | 30.0000 |
| Q25 | 132.0000 | 95.0000 |
| Median | 148.0000 | 103.0000 |
| Q75 | 165.0000 | 114.0000 |
| Max | 220.0000 | 190.0000 |

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
| 1 | 793.3532 | 28.1665 | 0.1584 |
| 2 | 666.2547 | 25.8119 | 0.0891 |
| 3 | 740.5651 | 27.2133 | 0.0159 |
| 4 | 913.6624 | 30.2268 | -0.0480 |
| 5 | 686.0533 | 26.1926 | 0.1064 |
| **Mean** | **759.9777** | **27.5222** | **0.0644** |
| **Std** | **88.7227** | **1.5823** | **0.0724** |

## Test Set 성능 (Test 20%)

| MSE | RMSE | R² |
|---|---|---|
| **897.2981** | **29.9549** | **0.0617** |

## 상위 20 계수 (Train 학습)

| Feature | Coefficient | P-value |
|---|---|---|
| zero_crossing_rate | 60682.719799 | 0.0000 |
| wavelet_energy_ratio_D1 | -60402.816848 | 0.0000 |
| skewness | -291.014041 | 0.0000 |
| slope_mean | 159.440223 | 0.1190 |
| valley_count | -68.142161 | 0.0000 |
| slope_max | -56.381209 | 0.1780 |
| peak_std_height | -14.029205 | 0.0131 |
| peak_first_pos | -6.896758 | 0.0213 |
| peak_main_pos | 5.279028 | 0.0148 |
| peak_max_width | -2.731148 | 0.2718 |
| peak_mean_width | -2.589347 | 0.4300 |
| wavelet_cD3_energy | -1.266456 | 0.5683 |
| first_high_pos | -1.131893 | 0.7053 |
| wavelet_cD2_energy | -0.439983 | 0.7957 |
| Intercept | 132.8035 |
