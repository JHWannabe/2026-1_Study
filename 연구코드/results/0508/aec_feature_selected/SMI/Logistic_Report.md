# Logistic Regression Report — SMI

**Input features:** AEC 선택 피처 (|r|<0.8, VIF<10.0) — Train 기준 12개, 폴드 평균 12.8개

## 피처 선택 결과 (폴드별)

| Fold | 선택된 AEC 피처 수 |
|---|---|
| 1 | 13 |
| 2 | 13 |
| 3 | 13 |
| 4 | 13 |
| 5 | 12 |
| **Mean** | **12.8** |

## Train 기준 최종 선택 피처

총 **12개**: skewness, peak_count, peak_std_height, peak_first_pos, peak_main_pos, peak_mean_width, slope_mean, zero_crossing_rate, first_high_pos, wavelet_cD3_energy, wavelet_cD2_energy, wavelet_energy_ratio_D1

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

## 성별 하위 25% 임계값

| Sex | Threshold |
|---|---|
| Male | 46.6563 |
| Female | 38.4583 |

## 5-Fold CV 성능 (Train 80%)

| Fold | Accuracy | AUC-ROC | AUPRC | Brier |
|---|---|---|---|---|
| 1 | 0.7455 | 0.5510 | 0.2845 | 0.1886 |
| 2 | 0.7455 | 0.4814 | 0.2558 | 0.1889 |
| 3 | 0.7455 | 0.5626 | 0.2700 | 0.1903 |
| 4 | 0.7443 | 0.4653 | 0.2430 | 0.1939 |
| 5 | 0.7489 | 0.4892 | 0.2566 | 0.1924 |
| **Mean** | **0.7459** | **0.5099** | **0.2620** | **0.1908** |
| **Std** | **0.0015** | **0.0392** | **0.0141** | **0.0021** |

## Test Set 성능 (Test 20%)

| Accuracy | AUC-ROC | AUPRC | Brier |
|---|---|---|---|
| **0.7491** | **0.5323** | **0.2752** | **0.1886** |

## Confusion Matrix (Test)

|  | Pred Normal | Pred Low SMI |
|---|---|---|
| Actual Normal | 206 | 0 |
| Actual Low SMI | 69 | 0 |

## Classification Report (Test)

```
precision    recall  f1-score   support

      Normal       0.75      1.00      0.86       206
     Low SMI       0.00      0.00      0.00        69

    accuracy                           0.75       275
   macro avg       0.37      0.50      0.43       275
weighted avg       0.56      0.75      0.64       275
```

## 상위 20 계수 (Train 학습)

| Feature | Coefficient | Odds Ratio | P-value |
|---|---|---|---|
| skewness | 7.756135 | 2335.8594 | 1.0000 |
| slope_mean | -3.647441 | 0.0261 | 1.0000 |
| zero_crossing_rate | -2.739664 | 0.0646 | 1.0000 |
| wavelet_energy_ratio_D1 | -2.696047 | 0.0675 | 1.0000 |
| peak_count | 1.479835 | 4.3922 | 1.0000 |
| peak_std_height | 0.492224 | 1.6360 | 1.0000 |
| peak_first_pos | -0.287656 | 0.7500 | 1.0000 |
| wavelet_cD3_energy | -0.249363 | 0.7793 | 1.0000 |
| first_high_pos | -0.066753 | 0.9354 | 1.0000 |
| peak_mean_width | 0.054062 | 1.0556 | 1.0000 |
| wavelet_cD2_energy | -0.052481 | 0.9489 | 1.0000 |
| peak_main_pos | -0.042850 | 0.9581 | 1.0000 |
