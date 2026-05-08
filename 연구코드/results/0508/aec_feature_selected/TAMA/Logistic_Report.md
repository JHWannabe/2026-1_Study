# Logistic Regression Report — TAMA

**Input features:** AEC 선택 피처 (|r|<0.8, VIF<10.0) — Train 기준 14개, 폴드 평균 13.6개

## 피처 선택 결과 (폴드별)

| Fold | 선택된 AEC 피처 수 |
|---|---|
| 1 | 13 |
| 2 | 14 |
| 3 | 13 |
| 4 | 14 |
| 5 | 14 |
| **Mean** | **13.6** |

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

## 성별 하위 25% 임계값

| Sex | Threshold |
|---|---|
| Male | 132.0000 |
| Female | 95.0000 |

## 5-Fold CV 성능 (Train 80%)

| Fold | Accuracy | AUC-ROC | AUPRC | Brier |
|---|---|---|---|---|
| 1 | 0.7455 | 0.4764 | 0.2802 | 0.1972 |
| 2 | 0.7318 | 0.4057 | 0.2335 | 0.1993 |
| 3 | 0.7273 | 0.5343 | 0.2913 | 0.1960 |
| 4 | 0.7397 | 0.5271 | 0.2832 | 0.1965 |
| 5 | 0.7306 | 0.5795 | 0.3162 | 0.1912 |
| **Mean** | **0.7350** | **0.5046** | **0.2809** | **0.1960** |
| **Std** | **0.0066** | **0.0593** | **0.0268** | **0.0026** |

## Test Set 성능 (Test 20%)

| Accuracy | AUC-ROC | AUPRC | Brier |
|---|---|---|---|
| **0.7382** | **0.6388** | **0.3869** | **0.1880** |

## Confusion Matrix (Test)

|  | Pred Normal | Pred Low TAMA |
|---|---|---|
| Actual Normal | 203 | 0 |
| Actual Low TAMA | 72 | 0 |

## Classification Report (Test)

```
precision    recall  f1-score   support

      Normal       0.74      1.00      0.85       203
    Low TAMA       0.00      0.00      0.00        72

    accuracy                           0.74       275
   macro avg       0.37      0.50      0.42       275
weighted avg       0.54      0.74      0.63       275
```

## 상위 20 계수 (Train 학습)

| Feature | Coefficient | Odds Ratio | P-value |
|---|---|---|---|
| slope_mean | 16.046364 | 9307808.6096 | 1.0000 |
| wavelet_energy_ratio_D1 | -13.307236 | 0.0000 | 1.0000 |
| zero_crossing_rate | -12.941807 | 0.0000 | 1.0000 |
| skewness | 9.174204 | 9645.0915 | 1.0000 |
| valley_count | 1.676597 | 5.3473 | 1.0000 |
| first_high_pos | -0.534927 | 0.5857 | 1.0000 |
| slope_max | -0.444225 | 0.6413 | 1.0000 |
| peak_max_width | 0.232742 | 1.2621 | 1.0000 |
| peak_std_height | 0.105387 | 1.1111 | 1.0000 |
| peak_main_pos | 0.096698 | 1.1015 | 1.0000 |
| peak_mean_width | -0.077086 | 0.9258 | 1.0000 |
| wavelet_cD2_energy | 0.045979 | 1.0471 | 1.0000 |
| peak_first_pos | -0.042712 | 0.9582 | 1.0000 |
| wavelet_cD3_energy | -0.029978 | 0.9705 | 1.0000 |
