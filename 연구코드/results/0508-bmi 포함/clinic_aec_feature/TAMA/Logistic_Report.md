# Logistic Regression Report — TAMA

**Input features:** PatientSex, PatientAge, BMI (고정) + AEC 선택 피처 (|r|<0.8, VIF<10.0) — 전체모델 17개 (AEC 14개), 폴드 평균 AEC 13.0개

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

## 피처 선택 결과 (폴드별 AEC 피처 수)

| Fold | 선택된 AEC 피처 수 |
|---|---|
| 1 | 12 |
| 2 | 14 |
| 3 | 12 |
| 4 | 13 |
| 5 | 14 |
| **Mean** | **13.0** |

## 5-Fold CV 성능 (Train 80%)

| Fold | Accuracy | AUC-ROC |
|---|---|---|
| 1 | 0.7545 | 0.7263 |
| 2 | 0.7273 | 0.6901 |
| 3 | 0.7364 | 0.6939 |
| 4 | 0.7534 | 0.7079 |
| 5 | 0.7808 | 0.7551 |
| **Mean** | **0.7505** | **0.7147** |
| **Std** | **0.0184** | **0.0239** |

## Test Set 성능 (Test 20%)

| Accuracy | AUC-ROC |
|---|---|
| **0.7782** | **0.7712** |

## Confusion Matrix (Test)

|  | Pred Normal | Pred Low TAMA |
|---|---|---|
| Actual Normal | 193 | 10 |
| Actual Low TAMA | 51 | 21 |

## Classification Report (Test)

```
precision    recall  f1-score   support

      Normal       0.79      0.95      0.86       203
    Low TAMA       0.68      0.29      0.41        72

    accuracy                           0.78       275
   macro avg       0.73      0.62      0.64       275
weighted avg       0.76      0.78      0.74       275
```

## 상위 20 계수 (Train 학습)

| Feature | Coefficient | Odds Ratio | P-value |
|---|---|---|---|
| zero_crossing_rate | 4.006867 | 54.9743 | 0.7651 |
| slope_mean | 0.668300 | 1.9509 | 0.0201 |
| skewness | 0.294715 | 1.3427 | 0.0427 |
| BMI | -0.237201 | 0.7888 | 0.0000 |
| PatientSex | -0.204308 | 0.8152 | 0.2546 |
| PatientAge | 0.039360 | 1.0401 | 0.0000 |
| valley_count | 0.014125 | 1.0142 | 0.4860 |
| slope_max | 0.008940 | 1.0090 | 0.7784 |
| peak_std_height | -0.005509 | 0.9945 | 0.4637 |
| peak_first_pos | -0.003938 | 0.9961 | 0.4313 |
| first_high_pos | -0.003631 | 0.9964 | 0.3027 |
| wavelet_energy_ratio_D1 | -0.001375 | 0.9986 | 1.0000 |
| peak_main_pos | 0.001066 | 1.0011 | 0.5742 |
| peak_mean_width | -0.000578 | 0.9994 | 0.9306 |
| wavelet_cD2_energy | 0.000396 | 1.0004 | 0.5760 |
| wavelet_cD3_energy | -0.000315 | 0.9997 | 0.0118 |
| peak_max_width | -0.000128 | 0.9999 | 0.9716 |
