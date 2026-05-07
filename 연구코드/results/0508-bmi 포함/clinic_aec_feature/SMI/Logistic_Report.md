# Logistic Regression Report — SMI

**Input features:** PatientSex, PatientAge, BMI (고정) + AEC 선택 피처 (|r|<0.8, VIF<10.0) — 전체모델 15개 (AEC 12개), 폴드 평균 AEC 12.6개

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

## 피처 선택 결과 (폴드별 AEC 피처 수)

| Fold | 선택된 AEC 피처 수 |
|---|---|
| 1 | 13 |
| 2 | 12 |
| 3 | 13 |
| 4 | 13 |
| 5 | 12 |
| **Mean** | **12.6** |

## 5-Fold CV 성능 (Train 80%)

| Fold | Accuracy | AUC-ROC |
|---|---|---|
| 1 | 0.7545 | 0.7040 |
| 2 | 0.7364 | 0.6698 |
| 3 | 0.7682 | 0.7191 |
| 4 | 0.7945 | 0.7764 |
| 5 | 0.7580 | 0.7468 |
| **Mean** | **0.7623** | **0.7232** |
| **Std** | **0.0191** | **0.0364** |

## Test Set 성능 (Test 20%)

| Accuracy | AUC-ROC |
|---|---|
| **0.7927** | **0.8326** |

## Confusion Matrix (Test)

|  | Pred Normal | Pred Low SMI |
|---|---|---|
| Actual Normal | 202 | 4 |
| Actual Low SMI | 53 | 16 |

## Classification Report (Test)

```
precision    recall  f1-score   support

      Normal       0.79      0.98      0.88       206
     Low SMI       0.80      0.23      0.36        69

    accuracy                           0.79       275
   macro avg       0.80      0.61      0.62       275
weighted avg       0.79      0.79      0.75       275
```

## 상위 20 계수 (Train 학습)

| Feature | Coefficient | Odds Ratio | P-value |
|---|---|---|---|
| zero_crossing_rate | -12.368818 | 0.0000 | 0.3964 |
| PatientSex | -0.647167 | 0.5235 | 0.0005 |
| BMI | -0.317063 | 0.7283 | 0.0000 |
| skewness | 0.208650 | 1.2320 | 0.1805 |
| slope_mean | -0.143268 | 0.8665 | 0.6101 |
| peak_count | 0.010990 | 1.0111 | 0.7304 |
| PatientAge | 0.010282 | 1.0103 | 0.1281 |
| peak_mean_width | 0.005390 | 1.0054 | 0.3436 |
| wavelet_energy_ratio_D1 | -0.002824 | 0.9972 | 1.0000 |
| first_high_pos | -0.002239 | 0.9978 | 0.5101 |
| peak_main_pos | 0.001448 | 1.0014 | 0.4540 |
| peak_first_pos | 0.001035 | 1.0010 | 0.8220 |
| wavelet_cD2_energy | 0.000420 | 1.0004 | 0.4937 |
| peak_std_height | -0.000229 | 0.9998 | 0.9737 |
| wavelet_cD3_energy | -0.000181 | 0.9998 | 0.1102 |
