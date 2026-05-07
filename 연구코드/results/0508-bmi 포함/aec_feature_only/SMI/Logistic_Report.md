# Logistic Regression Report — SMI

**Input features:** signal_length ~ wavelet_energy_ratio_D1 (65개)

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

| Fold | Accuracy | AUC-ROC |
|---|---|---|
| 1 | 0.7455 | 0.6617 |
| 2 | 0.7318 | 0.5374 |
| 3 | 0.7500 | 0.5994 |
| 4 | 0.7489 | 0.5884 |
| 5 | 0.7443 | 0.5187 |
| **Mean** | **0.7441** | **0.5811** |
| **Std** | **0.0065** | **0.0504** |

## Test Set 성능 (Test 20%)

| Accuracy | AUC-ROC |
|---|---|
| **0.7527** | **0.6287** |

## Confusion Matrix (Test)

|  | Pred Normal | Pred Low SMI |
|---|---|---|
| Actual Normal | 206 | 0 |
| Actual Low SMI | 68 | 1 |

## Classification Report (Test)

```
precision    recall  f1-score   support

      Normal       0.75      1.00      0.86       206
     Low SMI       1.00      0.01      0.03        69

    accuracy                           0.75       275
   macro avg       0.88      0.51      0.44       275
weighted avg       0.81      0.75      0.65       275
```

## 상위 20 계수 (Train 학습)

| Feature | Coefficient | Odds Ratio | P-value |
|---|---|---|---|
| wavelet_cA_energy | 123.521672 | 441347295768898278425116833170362744508473524090830848.0000 | nan |
| band2_energy | 55.673830 | 1509514095443223955636224.0000 | nan |
| band1_energy | -51.820176 | 0.0000 | nan |
| spectral_energy | 50.880231 | 12502694294932846280704.0000 | nan |
| signal_energy | 34.765045 | 1253912135367178.7500 | nan |
| band3_energy | 20.286753 | 646286521.9756 | nan |
| band4_energy | 12.973392 | 430796.6983 | nan |
| slope_min | -4.417267 | 0.0121 | nan |
| wavelet_cD1_energy | -4.417084 | 0.0121 | nan |
| kurtosis | -4.416788 | 0.0121 | nan |
| slope_mean | -4.416703 | 0.0121 | nan |
| dominant_freq | -4.416649 | 0.0121 | nan |
| spectral_rolloff | -4.416649 | 0.0121 | nan |
| wavelet_energy_ratio_D1 | -4.416649 | 0.0121 | nan |
| band4_energy_ratio | -4.416649 | 0.0121 | nan |
| band3_energy_ratio | -4.416649 | 0.0121 | nan |
| band2_energy_ratio | -4.416649 | 0.0121 | nan |
| zero_crossing_rate | -4.416648 | 0.0121 | nan |
| spectral_centroid | -4.416643 | 0.0121 | nan |
| spectral_spread | -4.416634 | 0.0121 | nan |
