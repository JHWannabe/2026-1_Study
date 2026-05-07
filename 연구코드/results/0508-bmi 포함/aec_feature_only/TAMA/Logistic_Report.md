# Logistic Regression Report — TAMA

**Input features:** signal_length ~ wavelet_energy_ratio_D1 (65개)

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

| Fold | Accuracy | AUC-ROC |
|---|---|---|
| 1 | 0.7409 | 0.4813 |
| 2 | 0.7409 | 0.4255 |
| 3 | 0.7364 | 0.4872 |
| 4 | 0.7397 | 0.4906 |
| 5 | 0.7397 | 0.4317 |
| **Mean** | **0.7395** | **0.4633** |
| **Std** | **0.0017** | **0.0285** |

## Test Set 성능 (Test 20%)

| Accuracy | AUC-ROC |
|---|---|
| **0.7382** | **0.5099** |

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
| spectral_energy | -0.088750 | 0.9151 | nan |
| band1_energy | -0.088724 | 0.9151 | nan |
| fft_mag_max | 0.002840 | 1.0028 | nan |
| AUC | 0.002840 | 1.0028 | nan |
| fft_mag_std | 0.002830 | 1.0028 | nan |
| fft_mag_mean | 0.002829 | 1.0028 | nan |
| signal_length | 0.002829 | 1.0028 | nan |
| max | 0.002829 | 1.0028 | nan |
| peak_max_height | 0.002829 | 1.0028 | nan |
| p95 | 0.002829 | 1.0028 | nan |
| p90 | 0.002829 | 1.0028 | nan |
| peak_mean_height | 0.002829 | 1.0028 | nan |
| p75 | 0.002829 | 1.0028 | nan |
| RMSE | 0.002829 | 1.0028 | nan |
| mean | 0.002829 | 1.0028 | nan |
| AUC_normalized | 0.002829 | 1.0028 | nan |
| median | 0.002829 | 1.0028 | nan |
| peak_last_pos | 0.002829 | 1.0028 | nan |
| p5 | 0.002829 | 1.0028 | nan |
| p25 | 0.002829 | 1.0028 | nan |
