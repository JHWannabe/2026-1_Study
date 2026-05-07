# Logistic Regression Report — TAMA

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
| 1 | 0.7455 | 0.7247 |
| 2 | 0.7727 | 0.7276 |
| 3 | 0.7182 | 0.6996 |
| 4 | 0.7671 | 0.7039 |
| 5 | 0.7534 | 0.7592 |
| **Mean** | **0.7514** | **0.7230** |
| **Std** | **0.0192** | **0.0212** |

## Test Set 성능 (Test 20%)

| Accuracy | AUC-ROC |
|---|---|
| **0.7636** | **0.7869** |

## Confusion Matrix (Test)

|  | Pred Normal | Pred Low TAMA |
|---|---|---|
| Actual Normal | 192 | 11 |
| Actual Low TAMA | 54 | 18 |

## Classification Report (Test)

```
precision    recall  f1-score   support

      Normal       0.78      0.95      0.86       203
    Low TAMA       0.62      0.25      0.36        72

    accuracy                           0.76       275
   macro avg       0.70      0.60      0.61       275
weighted avg       0.74      0.76      0.72       275
```

## 계수 (Train 학습)

| Feature | Coefficient | Odds Ratio | P-value |
|---|---|---|---|
| PatientSex | -0.1578 | 0.8541 | 0.3158 |
| PatientAge | 0.0359 | 1.0366 | 0.0000 |
| BMI | -0.2508 | 0.7782 | 0.0000 |
