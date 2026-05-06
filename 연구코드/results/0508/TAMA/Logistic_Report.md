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

## 5-Fold CV 성능

| Fold | Accuracy | AUC-ROC |
|---|---|---|
| 1 | 0.7964 | 0.7575 |
| 2 | 0.7455 | 0.7210 |
| 3 | 0.7527 | 0.7429 |
| 4 | 0.7555 | 0.7431 |
| 5 | 0.7190 | 0.7307 |
| **Mean** | **0.7538** | **0.7390** |
| **Std** | **0.0249** | **0.0124** |

OOF 전체 AUC-ROC = **0.7372**

## Confusion Matrix (OOF)

|  | Pred Normal | Pred Low TAMA |
|---|---|---|
| Actual Normal | 963 | 52 |
| Actual Low TAMA | 286 | 72 |

## Classification Report (OOF)

```
precision    recall  f1-score   support

      Normal       0.77      0.95      0.85      1015
    Low TAMA       0.58      0.20      0.30       358

    accuracy                           0.75      1373
   macro avg       0.68      0.57      0.57      1373
weighted avg       0.72      0.75      0.71      1373
```

## 계수 (전체 데이터 학습)

| Feature | Coefficient (log-odds) |
|---|---|
| PatientSex | -0.1502 |
| PatientAge | 0.0385 |
| BMI | -0.2718 |
