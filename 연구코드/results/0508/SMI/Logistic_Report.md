# Logistic Regression Report — SMI

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

## 5-Fold CV 성능

| Fold | Accuracy | AUC-ROC |
|---|---|---|
| 1 | 0.7673 | 0.7270 |
| 2 | 0.7745 | 0.7804 |
| 3 | 0.7709 | 0.7759 |
| 4 | 0.7628 | 0.7858 |
| 5 | 0.7737 | 0.7586 |
| **Mean** | **0.7698** | **0.7655** |
| **Std** | **0.0044** | **0.0213** |

OOF 전체 AUC-ROC = **0.7637**

## Confusion Matrix (OOF)

|  | Pred Normal | Pred Low SMI |
|---|---|---|
| Actual Normal | 966 | 63 |
| Actual Low SMI | 253 | 91 |

## Classification Report (OOF)

```
precision    recall  f1-score   support

      Normal       0.79      0.94      0.86      1029
     Low SMI       0.59      0.26      0.37       344

    accuracy                           0.77      1373
   macro avg       0.69      0.60      0.61      1373
weighted avg       0.74      0.77      0.74      1373
```

## 계수 (전체 데이터 학습)

| Feature | Coefficient (log-odds) |
|---|---|
| PatientSex | -0.4002 |
| PatientAge | 0.0058 |
| BMI | -0.3586 |
