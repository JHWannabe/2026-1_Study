# Model1_ClinicOnly Report -- SMI

**Age + Sex + BMI (Logistic Regression)**

## 5-Fold CV (Train 80%)

| Fold | AUC | AUPRC | Brier | Acc |
|---|---|---|---|---|
| 1 | 0.7136 | 0.4411 | 0.1693 | 0.7500 |
| 2 | 0.6931 | 0.3914 | 0.1776 | 0.7409 |
| 3 | 0.7600 | 0.4962 | 0.1591 | 0.7636 |
| 4 | 0.7804 | 0.5224 | 0.1542 | 0.7991 |
| 5 | 0.7701 | 0.5083 | 0.1567 | 0.7534 |
| Mean | 0.7434 | 0.4719 | 0.1634 | 0.7614 |
| Std  | 0.0340  | 0.0488  | 0.0088  | 0.0202  |

## Test Set (20%)

| AUC | AUPRC | Brier | Acc |
|---|---|---|---|
| **0.8402** | **0.6618** | **0.1394** | **0.8000** |

## Classification Report

```
precision    recall  f1-score   support

      Normal       0.80      0.98      0.88       206
     Low SMI       0.82      0.26      0.40        69

    accuracy                           0.80       275
   macro avg       0.81      0.62      0.64       275
weighted avg       0.80      0.80      0.76       275
```
