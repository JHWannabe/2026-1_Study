# Model4_ClinicHC Report -- SMI

**Age + Sex + BMI + handcrafted AEC (31 features, corr-dedup)**

## 5-Fold CV (Train 80%)

| Fold | AUC | AUPRC | Brier | Acc |
|---|---|---|---|---|
| 1 | 0.7060 | 0.4514 | 0.1707 | 0.7727 |
| 2 | 0.6653 | 0.3840 | 0.1854 | 0.7318 |
| 3 | 0.7185 | 0.4671 | 0.1688 | 0.7545 |
| 4 | 0.7917 | 0.5300 | 0.1512 | 0.7808 |
| 5 | 0.7215 | 0.4533 | 0.1686 | 0.7397 |
| Mean | 0.7206 | 0.4572 | 0.1689 | 0.7559 |
| Std  | 0.0408  | 0.0465  | 0.0109  | 0.0187  |

## Test Set (20%)

| AUC | AUPRC | Brier | Acc |
|---|---|---|---|
| **0.8291** | **0.6521** | **0.1373** | **0.8073** |

## Classification Report

```
precision    recall  f1-score   support

      Normal       0.81      0.98      0.88       206
     Low SMI       0.81      0.30      0.44        69

    accuracy                           0.81       275
   macro avg       0.81      0.64      0.66       275
weighted avg       0.81      0.81      0.77       275
```
