# Model2_AEConly Report -- SMI

**AEC-CNN only (no clinical)**

## 5-Fold CV (Train 80%)

| Fold | AUC | AUPRC | Brier | Acc |
|---|---|---|---|---|
| 1 | 0.3702 | 0.2214 | 0.2720 | 0.2500 |
| 2 | 0.5657 | 0.2773 | 0.2383 | 0.7500 |
| 3 | 0.5549 | 0.3150 | 0.2527 | 0.5227 |
| 4 | 0.3501 | 0.1990 | 0.2419 | 0.7489 |
| 5 | 0.5935 | 0.2972 | 0.2532 | 0.3881 |
| Mean | 0.4869 | 0.2620 | 0.2516 | 0.5319 |
| Std  | 0.1044  | 0.0445  | 0.0118  | 0.1974  |

## Test Set (20%)

| AUC | AUPRC | Brier | Acc |
|---|---|---|---|
| **0.6093** | **0.3347** | **0.2445** | **0.5418** |

## Classification Report

```
precision    recall  f1-score   support

      Normal       0.79      0.53      0.63       206
     Low SMI       0.29      0.58      0.39        69

    accuracy                           0.54       275
   macro avg       0.54      0.55      0.51       275
weighted avg       0.66      0.54      0.57       275
```
