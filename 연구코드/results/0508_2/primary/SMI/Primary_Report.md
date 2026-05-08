# Primary Model Report — SMI

**구조**: AEC CNN score → Low SMI ~ Age + Sex + BMI + CNN_score

## 5-Fold CV 성능 (Train 80%)

| Fold | AUC | AUPRC | Brier | Acc |
|---|---|---|---|---|
| 1 | 0.6082 | 0.3057 | 0.3096 | 0.6500 |
| 2 | 0.6602 | 0.4018 | 0.2636 | 0.7000 |
| 3 | 0.7239 | 0.4523 | 0.2307 | 0.7364 |
| 4 | 0.7008 | 0.4430 | 0.2722 | 0.6895 |
| 5 | 0.6630 | 0.4261 | 0.3004 | 0.6667 |
| **Mean** | **0.6712** | **0.4058** | **0.2753** | **0.6885** |
| **Std**  | **0.0395**  | **0.0529**  | **0.0281**  | **0.0296**  |

## Test Set 성능 (Test 20%)

| AUC | AUPRC | Brier | Acc |
|---|---|---|---|
| **0.6699** | **0.3849** | **0.2675** | **0.6836** |

## Classification Report (Test)

```
precision    recall  f1-score   support

      Normal       0.77      0.83      0.80       206
     Low SMI       0.33      0.25      0.28        69

    accuracy                           0.68       275
   macro avg       0.55      0.54      0.54       275
weighted avg       0.66      0.68      0.67       275
```
