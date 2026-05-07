# Primary Model Report — SMI

**구조**: AEC CNN score → Low SMI ~ Age + Sex + BMI + CNN_score

## 5-Fold CV 성능 (Train 80%)

| Fold | AUC | AUPRC | Brier | Acc |
|---|---|---|---|---|
| 1 | 0.6267 | 0.3372 | 0.2961 | 0.6591 |
| 2 | 0.6658 | 0.3622 | 0.3025 | 0.6636 |
| 3 | 0.7377 | 0.4395 | 0.2559 | 0.7182 |
| 4 | 0.7068 | 0.4269 | 0.2501 | 0.7215 |
| 5 | 0.6813 | 0.4348 | 0.2955 | 0.6758 |
| **Mean** | **0.6836** | **0.4001** | **0.2800** | **0.6876** |
| **Std**  | **0.0375**  | **0.0421**  | **0.0223**  | **0.0269**  |

## Test Set 성능 (Test 20%)

| AUC | AUPRC | Brier | Acc |
|---|---|---|---|
| **0.6881** | **0.4068** | **0.2594** | **0.6691** |

## Classification Report (Test)

```
precision    recall  f1-score   support

      Normal       0.76      0.82      0.79       206
     Low SMI       0.29      0.22      0.25        69

    accuracy                           0.67       275
   macro avg       0.52      0.52      0.52       275
weighted avg       0.64      0.67      0.65       275
```
