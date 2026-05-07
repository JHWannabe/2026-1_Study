# Secondary Model Report — SMI

**구조**: Late-fusion CNN — AEC branch + Age/Sex/BMI → end-to-end binary classification

## 5-Fold CV 성능 (Train 80%)

| Fold | AUC | AUPRC | Brier | Acc |
|---|---|---|---|---|
| 1 | 0.5532 | 0.2979 | 0.2967 | 0.6818 |
| 2 | 0.5938 | 0.3727 | 0.2765 | 0.6955 |
| 3 | 0.6378 | 0.3546 | 0.2917 | 0.6727 |
| 4 | 0.6402 | 0.3969 | 0.2596 | 0.7215 |
| 5 | 0.6389 | 0.4073 | 0.2752 | 0.6986 |
| **Mean** | **0.6128** | **0.3659** | **0.2799** | **0.6940** |
| **Std**  | **0.0346**  | **0.0387**  | **0.0132**  | **0.0166**  |

## Test Set 성능 (Test 20%)

| AUC | AUPRC | Brier | Acc |
|---|---|---|---|
| **0.6303** | **0.3907** | **0.2431** | **0.7164** |

## Classification Report (Test)

```
precision    recall  f1-score   support

      Normal       0.78      0.86      0.82       206
     Low SMI       0.41      0.29      0.34        69

    accuracy                           0.72       275
   macro avg       0.60      0.57      0.58       275
weighted avg       0.69      0.72      0.70       275
```
