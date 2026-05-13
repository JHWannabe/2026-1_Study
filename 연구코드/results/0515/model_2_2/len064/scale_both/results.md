# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:03  |  5-Fold CV  |  Median best epoch: 3

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 357 | 297 | 83.2% | 60 | 16.8% |
| Train | F | 660 | 609 | 92.3% | 51 | 7.7% |
| Train | **All** | **1017** | **906** | **89.1%** | **111** | **10.9%** |
| Test | M | 99 | 82 | 82.8% | 17 | 17.2% |
| Test | F | 156 | 145 | 92.9% | 11 | 7.1% |
| Test | **All** | **255** | **227** | **89.0%** | **28** | **11.0%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 357 | 59.35 ± 12.80 | 18.00 | 60.00 | 88.00 |
| Train | F | 660 | 55.60 ± 11.74 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **1017** | **56.92 ± 12.25** | **14.00** | **57.00** | **91.00** |
| Test | M | 99 | 61.58 ± 10.97 | 34.00 | 61.00 | 89.00 |
| Test | F | 156 | 55.71 ± 13.31 | 11.00 | 55.00 | 86.00 |
| Test | **All** | **255** | **57.98 ± 12.78** | **11.00** | **59.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 357 | 24.17 ± 3.31 | 14.48 | 24.16 | 36.76 |
| Train | F | 660 | 23.01 ± 3.24 | 15.62 | 22.69 | 34.61 |
| Train | **All** | **1017** | **23.42 ± 3.31** | **14.48** | **23.24** | **36.76** |
| Test | M | 99 | 24.03 ± 3.22 | 16.80 | 24.16 | 33.87 |
| Test | F | 156 | 23.22 ± 3.52 | 14.40 | 22.71 | 36.24 |
| Test | **All** | **255** | **23.54 ± 3.43** | **14.40** | **23.53** | **36.24** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8072 | 0.4731 | 0.1706 | 0.7451 | 0.3158 |
| 2 | 0.6649 | 0.2372 | 0.1962 | 0.7353 | 0.3077 |
| 3 | 0.7328 | 0.2585 | 0.2025 | 0.6749 | 0.2979 |
| 4 | 0.8006 | 0.3533 | 0.1772 | 0.7586 | 0.4096 |
| 5 | 0.8430 | 0.4365 | 0.1708 | 0.7685 | 0.4337 |
| **Mean** | **0.7697** | **0.3517** | **0.1835** | **0.7365** | **0.3529** |
| **±Std** | 0.0634 | 0.0935 | 0.0133 | 0.0328 | 0.0569 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8027 | 0.3597 | 0.2422 | 0.6029 | 0.3193 |
| 2 | 0.7024 | 0.2345 | 0.1695 | 0.7255 | 0.2432 |
| 3 | 0.7830 | 0.3343 | 0.2693 | 0.5567 | 0.3077 |
| 4 | 0.7579 | 0.3902 | 0.2121 | 0.6650 | 0.2917 |
| 5 | 0.8973 | 0.5318 | 0.1948 | 0.6502 | 0.3717 |
| **Mean** | **0.7887** | **0.3701** | **0.2176** | **0.6401** | **0.3067** |
| **±Std** | 0.0639 | 0.0963 | 0.0350 | 0.0572 | 0.0416 |

CrossAttn best val AUC per fold: Fold1=0.8027, Fold2=0.7024, Fold3=0.7830, Fold4=0.7579, Fold5=0.8973

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8177 | 0.2985 | 0.1914 | 0.7137 | 0.3761 |
| CrossAttn | 0.7683 | 0.3231 | 0.1669 | 0.7294 | 0.3429 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7489 | 0.3104 | 0.2756 | 0.5556 | 0.4054 |
| F | 156 | 0.8307 | 0.3278 | 0.1379 | 0.8141 | 0.3256 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7461 | 0.3889 | 0.1904 | 0.6970 | 0.4444 |
| F | 156 | 0.7574 | 0.2500 | 0.1521 | 0.7500 | 0.2353 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 160 | 67 |
| **True: Sarco**  | 6 | 22 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 59 |
| **True: Sarco**  | 10 | 18 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR & CrossAttn) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curves |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices (LR & CrossAttn) |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
