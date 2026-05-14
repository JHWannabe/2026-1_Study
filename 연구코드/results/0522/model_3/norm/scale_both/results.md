# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:56  |  5-Fold CV  |  Median best epoch: 9

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 355 | 292 | 82.3% | 63 | 17.7% |
| Train | F | 661 | 614 | 92.9% | 47 | 7.1% |
| Train | **All** | **1016** | **906** | **89.2%** | **110** | **10.8%** |
| Test | M | 97 | 83 | 85.6% | 14 | 14.4% |
| Test | F | 158 | 145 | 91.8% | 13 | 8.2% |
| Test | **All** | **255** | **228** | **89.4%** | **27** | **10.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 355 | 59.92 ± 12.67 | 18.00 | 60.00 | 89.00 |
| Train | F | 661 | 55.55 ± 11.94 | 18.00 | 55.00 | 91.00 |
| Train | **All** | **1016** | **57.07 ± 12.38** | **18.00** | **57.00** | **91.00** |
| Test | M | 97 | 58.63 ± 12.43 | 28.00 | 59.00 | 88.00 |
| Test | F | 158 | 55.27 ± 11.46 | 23.00 | 56.00 | 86.00 |
| Test | **All** | **255** | **56.55 ± 11.95** | **23.00** | **57.00** | **88.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 355 | 24.22 ± 3.38 | 14.48 | 24.16 | 36.76 |
| Train | F | 661 | 23.14 ± 3.39 | 14.40 | 22.83 | 36.24 |
| Train | **All** | **1016** | **23.52 ± 3.42** | **14.40** | **23.37** | **36.76** |
| Test | M | 97 | 24.50 ± 3.14 | 18.37 | 24.49 | 35.68 |
| Test | F | 158 | 23.11 ± 3.24 | 16.87 | 22.72 | 34.23 |
| Test | **All** | **255** | **23.64 ± 3.27** | **16.87** | **23.34** | **35.68** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7575 | 0.4053 | 0.1721 | 0.7402 | 0.3457 |
| 2 | 0.7953 | 0.3091 | 0.1574 | 0.7685 | 0.3733 |
| 3 | 0.7572 | 0.2794 | 0.1633 | 0.7734 | 0.3429 |
| 4 | 0.7539 | 0.2745 | 0.1763 | 0.7537 | 0.3590 |
| 5 | 0.7358 | 0.3011 | 0.1783 | 0.7635 | 0.3514 |
| **Mean** | **0.7599** | **0.3139** | **0.1695** | **0.7599** | **0.3544** |
| **±Std** | 0.0194 | 0.0475 | 0.0079 | 0.0118 | 0.0109 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8177 | 0.4161 | 0.1643 | 0.7549 | 0.4048 |
| 2 | 0.8433 | 0.4794 | 0.1690 | 0.7438 | 0.3953 |
| 3 | 0.8214 | 0.3831 | 0.1188 | 0.8325 | 0.3462 |
| 4 | 0.8576 | 0.5329 | 0.2121 | 0.7094 | 0.4040 |
| 5 | 0.8380 | 0.3794 | 0.2072 | 0.7143 | 0.3958 |
| **Mean** | **0.8356** | **0.4382** | **0.1743** | **0.7510** | **0.3892** |
| **±Std** | 0.0146 | 0.0594 | 0.0338 | 0.0443 | 0.0219 |

CrossAttn best val AUC per fold: Fold1=0.8177, Fold2=0.8433, Fold3=0.8214, Fold4=0.8576, Fold5=0.8380

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6886 | 0.2024 | 0.1771 | 0.7569 | 0.2791 |
| CrossAttn | 0.7359 | 0.2698 | 0.1740 | 0.7608 | 0.2989 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6781 | 0.2358 | 0.2390 | 0.6495 | 0.3462 |
| F | 158 | 0.6711 | 0.2152 | 0.1392 | 0.8228 | 0.1765 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7599 | 0.3310 | 0.2425 | 0.6598 | 0.3774 |
| F | 158 | 0.6785 | 0.2115 | 0.1319 | 0.8228 | 0.1765 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 181 | 47 |
| **True: Sarco**  | 15 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 181 | 47 |
| **True: Sarco**  | 14 | 13 |

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
