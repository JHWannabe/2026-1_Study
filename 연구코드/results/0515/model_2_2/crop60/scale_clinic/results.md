# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:10  |  5-Fold CV  |  Median best epoch: 9

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
| 1 | 0.6933 | 0.2861 | 0.1919 | 0.7108 | 0.2532 |
| 2 | 0.6483 | 0.2333 | 0.2032 | 0.6912 | 0.2759 |
| 3 | 0.5819 | 0.1665 | 0.2434 | 0.6453 | 0.2000 |
| 4 | 0.5522 | 0.1623 | 0.2061 | 0.7143 | 0.2368 |
| 5 | 0.6848 | 0.2269 | 0.2094 | 0.7192 | 0.3133 |
| **Mean** | **0.6321** | **0.2150** | **0.2108** | **0.6962** | **0.2558** |
| **±Std** | 0.0560 | 0.0462 | 0.0173 | 0.0271 | 0.0379 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7942 | 0.3485 | 0.2051 | 0.6863 | 0.3469 |
| 2 | 0.6824 | 0.2311 | 0.2148 | 0.6373 | 0.2449 |
| 3 | 0.8026 | 0.2928 | 0.1598 | 0.7586 | 0.3797 |
| 4 | 0.7745 | 0.4239 | 0.2051 | 0.6601 | 0.3168 |
| 5 | 0.9021 | 0.5633 | 0.1543 | 0.7438 | 0.4468 |
| **Mean** | **0.7912** | **0.3719** | **0.1878** | **0.6972** | **0.3470** |
| **±Std** | 0.0701 | 0.1149 | 0.0255 | 0.0470 | 0.0669 |

CrossAttn best val AUC per fold: Fold1=0.7942, Fold2=0.6824, Fold3=0.8026, Fold4=0.7745, Fold5=0.9021

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6768 | 0.2155 | 0.2199 | 0.6667 | 0.2609 |
| CrossAttn | 0.8082 | 0.3225 | 0.1746 | 0.7216 | 0.3826 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.5940 | 0.2087 | 0.2605 | 0.5758 | 0.2759 |
| F | 156 | 0.7618 | 0.2890 | 0.1941 | 0.7244 | 0.2456 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7726 | 0.3675 | 0.2134 | 0.6465 | 0.4444 |
| F | 156 | 0.8031 | 0.3205 | 0.1499 | 0.7692 | 0.3077 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 155 | 72 |
| **True: Sarco**  | 13 | 15 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 162 | 65 |
| **True: Sarco**  | 6 | 22 |

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
