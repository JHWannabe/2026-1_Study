# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 17:15  |  5-Fold CV  |  Median best epoch: 7

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
| 1 | 0.7033 | 0.2206 | 0.1868 | 0.7206 | 0.2597 |
| 2 | 0.5686 | 0.1388 | 0.2197 | 0.6618 | 0.1882 |
| 3 | 0.5427 | 0.1595 | 0.2539 | 0.6207 | 0.1720 |
| 4 | 0.4877 | 0.1228 | 0.2406 | 0.6847 | 0.1795 |
| 5 | 0.6113 | 0.1835 | 0.2252 | 0.6995 | 0.2278 |
| **Mean** | **0.5827** | **0.1650** | **0.2252** | **0.6775** | **0.2055** |
| **±Std** | 0.0724 | 0.0345 | 0.0226 | 0.0343 | 0.0333 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8032 | 0.3824 | 0.1992 | 0.6961 | 0.3404 |
| 2 | 0.6848 | 0.2151 | 0.2299 | 0.6373 | 0.2745 |
| 3 | 0.7946 | 0.3240 | 0.1862 | 0.6995 | 0.3579 |
| 4 | 0.7988 | 0.4321 | 0.2383 | 0.6108 | 0.3361 |
| 5 | 0.8975 | 0.5758 | 0.1791 | 0.6700 | 0.3853 |
| **Mean** | **0.7958** | **0.3859** | **0.2065** | **0.6627** | **0.3389** |
| **±Std** | 0.0674 | 0.1193 | 0.0235 | 0.0342 | 0.0365 |

CrossAttn best val AUC per fold: Fold1=0.8032, Fold2=0.6848, Fold3=0.7946, Fold4=0.7988, Fold5=0.8975

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7212 | 0.2281 | 0.2122 | 0.6980 | 0.3304 |
| CrossAttn | 0.8126 | 0.3393 | 0.1681 | 0.7412 | 0.4000 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6535 | 0.2499 | 0.2554 | 0.6566 | 0.3704 |
| F | 156 | 0.7712 | 0.2313 | 0.1848 | 0.7244 | 0.2951 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7647 | 0.3685 | 0.2105 | 0.6869 | 0.4561 |
| F | 156 | 0.8238 | 0.3199 | 0.1412 | 0.7756 | 0.3396 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 159 | 68 |
| **True: Sarco**  | 9 | 19 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 60 |
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
