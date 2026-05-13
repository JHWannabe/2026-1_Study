# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 17:15  |  5-Fold CV  |  Median best epoch: 9

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
| 1 | 0.4908 | 0.1373 | 0.2339 | 0.6912 | 0.1600 |
| 2 | 0.6462 | 0.2087 | 0.2090 | 0.7157 | 0.2564 |
| 3 | 0.6492 | 0.1621 | 0.2108 | 0.6995 | 0.2989 |
| 4 | 0.6439 | 0.2688 | 0.2139 | 0.6897 | 0.2588 |
| 5 | 0.6753 | 0.2113 | 0.1958 | 0.7192 | 0.2597 |
| **Mean** | **0.6211** | **0.1976** | **0.2127** | **0.7030** | **0.2468** |
| **±Std** | 0.0661 | 0.0454 | 0.0123 | 0.0123 | 0.0461 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8054 | 0.3585 | 0.1689 | 0.7059 | 0.2857 |
| 2 | 0.7156 | 0.2229 | 0.1952 | 0.7010 | 0.3579 |
| 3 | 0.8034 | 0.3169 | 0.1832 | 0.6847 | 0.3469 |
| 4 | 0.8179 | 0.4119 | 0.1303 | 0.8079 | 0.4179 |
| 5 | 0.9179 | 0.6856 | 0.1785 | 0.7143 | 0.4200 |
| **Mean** | **0.8120** | **0.3992** | **0.1712** | **0.7228** | **0.3657** |
| **±Std** | 0.0643 | 0.1560 | 0.0221 | 0.0436 | 0.0500 |

CrossAttn best val AUC per fold: Fold1=0.8054, Fold2=0.7156, Fold3=0.8034, Fold4=0.8179, Fold5=0.9179

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6204 | 0.1835 | 0.2093 | 0.6863 | 0.2000 |
| CrossAttn | 0.8446 | 0.3566 | 0.1397 | 0.7961 | 0.4222 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6033 | 0.2806 | 0.2481 | 0.6263 | 0.2449 |
| F | 156 | 0.6276 | 0.1310 | 0.1846 | 0.7244 | 0.1569 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7834 | 0.3514 | 0.2016 | 0.6970 | 0.4444 |
| F | 156 | 0.8502 | 0.4553 | 0.1005 | 0.8590 | 0.3889 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 165 | 62 |
| **True: Sarco**  | 18 | 10 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 184 | 43 |
| **True: Sarco**  | 9 | 19 |

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
