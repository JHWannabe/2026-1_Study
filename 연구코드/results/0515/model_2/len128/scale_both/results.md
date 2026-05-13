# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:23  |  5-Fold CV  |  Median best epoch: 12

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
| 1 | 0.7403 | 0.2588 | 0.1764 | 0.7549 | 0.3243 |
| 2 | 0.6635 | 0.2761 | 0.1917 | 0.7255 | 0.2432 |
| 3 | 0.7534 | 0.2315 | 0.2006 | 0.6847 | 0.3333 |
| 4 | 0.8403 | 0.4826 | 0.1642 | 0.7833 | 0.4359 |
| 5 | 0.8564 | 0.4774 | 0.1825 | 0.7241 | 0.3913 |
| **Mean** | **0.7707** | **0.3453** | **0.1831** | **0.7345** | **0.3456** |
| **±Std** | 0.0706 | 0.1109 | 0.0125 | 0.0330 | 0.0653 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8177 | 0.3718 | 0.1525 | 0.7941 | 0.4000 |
| 2 | 0.7017 | 0.2273 | 0.2054 | 0.7059 | 0.3023 |
| 3 | 0.7988 | 0.3338 | 0.1920 | 0.6502 | 0.3238 |
| 4 | 0.8114 | 0.4493 | 0.1966 | 0.6798 | 0.3564 |
| 5 | 0.9043 | 0.6239 | 0.1412 | 0.7291 | 0.4211 |
| **Mean** | **0.8068** | **0.4012** | **0.1776** | **0.7118** | **0.3607** |
| **±Std** | 0.0644 | 0.1323 | 0.0257 | 0.0488 | 0.0447 |

CrossAttn best val AUC per fold: Fold1=0.8177, Fold2=0.7017, Fold3=0.7988, Fold4=0.8114, Fold5=0.9043

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8093 | 0.3148 | 0.1847 | 0.7294 | 0.3670 |
| CrossAttn | 0.8310 | 0.3850 | 0.1737 | 0.7412 | 0.4000 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7504 | 0.3329 | 0.2530 | 0.6162 | 0.4242 |
| F | 156 | 0.8050 | 0.3681 | 0.1413 | 0.8013 | 0.2791 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7733 | 0.4250 | 0.2323 | 0.6667 | 0.4762 |
| F | 156 | 0.8451 | 0.3648 | 0.1364 | 0.7885 | 0.2979 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 166 | 61 |
| **True: Sarco**  | 8 | 20 |

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
