# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 16:48  |  5-Fold CV  |  Median best epoch: 5

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
| 1 | 0.7430 | 0.2622 | 0.1766 | 0.7500 | 0.3377 |
| 2 | 0.6680 | 0.2776 | 0.1932 | 0.7255 | 0.2432 |
| 3 | 0.7551 | 0.2340 | 0.1988 | 0.6995 | 0.3297 |
| 4 | 0.8410 | 0.4761 | 0.1630 | 0.7833 | 0.4211 |
| 5 | 0.8358 | 0.4954 | 0.1799 | 0.7094 | 0.3656 |
| **Mean** | **0.7686** | **0.3491** | **0.1823** | **0.7335** | **0.3394** |
| **±Std** | 0.0644 | 0.1127 | 0.0126 | 0.0302 | 0.0578 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8059 | 0.4088 | 0.1916 | 0.7451 | 0.3500 |
| 2 | 0.6973 | 0.2624 | 0.1967 | 0.7304 | 0.3373 |
| 3 | 0.8009 | 0.3260 | 0.2253 | 0.6552 | 0.3396 |
| 4 | 0.8332 | 0.4575 | 0.1843 | 0.7241 | 0.3913 |
| 5 | 0.9108 | 0.5875 | 0.2349 | 0.6010 | 0.3415 |
| **Mean** | **0.8096** | **0.4085** | **0.2066** | **0.6912** | **0.3519** |
| **±Std** | 0.0686 | 0.1119 | 0.0198 | 0.0547 | 0.0201 |

CrossAttn best val AUC per fold: Fold1=0.8059, Fold2=0.6973, Fold3=0.8009, Fold4=0.8332, Fold5=0.9108

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8104 | 0.3210 | 0.1855 | 0.7373 | 0.3853 |
| CrossAttn | 0.8353 | 0.3323 | 0.1737 | 0.7490 | 0.4074 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7418 | 0.3385 | 0.2558 | 0.6162 | 0.4242 |
| F | 156 | 0.8119 | 0.3779 | 0.1409 | 0.8141 | 0.3256 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7676 | 0.3538 | 0.2242 | 0.6566 | 0.4516 |
| F | 156 | 0.8633 | 0.3915 | 0.1416 | 0.8077 | 0.3478 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 60 |
| **True: Sarco**  | 7 | 21 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 169 | 58 |
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
