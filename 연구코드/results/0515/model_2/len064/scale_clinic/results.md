# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 16:20  |  5-Fold CV  |  Median best epoch: 16

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
| 1 | 0.5704 | 0.1823 | 0.2029 | 0.6863 | 0.2195 |
| 2 | 0.6486 | 0.1975 | 0.2123 | 0.6667 | 0.2273 |
| 3 | 0.6750 | 0.2201 | 0.2107 | 0.6847 | 0.2889 |
| 4 | 0.7024 | 0.3045 | 0.1944 | 0.7438 | 0.3500 |
| 5 | 0.7730 | 0.2986 | 0.1877 | 0.7094 | 0.3371 |
| **Mean** | **0.6739** | **0.2406** | **0.2016** | **0.6982** | **0.2846** |
| **±Std** | 0.0663 | 0.0512 | 0.0094 | 0.0266 | 0.0540 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8389 | 0.3839 | 0.1977 | 0.7108 | 0.3516 |
| 2 | 0.7163 | 0.2267 | 0.1802 | 0.7500 | 0.3704 |
| 3 | 0.8056 | 0.3923 | 0.2345 | 0.6010 | 0.3193 |
| 4 | 0.8044 | 0.3855 | 0.1596 | 0.7488 | 0.3704 |
| 5 | 0.9367 | 0.6979 | 0.1226 | 0.8177 | 0.5195 |
| **Mean** | **0.8204** | **0.4173** | **0.1789** | **0.7257** | **0.3862** |
| **±Std** | 0.0710 | 0.1535 | 0.0374 | 0.0712 | 0.0692 |

CrossAttn best val AUC per fold: Fold1=0.8389, Fold2=0.7163, Fold3=0.8056, Fold4=0.8044, Fold5=0.9367

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6946 | 0.2337 | 0.2028 | 0.7176 | 0.3077 |
| CrossAttn | 0.8298 | 0.3919 | 0.1833 | 0.7216 | 0.3826 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6714 | 0.3120 | 0.2430 | 0.6364 | 0.3333 |
| F | 156 | 0.6934 | 0.2006 | 0.1774 | 0.7692 | 0.2800 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7920 | 0.4510 | 0.2363 | 0.6566 | 0.4516 |
| F | 156 | 0.8508 | 0.3472 | 0.1497 | 0.7628 | 0.3019 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 60 |
| **True: Sarco**  | 12 | 16 |

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
