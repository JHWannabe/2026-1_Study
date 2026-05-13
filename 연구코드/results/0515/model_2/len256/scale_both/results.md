# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 17:06  |  5-Fold CV  |  Median best epoch: 6

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
| 1 | 0.7275 | 0.2489 | 0.1761 | 0.7500 | 0.3200 |
| 2 | 0.6635 | 0.2747 | 0.1899 | 0.7304 | 0.2466 |
| 3 | 0.7401 | 0.2183 | 0.2022 | 0.6749 | 0.3125 |
| 4 | 0.8275 | 0.4662 | 0.1655 | 0.7537 | 0.3902 |
| 5 | 0.8491 | 0.4777 | 0.1818 | 0.6946 | 0.3542 |
| **Mean** | **0.7615** | **0.3372** | **0.1831** | **0.7207** | **0.3247** |
| **±Std** | 0.0682 | 0.1116 | 0.0124 | 0.0311 | 0.0478 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8107 | 0.3079 | 0.2026 | 0.6765 | 0.3654 |
| 2 | 0.7026 | 0.2078 | 0.1914 | 0.7059 | 0.3182 |
| 3 | 0.8091 | 0.3788 | 0.1731 | 0.6847 | 0.3469 |
| 4 | 0.8199 | 0.4476 | 0.2238 | 0.6404 | 0.3303 |
| 5 | 0.8905 | 0.6216 | 0.1614 | 0.7291 | 0.4330 |
| **Mean** | **0.8066** | **0.3928** | **0.1905** | **0.6873** | **0.3588** |
| **±Std** | 0.0601 | 0.1393 | 0.0219 | 0.0297 | 0.0404 |

CrossAttn best val AUC per fold: Fold1=0.8107, Fold2=0.7026, Fold3=0.8091, Fold4=0.8199, Fold5=0.8905

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8029 | 0.3056 | 0.1843 | 0.7294 | 0.3670 |
| CrossAttn | 0.8430 | 0.3894 | 0.2404 | 0.6196 | 0.3576 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7446 | 0.3225 | 0.2514 | 0.6061 | 0.4179 |
| F | 156 | 0.8025 | 0.3677 | 0.1418 | 0.8077 | 0.2857 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7970 | 0.4101 | 0.2970 | 0.5455 | 0.4304 |
| F | 156 | 0.8282 | 0.4218 | 0.2045 | 0.6667 | 0.2778 |

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
| **True: Normal** | 131 | 96 |
| **True: Sarco**  | 1 | 27 |

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
