# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 16:38  |  5-Fold CV  |  Median best epoch: 3

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
| 1 | 0.6266 | 0.2256 | 0.1889 | 0.7402 | 0.2740 |
| 2 | 0.5753 | 0.1371 | 0.2061 | 0.6912 | 0.1370 |
| 3 | 0.5500 | 0.1481 | 0.2601 | 0.6502 | 0.2526 |
| 4 | 0.5060 | 0.1321 | 0.2296 | 0.6946 | 0.2250 |
| 5 | 0.6258 | 0.2318 | 0.2163 | 0.6847 | 0.2195 |
| **Mean** | **0.5767** | **0.1750** | **0.2202** | **0.6922** | **0.2216** |
| **±Std** | 0.0461 | 0.0442 | 0.0240 | 0.0287 | 0.0467 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8127 | 0.3695 | 0.2056 | 0.6716 | 0.3495 |
| 2 | 0.6832 | 0.2195 | 0.2136 | 0.6618 | 0.2887 |
| 3 | 0.7983 | 0.3079 | 0.2299 | 0.6355 | 0.3273 |
| 4 | 0.7785 | 0.4215 | 0.2114 | 0.6749 | 0.3265 |
| 5 | 0.9048 | 0.5563 | 0.1426 | 0.7537 | 0.4444 |
| **Mean** | **0.7955** | **0.3749** | **0.2006** | **0.6795** | **0.3473** |
| **±Std** | 0.0709 | 0.1129 | 0.0301 | 0.0396 | 0.0524 |

CrossAttn best val AUC per fold: Fold1=0.8127, Fold2=0.6832, Fold3=0.7983, Fold4=0.7785, Fold5=0.9048

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6946 | 0.2294 | 0.2174 | 0.6745 | 0.2783 |
| CrossAttn | 0.7967 | 0.3055 | 0.2369 | 0.6078 | 0.3333 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.5488 | 0.2304 | 0.2877 | 0.5354 | 0.2069 |
| F | 156 | 0.8414 | 0.2969 | 0.1729 | 0.7628 | 0.3509 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7575 | 0.3177 | 0.2788 | 0.5455 | 0.4156 |
| F | 156 | 0.7693 | 0.4058 | 0.2103 | 0.6474 | 0.2466 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 156 | 71 |
| **True: Sarco**  | 12 | 16 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 130 | 97 |
| **True: Sarco**  | 3 | 25 |

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
