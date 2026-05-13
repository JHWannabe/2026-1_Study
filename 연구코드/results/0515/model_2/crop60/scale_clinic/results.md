# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:11  |  5-Fold CV  |  Median best epoch: 7

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
| 1 | 0.6319 | 0.1931 | 0.1997 | 0.7108 | 0.2133 |
| 2 | 0.6003 | 0.1960 | 0.1788 | 0.7500 | 0.2388 |
| 3 | 0.6911 | 0.1804 | 0.2251 | 0.6995 | 0.2989 |
| 4 | 0.6723 | 0.2889 | 0.1990 | 0.7143 | 0.2750 |
| 5 | 0.6936 | 0.3034 | 0.1807 | 0.7241 | 0.2821 |
| **Mean** | **0.6578** | **0.2324** | **0.1967** | **0.7197** | **0.2616** |
| **±Std** | 0.0363 | 0.0525 | 0.0167 | 0.0171 | 0.0311 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8189 | 0.3619 | 0.1895 | 0.7304 | 0.3373 |
| 2 | 0.7036 | 0.2229 | 0.2123 | 0.6520 | 0.3107 |
| 3 | 0.7818 | 0.2835 | 0.2799 | 0.5517 | 0.2835 |
| 4 | 0.8021 | 0.4070 | 0.1706 | 0.7291 | 0.3820 |
| 5 | 0.9093 | 0.6461 | 0.1120 | 0.8621 | 0.5484 |
| **Mean** | **0.8031** | **0.3843** | **0.1929** | **0.7050** | **0.3724** |
| **±Std** | 0.0662 | 0.1454 | 0.0548 | 0.1022 | 0.0938 |

CrossAttn best val AUC per fold: Fold1=0.8189, Fold2=0.7036, Fold3=0.7818, Fold4=0.8021, Fold5=0.9093

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7409 | 0.2596 | 0.1941 | 0.7216 | 0.3107 |
| CrossAttn | 0.8195 | 0.3400 | 0.2390 | 0.6275 | 0.3448 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7231 | 0.3342 | 0.2177 | 0.6970 | 0.3750 |
| F | 156 | 0.7643 | 0.2330 | 0.1791 | 0.7372 | 0.2545 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7869 | 0.3687 | 0.2587 | 0.6061 | 0.4348 |
| F | 156 | 0.8201 | 0.3418 | 0.2265 | 0.6410 | 0.2632 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 59 |
| **True: Sarco**  | 12 | 16 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 135 | 92 |
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
