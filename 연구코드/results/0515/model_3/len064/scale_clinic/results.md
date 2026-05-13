# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 18:55  |  5-Fold CV  |  Median best epoch: 6

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
| 1 | 0.5809 | 0.2194 | 0.2061 | 0.7059 | 0.1892 |
| 2 | 0.6375 | 0.1944 | 0.2104 | 0.6765 | 0.2143 |
| 3 | 0.6439 | 0.2047 | 0.2157 | 0.6798 | 0.2857 |
| 4 | 0.7230 | 0.3383 | 0.1901 | 0.7340 | 0.3415 |
| 5 | 0.7255 | 0.2617 | 0.1934 | 0.6897 | 0.2921 |
| **Mean** | **0.6622** | **0.2437** | **0.2031** | **0.6972** | **0.2646** |
| **±Std** | 0.0552 | 0.0526 | 0.0099 | 0.0211 | 0.0554 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8159 | 0.4065 | 0.2135 | 0.6765 | 0.3265 |
| 2 | 0.6803 | 0.2514 | 0.2044 | 0.6814 | 0.2857 |
| 3 | 0.7720 | 0.2484 | 0.1813 | 0.7389 | 0.4045 |
| 4 | 0.8332 | 0.4303 | 0.1573 | 0.7685 | 0.4337 |
| 5 | 0.9191 | 0.6886 | 0.1571 | 0.7488 | 0.4396 |
| **Mean** | **0.8041** | **0.4051** | **0.1827** | **0.7228** | **0.3780** |
| **±Std** | 0.0782 | 0.1608 | 0.0233 | 0.0371 | 0.0613 |

CrossAttn best val AUC per fold: Fold1=0.8159, Fold2=0.6803, Fold3=0.7720, Fold4=0.8332, Fold5=0.9191

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7052 | 0.2353 | 0.2033 | 0.7216 | 0.2970 |
| CrossAttn | 0.8313 | 0.3689 | 0.1542 | 0.7765 | 0.4242 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6923 | 0.3149 | 0.2407 | 0.6465 | 0.3137 |
| F | 156 | 0.7009 | 0.2136 | 0.1796 | 0.7692 | 0.2800 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7812 | 0.4178 | 0.2029 | 0.7071 | 0.4912 |
| F | 156 | 0.8332 | 0.3904 | 0.1232 | 0.8205 | 0.3333 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 169 | 58 |
| **True: Sarco**  | 13 | 15 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 177 | 50 |
| **True: Sarco**  | 7 | 21 |

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
