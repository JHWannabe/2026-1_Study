# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 17:24  |  5-Fold CV  |  Median best epoch: 8

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
| 1 | 0.7562 | 0.2614 | 0.1781 | 0.7451 | 0.3158 |
| 2 | 0.6736 | 0.2702 | 0.1881 | 0.7304 | 0.2254 |
| 3 | 0.7554 | 0.2286 | 0.1997 | 0.7241 | 0.3778 |
| 4 | 0.8267 | 0.4558 | 0.1678 | 0.7635 | 0.4146 |
| 5 | 0.8564 | 0.5208 | 0.1777 | 0.7192 | 0.4000 |
| **Mean** | **0.7737** | **0.3474** | **0.1823** | **0.7365** | **0.3467** |
| **±Std** | 0.0637 | 0.1177 | 0.0108 | 0.0161 | 0.0694 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8207 | 0.3374 | 0.1845 | 0.7549 | 0.4186 |
| 2 | 0.7077 | 0.2207 | 0.2033 | 0.6912 | 0.3226 |
| 3 | 0.8265 | 0.3412 | 0.2636 | 0.4926 | 0.2797 |
| 4 | 0.7996 | 0.3841 | 0.1091 | 0.8325 | 0.3462 |
| 5 | 0.9076 | 0.6441 | 0.1895 | 0.6897 | 0.4000 |
| **Mean** | **0.8124** | **0.3855** | **0.1900** | **0.6922** | **0.3534** |
| **±Std** | 0.0640 | 0.1402 | 0.0493 | 0.1127 | 0.0507 |

CrossAttn best val AUC per fold: Fold1=0.8207, Fold2=0.7077, Fold3=0.8265, Fold4=0.7996, Fold5=0.9076

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8005 | 0.2966 | 0.1860 | 0.7294 | 0.3894 |
| CrossAttn | 0.8365 | 0.3929 | 0.2476 | 0.6431 | 0.3636 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7367 | 0.3047 | 0.2544 | 0.6465 | 0.4615 |
| F | 156 | 0.8082 | 0.3769 | 0.1427 | 0.7821 | 0.2917 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.8099 | 0.4433 | 0.2741 | 0.6263 | 0.4789 |
| F | 156 | 0.8144 | 0.3599 | 0.2308 | 0.6538 | 0.2500 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 164 | 63 |
| **True: Sarco**  | 6 | 22 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 138 | 89 |
| **True: Sarco**  | 2 | 26 |

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
