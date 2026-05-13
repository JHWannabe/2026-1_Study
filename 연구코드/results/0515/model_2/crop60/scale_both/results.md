# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:21  |  5-Fold CV  |  Median best epoch: 17

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
| 1 | 0.7880 | 0.3092 | 0.1760 | 0.7549 | 0.3421 |
| 2 | 0.6707 | 0.2694 | 0.1887 | 0.7304 | 0.2254 |
| 3 | 0.7592 | 0.2285 | 0.1997 | 0.7044 | 0.3617 |
| 4 | 0.8310 | 0.4859 | 0.1664 | 0.7488 | 0.3855 |
| 5 | 0.8641 | 0.5087 | 0.1787 | 0.7094 | 0.3918 |
| **Mean** | **0.7826** | **0.3603** | **0.1819** | **0.7296** | **0.3413** |
| **±Std** | 0.0665 | 0.1149 | 0.0114 | 0.0203 | 0.0606 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8252 | 0.4096 | 0.1603 | 0.7549 | 0.3590 |
| 2 | 0.7173 | 0.2493 | 0.1994 | 0.6667 | 0.3061 |
| 3 | 0.8104 | 0.3373 | 0.2327 | 0.6059 | 0.3333 |
| 4 | 0.8184 | 0.4133 | 0.1422 | 0.7833 | 0.4054 |
| 5 | 0.8983 | 0.5521 | 0.2270 | 0.6256 | 0.3559 |
| **Mean** | **0.8139** | **0.3923** | **0.1923** | **0.6873** | **0.3520** |
| **±Std** | 0.0577 | 0.0998 | 0.0358 | 0.0702 | 0.0328 |

CrossAttn best val AUC per fold: Fold1=0.8252, Fold2=0.7173, Fold3=0.8104, Fold4=0.8184, Fold5=0.8983

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8060 | 0.3057 | 0.1875 | 0.7294 | 0.3894 |
| CrossAttn | 0.8315 | 0.3683 | 0.1957 | 0.7020 | 0.3871 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7439 | 0.3192 | 0.2551 | 0.6465 | 0.4776 |
| F | 156 | 0.8094 | 0.3631 | 0.1446 | 0.7821 | 0.2609 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7956 | 0.4331 | 0.2354 | 0.6364 | 0.4545 |
| F | 156 | 0.8313 | 0.3129 | 0.1705 | 0.7436 | 0.3103 |

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
| **True: Normal** | 155 | 72 |
| **True: Sarco**  | 4 | 24 |

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
