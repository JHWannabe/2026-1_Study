# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 17:26  |  5-Fold CV  |  Median best epoch: 5

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
| 1 | 0.7580 | 0.2595 | 0.1790 | 0.7402 | 0.3117 |
| 2 | 0.6760 | 0.2707 | 0.1896 | 0.7304 | 0.2466 |
| 3 | 0.7546 | 0.2255 | 0.1976 | 0.7143 | 0.3556 |
| 4 | 0.8307 | 0.4631 | 0.1662 | 0.7783 | 0.4156 |
| 5 | 0.8317 | 0.5287 | 0.1757 | 0.7192 | 0.3596 |
| **Mean** | **0.7702** | **0.3495** | **0.1816** | **0.7365** | **0.3378** |
| **±Std** | 0.0578 | 0.1222 | 0.0109 | 0.0228 | 0.0563 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8257 | 0.3982 | 0.1773 | 0.7304 | 0.3956 |
| 2 | 0.6933 | 0.2790 | 0.2299 | 0.6471 | 0.2800 |
| 3 | 0.7898 | 0.3319 | 0.1941 | 0.7143 | 0.3830 |
| 4 | 0.8147 | 0.4425 | 0.1427 | 0.7882 | 0.4110 |
| 5 | 0.9053 | 0.5912 | 0.1721 | 0.6897 | 0.4000 |
| **Mean** | **0.8057** | **0.4086** | **0.1832** | **0.7139** | **0.3739** |
| **±Std** | 0.0683 | 0.1070 | 0.0286 | 0.0466 | 0.0478 |

CrossAttn best val AUC per fold: Fold1=0.8257, Fold2=0.6933, Fold3=0.7898, Fold4=0.8147, Fold5=0.9053

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8052 | 0.3185 | 0.1863 | 0.7333 | 0.3929 |
| CrossAttn | 0.8107 | 0.3525 | 0.1935 | 0.6980 | 0.3304 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7403 | 0.3409 | 0.2571 | 0.6263 | 0.4478 |
| F | 156 | 0.8107 | 0.3817 | 0.1414 | 0.8013 | 0.3111 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7568 | 0.3912 | 0.2301 | 0.6263 | 0.3729 |
| F | 156 | 0.8213 | 0.3792 | 0.1703 | 0.7436 | 0.2857 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 165 | 62 |
| **True: Sarco**  | 6 | 22 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 159 | 68 |
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
