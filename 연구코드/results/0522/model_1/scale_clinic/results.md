# SMI Binary Classification — Results

Generated: 2026-05-14 18:08  |  5-Fold CV  |  ResNet1D median best epoch: 77

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 355 | 292 | 82.3% | 63 | 17.7% |
| Train | F | 661 | 614 | 92.9% | 47 | 7.1% |
| Train | **All** | **1016** | **906** | **89.2%** | **110** | **10.8%** |
| Test | M | 97 | 83 | 85.6% | 14 | 14.4% |
| Test | F | 158 | 145 | 91.8% | 13 | 8.2% |
| Test | **All** | **255** | **228** | **89.4%** | **27** | **10.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 355 | 59.92 ± 12.67 | 18.00 | 60.00 | 89.00 |
| Train | F | 661 | 55.55 ± 11.94 | 18.00 | 55.00 | 91.00 |
| Train | **All** | **1016** | **57.07 ± 12.38** | **18.00** | **57.00** | **91.00** |
| Test | M | 97 | 58.63 ± 12.43 | 28.00 | 59.00 | 88.00 |
| Test | F | 158 | 55.27 ± 11.46 | 23.00 | 56.00 | 86.00 |
| Test | **All** | **255** | **56.55 ± 11.95** | **23.00** | **57.00** | **88.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 355 | 24.22 ± 3.38 | 14.48 | 24.16 | 36.76 |
| Train | F | 661 | 23.14 ± 3.39 | 14.40 | 22.83 | 36.24 |
| Train | **All** | **1016** | **23.52 ± 3.42** | **14.40** | **23.37** | **36.76** |
| Test | M | 97 | 24.50 ± 3.14 | 18.37 | 24.49 | 35.68 |
| Test | F | 158 | 23.11 ± 3.24 | 16.87 | 22.72 | 34.23 |
| Test | **All** | **255** | **23.64 ± 3.27** | **16.87** | **23.34** | **35.68** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7895 | 0.3748 | 0.1789 | 0.7304 | 0.3529 |
| 2 | 0.8350 | 0.4191 | 0.1590 | 0.7586 | 0.3951 |
| 3 | 0.8222 | 0.3207 | 0.1665 | 0.7537 | 0.3750 |
| 4 | 0.8262 | 0.4497 | 0.1907 | 0.7241 | 0.3778 |
| 5 | 0.8079 | 0.3249 | 0.1858 | 0.7241 | 0.3778 |
| **Mean** | **0.8162** | **0.3779** | **0.1762** | **0.7382** | **0.3757** |
| **±Std** | 0.0160 | 0.0509 | 0.0118 | 0.0149 | 0.0134 |

### ResNet1D

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8492 | 0.5552 | 0.1514 | 0.8186 | 0.4789 |
| 2 | 0.8672 | 0.5010 | 0.1568 | 0.7488 | 0.4270 |
| 3 | 0.8076 | 0.4183 | 0.1422 | 0.7980 | 0.3692 |
| 4 | 0.8792 | 0.4684 | 0.1804 | 0.7833 | 0.4500 |
| 5 | 0.8322 | 0.3177 | 0.1734 | 0.7685 | 0.4471 |
| **Mean** | **0.8471** | **0.4521** | **0.1609** | **0.7834** | **0.4344** |
| **±Std** | 0.0254 | 0.0806 | 0.0141 | 0.0240 | 0.0366 |

ResNet1D best val AUC per fold: Fold1=0.8492, Fold2=0.8672, Fold3=0.8076, Fold4=0.8792, Fold5=0.8322

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7497 | 0.2647 | 0.1848 | 0.7490 | 0.3043 |
| ResNet1D  | 0.7268 | 0.2527 | 0.1740 | 0.7843 | 0.3373 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7814 | 0.3556 | 0.2456 | 0.6495 | 0.3929 |
| F | 158 | 0.7093 | 0.1623 | 0.1475 | 0.8101 | 0.1667 |

#### ResNet1D

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7108 | 0.3039 | 0.2298 | 0.7010 | 0.4314 |
| F | 158 | 0.7135 | 0.1946 | 0.1398 | 0.8354 | 0.1875 |

---

## 3. Confusion Matrices (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 177 | 51 |
| **True: Sarco**  | 13 | 14 |

### ResNet1D

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 186 | 42 |
| **True: Sarco**  | 13 | 14 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR & ResNet1D) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `confusion_matrices.png` | Test-set confusion matrices (overall + by sex) |
| `training_curves.png` | ResNet1D loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curves (overall) |
| `test_roc_by_sex.png` | Final test-set ROC curves split by sex |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
