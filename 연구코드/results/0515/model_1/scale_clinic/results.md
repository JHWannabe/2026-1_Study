# SMI Binary Classification — Results

Generated: 2026-05-13 17:44  |  5-Fold CV  |  ResNet1D median best epoch: 33

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
| 1 | 0.8137 | 0.3911 | 0.1828 | 0.7598 | 0.3951 |
| 2 | 0.6623 | 0.2318 | 0.2013 | 0.7157 | 0.2750 |
| 3 | 0.7931 | 0.3106 | 0.1959 | 0.6995 | 0.3441 |
| 4 | 0.8101 | 0.4131 | 0.1716 | 0.7192 | 0.3448 |
| 5 | 0.8767 | 0.5216 | 0.1872 | 0.7143 | 0.3958 |
| **Mean** | **0.7912** | **0.3737** | **0.1877** | **0.7217** | **0.3510** |
| **±Std** | 0.0704 | 0.0978 | 0.0104 | 0.0202 | 0.0443 |

### ResNet1D

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8092 | 0.3884 | 0.1710 | 0.7843 | 0.3714 |
| 2 | 0.7122 | 0.2745 | 0.1916 | 0.6618 | 0.3301 |
| 3 | 0.7918 | 0.3117 | 0.1949 | 0.6650 | 0.3462 |
| 4 | 0.7953 | 0.4185 | 0.1691 | 0.7931 | 0.4324 |
| 5 | 0.9161 | 0.5300 | 0.1911 | 0.7389 | 0.4301 |
| **Mean** | **0.8049** | **0.3846** | **0.1835** | **0.7286** | **0.3820** |
| **±Std** | 0.0652 | 0.0892 | 0.0111 | 0.0564 | 0.0423 |

ResNet1D best val AUC per fold: Fold1=0.8092, Fold2=0.7122, Fold3=0.7918, Fold4=0.7953, Fold5=0.9161

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8291 | 0.3249 | 0.1931 | 0.6980 | 0.3636 |
| ResNet1D  | 0.7914 | 0.3599 | 0.1748 | 0.7765 | 0.4242 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7669 | 0.3155 | 0.2668 | 0.5859 | 0.4225 |
| F | 156 | 0.8345 | 0.4171 | 0.1464 | 0.7692 | 0.2800 |

#### ResNet1D

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7123 | 0.3233 | 0.2361 | 0.6667 | 0.4590 |
| F | 156 | 0.8476 | 0.4160 | 0.1359 | 0.8462 | 0.3684 |

---

## 3. Confusion Matrices (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 156 | 71 |
| **True: Sarco**  | 6 | 22 |

### ResNet1D

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 177 | 50 |
| **True: Sarco**  | 7 | 21 |

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
