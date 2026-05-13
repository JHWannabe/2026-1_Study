# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:39  |  5-Fold CV  |  Median best epoch: 3

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
| 1 | 0.6711 | 0.1865 | 0.1770 | 0.7647 | 0.2500 |
| 2 | 0.6375 | 0.2546 | 0.1926 | 0.7157 | 0.1944 |
| 3 | 0.7295 | 0.2086 | 0.2001 | 0.7340 | 0.3721 |
| 4 | 0.7885 | 0.3827 | 0.1742 | 0.7586 | 0.3797 |
| 5 | 0.7981 | 0.3400 | 0.1825 | 0.7241 | 0.3636 |
| **Mean** | **0.7250** | **0.2745** | **0.1853** | **0.7394** | **0.3120** |
| **±Std** | 0.0632 | 0.0755 | 0.0097 | 0.0192 | 0.0755 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8224 | 0.4111 | 0.1453 | 0.8039 | 0.4118 |
| 2 | 0.6860 | 0.2401 | 0.2482 | 0.5980 | 0.2931 |
| 3 | 0.8124 | 0.3292 | 0.2212 | 0.6355 | 0.3273 |
| 4 | 0.7777 | 0.4106 | 0.1639 | 0.7340 | 0.3571 |
| 5 | 0.8935 | 0.4877 | 0.1705 | 0.7340 | 0.4375 |
| **Mean** | **0.7984** | **0.3758** | **0.1898** | **0.7011** | **0.3654** |
| **±Std** | 0.0676 | 0.0843 | 0.0385 | 0.0744 | 0.0531 |

CrossAttn best val AUC per fold: Fold1=0.8224, Fold2=0.6860, Fold3=0.8124, Fold4=0.7777, Fold5=0.8935

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7469 | 0.2829 | 0.1866 | 0.7294 | 0.3429 |
| CrossAttn | 0.8021 | 0.3010 | 0.2031 | 0.6902 | 0.3577 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6879 | 0.3061 | 0.2593 | 0.5960 | 0.3548 |
| F | 156 | 0.7618 | 0.3026 | 0.1404 | 0.8141 | 0.3256 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7561 | 0.3317 | 0.2558 | 0.6566 | 0.4516 |
| F | 156 | 0.8157 | 0.4332 | 0.1696 | 0.7115 | 0.2623 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 59 |
| **True: Sarco**  | 10 | 18 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 154 | 73 |
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
