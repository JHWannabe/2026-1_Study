# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 16:47  |  5-Fold CV  |  Median best epoch: 3

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
| 1 | 0.8064 | 0.4826 | 0.1694 | 0.7500 | 0.3200 |
| 2 | 0.6661 | 0.2384 | 0.1947 | 0.7353 | 0.3077 |
| 3 | 0.7280 | 0.2547 | 0.2036 | 0.6700 | 0.2947 |
| 4 | 0.7906 | 0.3421 | 0.1779 | 0.7734 | 0.4103 |
| 5 | 0.8420 | 0.4384 | 0.1699 | 0.7635 | 0.4286 |
| **Mean** | **0.7666** | **0.3513** | **0.1831** | **0.7384** | **0.3523** |
| **±Std** | 0.0624 | 0.0970 | 0.0138 | 0.0366 | 0.0557 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8002 | 0.3704 | 0.2031 | 0.7500 | 0.3377 |
| 2 | 0.6928 | 0.2182 | 0.2190 | 0.6667 | 0.2917 |
| 3 | 0.7996 | 0.3253 | 0.2150 | 0.6256 | 0.3214 |
| 4 | 0.7557 | 0.3825 | 0.1440 | 0.7931 | 0.2759 |
| 5 | 0.8985 | 0.6286 | 0.1624 | 0.7044 | 0.4000 |
| **Mean** | **0.7894** | **0.3850** | **0.1887** | **0.7080** | **0.3253** |
| **±Std** | 0.0672 | 0.1348 | 0.0300 | 0.0592 | 0.0432 |

CrossAttn best val AUC per fold: Fold1=0.8002, Fold2=0.6928, Fold3=0.7996, Fold4=0.7557, Fold5=0.8985

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8150 | 0.2996 | 0.1907 | 0.7216 | 0.3826 |
| CrossAttn | 0.7874 | 0.3362 | 0.2104 | 0.6118 | 0.2878 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7468 | 0.3147 | 0.2755 | 0.5758 | 0.4167 |
| F | 156 | 0.8270 | 0.3167 | 0.1370 | 0.8141 | 0.3256 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7604 | 0.3786 | 0.2416 | 0.5354 | 0.3611 |
| F | 156 | 0.7661 | 0.2967 | 0.1905 | 0.6603 | 0.2090 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 162 | 65 |
| **True: Sarco**  | 6 | 22 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 136 | 91 |
| **True: Sarco**  | 8 | 20 |

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
