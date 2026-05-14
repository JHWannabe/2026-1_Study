# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:14  |  5-Fold CV  |  Median best epoch: 11

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
| 1 | 0.7043 | 0.2173 | 0.2142 | 0.7157 | 0.3409 |
| 2 | 0.6512 | 0.1736 | 0.2130 | 0.7094 | 0.2532 |
| 3 | 0.7315 | 0.2447 | 0.1900 | 0.7241 | 0.3171 |
| 4 | 0.6341 | 0.2081 | 0.2028 | 0.7389 | 0.3117 |
| 5 | 0.6888 | 0.2961 | 0.1985 | 0.6946 | 0.2791 |
| **Mean** | **0.6820** | **0.2280** | **0.2037** | **0.7165** | **0.3004** |
| **±Std** | 0.0353 | 0.0410 | 0.0091 | 0.0148 | 0.0308 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8544 | 0.5076 | 0.1675 | 0.7892 | 0.4557 |
| 2 | 0.8483 | 0.3639 | 0.1538 | 0.7734 | 0.4390 |
| 3 | 0.8453 | 0.3775 | 0.1699 | 0.7833 | 0.4359 |
| 4 | 0.8757 | 0.6101 | 0.1494 | 0.7783 | 0.4578 |
| 5 | 0.8295 | 0.3544 | 0.1568 | 0.7734 | 0.4250 |
| **Mean** | **0.8506** | **0.4427** | **0.1595** | **0.7795** | **0.4427** |
| **±Std** | 0.0150 | 0.1005 | 0.0079 | 0.0061 | 0.0124 |

CrossAttn best val AUC per fold: Fold1=0.8544, Fold2=0.8483, Fold3=0.8453, Fold4=0.8757, Fold5=0.8295

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6181 | 0.1858 | 0.2189 | 0.6667 | 0.2202 |
| CrossAttn | 0.7370 | 0.3054 | 0.1781 | 0.7608 | 0.3441 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6213 | 0.2583 | 0.2547 | 0.5979 | 0.2642 |
| F | 158 | 0.5936 | 0.1250 | 0.1969 | 0.7089 | 0.1786 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7573 | 0.4065 | 0.2593 | 0.6289 | 0.4000 |
| F | 158 | 0.6743 | 0.2014 | 0.1283 | 0.8418 | 0.2424 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 158 | 70 |
| **True: Sarco**  | 15 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 178 | 50 |
| **True: Sarco**  | 11 | 16 |

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
