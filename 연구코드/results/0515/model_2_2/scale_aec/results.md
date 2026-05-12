# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 16:58  |  5-Fold CV  |  Median best epoch: 111

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 402 | 332 | 82.6% | 70 | 17.4% |
| Train | F | 695 | 645 | 92.8% | 50 | 7.2% |
| Train | **All** | **1097** | **977** | **89.1%** | **120** | **10.9%** |
| Test | M | 112 | 95 | 84.8% | 17 | 15.2% |
| Test | F | 163 | 150 | 92.0% | 13 | 8.0% |
| Test | **All** | **275** | **245** | **89.1%** | **30** | **10.9%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 402 | 59.81 ± 12.51 | 18.00 | 60.00 | 89.00 |
| Train | F | 695 | 55.36 ± 12.15 | 11.00 | 55.00 | 91.00 |
| Train | **All** | **1097** | **56.99 ± 12.47** | **11.00** | **58.00** | **91.00** |
| Test | M | 112 | 59.05 ± 12.52 | 23.00 | 59.50 | 84.00 |
| Test | F | 163 | 56.52 ± 12.29 | 22.00 | 56.00 | 87.00 |
| Test | **All** | **275** | **57.55 ± 12.45** | **22.00** | **58.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 402 | 24.22 ± 3.26 | 14.48 | 24.19 | 36.76 |
| Train | F | 695 | 23.09 ± 3.43 | 14.40 | 22.70 | 39.49 |
| Train | **All** | **1097** | **23.51 ± 3.41** | **14.40** | **23.30** | **39.49** |
| Test | M | 112 | 24.07 ± 3.30 | 16.44 | 24.16 | 35.20 |
| Test | F | 163 | 22.99 ± 3.19 | 16.06 | 22.83 | 34.23 |
| Test | **All** | **275** | **23.43 ± 3.28** | **16.06** | **23.44** | **35.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7966 | 0.2953 | 0.1975 | 0.6864 | 0.3429 |
| 2 | 0.8112 | 0.3142 | 0.1480 | 0.7955 | 0.3662 |
| 3 | 0.7889 | 0.3590 | 0.1944 | 0.7260 | 0.3750 |
| 4 | 0.7100 | 0.2108 | 0.1751 | 0.7397 | 0.2785 |
| 5 | 0.8175 | 0.4430 | 0.1703 | 0.7489 | 0.4086 |
| **Mean** | **0.7848** | **0.3245** | **0.1771** | **0.7393** | **0.3542** |
| **±Std** | 0.0388 | 0.0763 | 0.0180 | 0.0353 | 0.0434 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7538 | 0.3457 | 0.1908 | 0.7409 | 0.3448 |
| 2 | 0.7844 | 0.3426 | 0.2085 | 0.7091 | 0.3333 |
| 3 | 0.7816 | 0.2703 | 0.1841 | 0.7169 | 0.3542 |
| 4 | 0.7491 | 0.2696 | 0.1452 | 0.8311 | 0.3509 |
| 5 | 0.8534 | 0.5885 | 0.1562 | 0.7626 | 0.4222 |
| **Mean** | **0.7845** | **0.3634** | **0.1769** | **0.7521** | **0.3611** |
| **±Std** | 0.0373 | 0.1174 | 0.0231 | 0.0437 | 0.0314 |

CrossAttn best val AUC per fold: Fold1=0.7538, Fold2=0.7844, Fold3=0.7816, Fold4=0.7491, Fold5=0.8534

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7261 | 0.2982 | 0.1919 | 0.7236 | 0.3559 |
| CrossAttn | 0.6977 | 0.3071 | 0.1793 | 0.7600 | 0.3265 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.6941 | 0.4248 | 0.2628 | 0.6071 | 0.3714 |
| F | 163 | 0.7292 | 0.2275 | 0.1432 | 0.8037 | 0.3333 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.6830 | 0.3370 | 0.2507 | 0.6339 | 0.3279 |
| F | 163 | 0.6441 | 0.3008 | 0.1302 | 0.8466 | 0.3243 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 178 | 67 |
| **True: Sarco**  | 9 | 21 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 193 | 52 |
| **True: Sarco**  | 14 | 16 |

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
