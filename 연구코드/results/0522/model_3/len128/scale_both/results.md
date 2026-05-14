# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:41  |  5-Fold CV  |  Median best epoch: 11

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
| 1 | 0.7927 | 0.4208 | 0.1661 | 0.7353 | 0.3571 |
| 2 | 0.7953 | 0.3369 | 0.1636 | 0.7635 | 0.3846 |
| 3 | 0.8071 | 0.3017 | 0.1618 | 0.7833 | 0.3889 |
| 4 | 0.7936 | 0.4241 | 0.1851 | 0.7241 | 0.3636 |
| 5 | 0.8174 | 0.3824 | 0.1719 | 0.7734 | 0.4103 |
| **Mean** | **0.8012** | **0.3732** | **0.1697** | **0.7559** | **0.3809** |
| **±Std** | 0.0096 | 0.0477 | 0.0084 | 0.0226 | 0.0190 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8549 | 0.4585 | 0.1512 | 0.7794 | 0.4304 |
| 2 | 0.8438 | 0.3906 | 0.1705 | 0.7537 | 0.4186 |
| 3 | 0.8179 | 0.3560 | 0.1513 | 0.7783 | 0.3836 |
| 4 | 0.8744 | 0.6239 | 0.1959 | 0.7291 | 0.4211 |
| 5 | 0.8458 | 0.3776 | 0.1104 | 0.8374 | 0.4762 |
| **Mean** | **0.8474** | **0.4413** | **0.1559** | **0.7756** | **0.4260** |
| **±Std** | 0.0183 | 0.0975 | 0.0280 | 0.0360 | 0.0297 |

CrossAttn best val AUC per fold: Fold1=0.8549, Fold2=0.8438, Fold3=0.8179, Fold4=0.8744, Fold5=0.8458

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7336 | 0.2732 | 0.1778 | 0.7529 | 0.3077 |
| CrossAttn | 0.7047 | 0.2126 | 0.1730 | 0.7412 | 0.2500 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7590 | 0.3696 | 0.2302 | 0.6701 | 0.3846 |
| F | 158 | 0.6727 | 0.1668 | 0.1457 | 0.8038 | 0.2051 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6910 | 0.2440 | 0.2469 | 0.6186 | 0.2745 |
| F | 158 | 0.6753 | 0.2190 | 0.1276 | 0.8165 | 0.2162 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 178 | 50 |
| **True: Sarco**  | 13 | 14 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 178 | 50 |
| **True: Sarco**  | 16 | 11 |

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
