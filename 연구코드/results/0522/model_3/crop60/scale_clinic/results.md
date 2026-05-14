# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:28  |  5-Fold CV  |  Median best epoch: 15

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
| 1 | 0.7717 | 0.2505 | 0.1774 | 0.7451 | 0.3500 |
| 2 | 0.6389 | 0.1642 | 0.1878 | 0.7685 | 0.2540 |
| 3 | 0.7685 | 0.2476 | 0.1871 | 0.7291 | 0.3210 |
| 4 | 0.6758 | 0.3038 | 0.1768 | 0.7685 | 0.3562 |
| 5 | 0.6256 | 0.2318 | 0.2144 | 0.6897 | 0.2588 |
| **Mean** | **0.6961** | **0.2396** | **0.1887** | **0.7402** | **0.3080** |
| **±Std** | 0.0626 | 0.0448 | 0.0137 | 0.0293 | 0.0438 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8442 | 0.3734 | 0.1309 | 0.8137 | 0.4571 |
| 2 | 0.8699 | 0.4987 | 0.1503 | 0.7537 | 0.4186 |
| 3 | 0.8091 | 0.3483 | 0.1748 | 0.7635 | 0.4000 |
| 4 | 0.8468 | 0.5197 | 0.1805 | 0.7044 | 0.4000 |
| 5 | 0.8518 | 0.4219 | 0.1858 | 0.7833 | 0.4762 |
| **Mean** | **0.8444** | **0.4324** | **0.1645** | **0.7637** | **0.4304** |
| **±Std** | 0.0198 | 0.0674 | 0.0207 | 0.0360 | 0.0310 |

CrossAttn best val AUC per fold: Fold1=0.8442, Fold2=0.8699, Fold3=0.8091, Fold4=0.8468, Fold5=0.8518

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6488 | 0.1930 | 0.2161 | 0.6863 | 0.2157 |
| CrossAttn | 0.6902 | 0.2412 | 0.1795 | 0.7529 | 0.3226 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6592 | 0.3117 | 0.2166 | 0.6907 | 0.3182 |
| F | 158 | 0.6382 | 0.1443 | 0.2158 | 0.6835 | 0.1379 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7160 | 0.3067 | 0.2563 | 0.6186 | 0.3729 |
| F | 158 | 0.5989 | 0.2046 | 0.1324 | 0.8354 | 0.2353 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 164 | 64 |
| **True: Sarco**  | 16 | 11 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 177 | 51 |
| **True: Sarco**  | 12 | 15 |

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
