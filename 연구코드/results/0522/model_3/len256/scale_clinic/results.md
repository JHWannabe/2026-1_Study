# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:50  |  5-Fold CV  |  Median best epoch: 8

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
| 1 | 0.5769 | 0.1704 | 0.2451 | 0.6765 | 0.2143 |
| 2 | 0.5311 | 0.1166 | 0.2324 | 0.7192 | 0.1493 |
| 3 | 0.6730 | 0.1808 | 0.1990 | 0.7143 | 0.2162 |
| 4 | 0.6414 | 0.2454 | 0.1709 | 0.7586 | 0.3099 |
| 5 | 0.5987 | 0.2357 | 0.2152 | 0.6897 | 0.2588 |
| **Mean** | **0.6042** | **0.1898** | **0.2125** | **0.7116** | **0.2297** |
| **±Std** | 0.0495 | 0.0469 | 0.0260 | 0.0283 | 0.0532 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8354 | 0.5108 | 0.1394 | 0.7892 | 0.4416 |
| 2 | 0.8466 | 0.3939 | 0.1419 | 0.8079 | 0.4507 |
| 3 | 0.8202 | 0.3797 | 0.1327 | 0.7931 | 0.3636 |
| 4 | 0.8538 | 0.5635 | 0.1071 | 0.8670 | 0.5574 |
| 5 | 0.8338 | 0.3528 | 0.1712 | 0.7537 | 0.4318 |
| **Mean** | **0.8380** | **0.4401** | **0.1385** | **0.8022** | **0.4490** |
| **±Std** | 0.0115 | 0.0820 | 0.0205 | 0.0370 | 0.0623 |

CrossAttn best val AUC per fold: Fold1=0.8354, Fold2=0.8466, Fold3=0.8202, Fold4=0.8538, Fold5=0.8338

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5318 | 0.1779 | 0.2320 | 0.6824 | 0.1980 |
| CrossAttn | 0.7424 | 0.2571 | 0.1790 | 0.7176 | 0.2653 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.4802 | 0.2130 | 0.2849 | 0.6186 | 0.2128 |
| F | 158 | 0.5814 | 0.1855 | 0.1995 | 0.7215 | 0.1852 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7513 | 0.3362 | 0.2476 | 0.6082 | 0.3214 |
| F | 158 | 0.7088 | 0.1609 | 0.1368 | 0.7848 | 0.1905 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 164 | 64 |
| **True: Sarco**  | 17 | 10 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 170 | 58 |
| **True: Sarco**  | 14 | 13 |

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
