# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:20  |  5-Fold CV  |  Median best epoch: 4

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
| 1 | 0.8119 | 0.4452 | 0.1705 | 0.7500 | 0.3200 |
| 2 | 0.6803 | 0.2445 | 0.1917 | 0.7353 | 0.2895 |
| 3 | 0.7366 | 0.2175 | 0.2040 | 0.6798 | 0.3158 |
| 4 | 0.7958 | 0.4003 | 0.1734 | 0.7389 | 0.3765 |
| 5 | 0.8453 | 0.4929 | 0.1737 | 0.7586 | 0.4235 |
| **Mean** | **0.7740** | **0.3601** | **0.1827** | **0.7325** | **0.3451** |
| **±Std** | 0.0586 | 0.1097 | 0.0130 | 0.0276 | 0.0484 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8037 | 0.3502 | 0.2128 | 0.6520 | 0.3364 |
| 2 | 0.7002 | 0.2142 | 0.1788 | 0.7010 | 0.2824 |
| 3 | 0.7820 | 0.3141 | 0.2177 | 0.6158 | 0.3158 |
| 4 | 0.7682 | 0.3659 | 0.2127 | 0.6650 | 0.3333 |
| 5 | 0.8920 | 0.6003 | 0.1416 | 0.8079 | 0.4935 |
| **Mean** | **0.7892** | **0.3689** | **0.1927** | **0.6883** | **0.3523** |
| **±Std** | 0.0620 | 0.1271 | 0.0291 | 0.0657 | 0.0732 |

CrossAttn best val AUC per fold: Fold1=0.8037, Fold2=0.7002, Fold3=0.7820, Fold4=0.7682, Fold5=0.8920

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8238 | 0.3043 | 0.1866 | 0.7216 | 0.3826 |
| CrossAttn | 0.7859 | 0.3067 | 0.1409 | 0.8118 | 0.4146 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7461 | 0.3017 | 0.2712 | 0.5960 | 0.4286 |
| F | 156 | 0.8414 | 0.3429 | 0.1329 | 0.8013 | 0.3111 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7554 | 0.3634 | 0.1802 | 0.7273 | 0.4706 |
| F | 156 | 0.7755 | 0.2597 | 0.1160 | 0.8654 | 0.3226 |

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
| **True: Normal** | 190 | 37 |
| **True: Sarco**  | 11 | 17 |

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
