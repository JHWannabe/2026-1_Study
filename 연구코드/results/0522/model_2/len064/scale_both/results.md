# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:22  |  5-Fold CV  |  Median best epoch: 9

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
| 1 | 0.7950 | 0.4279 | 0.1671 | 0.7549 | 0.3590 |
| 2 | 0.7943 | 0.3506 | 0.1631 | 0.7586 | 0.3797 |
| 3 | 0.8217 | 0.3079 | 0.1600 | 0.7734 | 0.3784 |
| 4 | 0.7981 | 0.4202 | 0.1887 | 0.6946 | 0.3404 |
| 5 | 0.8117 | 0.3833 | 0.1755 | 0.7635 | 0.4000 |
| **Mean** | **0.8041** | **0.3780** | **0.1709** | **0.7490** | **0.3715** |
| **±Std** | 0.0108 | 0.0446 | 0.0103 | 0.0279 | 0.0202 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8544 | 0.5087 | 0.1610 | 0.7402 | 0.3765 |
| 2 | 0.8393 | 0.3991 | 0.1693 | 0.7241 | 0.3913 |
| 3 | 0.7888 | 0.3213 | 0.1502 | 0.7783 | 0.3836 |
| 4 | 0.8664 | 0.6010 | 0.2538 | 0.5764 | 0.3065 |
| 5 | 0.8245 | 0.3571 | 0.2253 | 0.6847 | 0.3725 |
| **Mean** | **0.8347** | **0.4374** | **0.1919** | **0.7007** | **0.3661** |
| **±Std** | 0.0269 | 0.1032 | 0.0404 | 0.0691 | 0.0305 |

CrossAttn best val AUC per fold: Fold1=0.8544, Fold2=0.8393, Fold3=0.7888, Fold4=0.8664, Fold5=0.8245

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7313 | 0.2543 | 0.1771 | 0.7490 | 0.2889 |
| CrossAttn | 0.7635 | 0.2771 | 0.1790 | 0.7255 | 0.3137 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7556 | 0.3415 | 0.2288 | 0.6598 | 0.3774 |
| F | 158 | 0.6684 | 0.1499 | 0.1454 | 0.8038 | 0.1622 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7608 | 0.3605 | 0.2374 | 0.6289 | 0.3793 |
| F | 158 | 0.7528 | 0.1948 | 0.1431 | 0.7848 | 0.2273 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 178 | 50 |
| **True: Sarco**  | 14 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 169 | 59 |
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
