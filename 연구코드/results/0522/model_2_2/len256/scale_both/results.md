# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:58  |  5-Fold CV  |  Median best epoch: 7

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
| 1 | 0.7465 | 0.2967 | 0.1601 | 0.7451 | 0.3158 |
| 2 | 0.8285 | 0.4313 | 0.1442 | 0.7931 | 0.4167 |
| 3 | 0.8749 | 0.3944 | 0.1462 | 0.7931 | 0.4615 |
| 4 | 0.8468 | 0.4210 | 0.1686 | 0.7734 | 0.4524 |
| 5 | 0.8117 | 0.3697 | 0.1721 | 0.7438 | 0.3953 |
| **Mean** | **0.8217** | **0.3826** | **0.1582** | **0.7697** | **0.4083** |
| **±Std** | 0.0430 | 0.0480 | 0.0114 | 0.0218 | 0.0521 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7962 | 0.3720 | 0.1702 | 0.7353 | 0.3721 |
| 2 | 0.8634 | 0.5083 | 0.1354 | 0.8128 | 0.4722 |
| 3 | 0.7993 | 0.3773 | 0.1892 | 0.7192 | 0.3294 |
| 4 | 0.8581 | 0.6369 | 0.2127 | 0.6650 | 0.3585 |
| 5 | 0.8546 | 0.4117 | 0.1588 | 0.7734 | 0.4524 |
| **Mean** | **0.8343** | **0.4612** | **0.1733** | **0.7411** | **0.3969** |
| **±Std** | 0.0300 | 0.1005 | 0.0263 | 0.0500 | 0.0555 |

CrossAttn best val AUC per fold: Fold1=0.7962, Fold2=0.8634, Fold3=0.7993, Fold4=0.8581, Fold5=0.8546

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6850 | 0.1884 | 0.1786 | 0.7569 | 0.2619 |
| CrossAttn | 0.7771 | 0.2733 | 0.2197 | 0.6902 | 0.3361 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7126 | 0.2540 | 0.2384 | 0.6701 | 0.3600 |
| F | 158 | 0.6249 | 0.1209 | 0.1418 | 0.8101 | 0.1176 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7745 | 0.3452 | 0.2716 | 0.6392 | 0.4068 |
| F | 158 | 0.7411 | 0.1852 | 0.1879 | 0.7215 | 0.2667 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 182 | 46 |
| **True: Sarco**  | 16 | 11 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 156 | 72 |
| **True: Sarco**  | 7 | 20 |

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
