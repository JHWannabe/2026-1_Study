# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:44  |  5-Fold CV  |  Median best epoch: 3

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
| 1 | 0.7737 | 0.4108 | 0.1688 | 0.7598 | 0.3797 |
| 2 | 0.8094 | 0.3242 | 0.1571 | 0.7882 | 0.4267 |
| 3 | 0.7755 | 0.2818 | 0.1616 | 0.7635 | 0.3684 |
| 4 | 0.7722 | 0.3128 | 0.1764 | 0.7488 | 0.3704 |
| 5 | 0.7677 | 0.3368 | 0.1761 | 0.7635 | 0.3684 |
| **Mean** | **0.7797** | **0.3333** | **0.1680** | **0.7648** | **0.3827** |
| **±Std** | 0.0151 | 0.0428 | 0.0077 | 0.0129 | 0.0224 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8262 | 0.4416 | 0.1235 | 0.8284 | 0.4776 |
| 2 | 0.8672 | 0.4956 | 0.1305 | 0.8276 | 0.4776 |
| 3 | 0.8335 | 0.3453 | 0.1846 | 0.7143 | 0.3696 |
| 4 | 0.8732 | 0.5755 | 0.1776 | 0.7931 | 0.4615 |
| 5 | 0.8566 | 0.3838 | 0.1644 | 0.7931 | 0.4878 |
| **Mean** | **0.8513** | **0.4483** | **0.1561** | **0.7913** | **0.4548** |
| **±Std** | 0.0185 | 0.0815 | 0.0248 | 0.0416 | 0.0435 |

CrossAttn best val AUC per fold: Fold1=0.8262, Fold2=0.8672, Fold3=0.8335, Fold4=0.8732, Fold5=0.8566

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7050 | 0.2144 | 0.1743 | 0.7686 | 0.2892 |
| CrossAttn | 0.7773 | 0.2606 | 0.2207 | 0.6549 | 0.3231 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7005 | 0.2592 | 0.2343 | 0.6701 | 0.3600 |
| F | 158 | 0.6769 | 0.2097 | 0.1375 | 0.8291 | 0.1818 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7814 | 0.3183 | 0.2978 | 0.5567 | 0.3768 |
| F | 158 | 0.7475 | 0.1750 | 0.1734 | 0.7152 | 0.2623 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 184 | 44 |
| **True: Sarco**  | 15 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 146 | 82 |
| **True: Sarco**  | 6 | 21 |

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
