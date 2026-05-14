# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:25  |  5-Fold CV  |  Median best epoch: 4

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
| 1 | 0.5232 | 0.1215 | 0.2264 | 0.6814 | 0.1333 |
| 2 | 0.7268 | 0.2497 | 0.1861 | 0.7488 | 0.3544 |
| 3 | 0.7107 | 0.2332 | 0.1769 | 0.7586 | 0.3288 |
| 4 | 0.7406 | 0.3259 | 0.1868 | 0.7389 | 0.3614 |
| 5 | 0.7531 | 0.2709 | 0.1898 | 0.7094 | 0.3059 |
| **Mean** | **0.6909** | **0.2402** | **0.1932** | **0.7274** | **0.2968** |
| **±Std** | 0.0850 | 0.0671 | 0.0172 | 0.0283 | 0.0841 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8089 | 0.4588 | 0.2111 | 0.6912 | 0.3505 |
| 2 | 0.8609 | 0.4788 | 0.1727 | 0.7389 | 0.4176 |
| 3 | 0.8154 | 0.3635 | 0.1916 | 0.7044 | 0.3617 |
| 4 | 0.8553 | 0.6110 | 0.1549 | 0.7833 | 0.4500 |
| 5 | 0.8483 | 0.3764 | 0.2241 | 0.6502 | 0.3604 |
| **Mean** | **0.8378** | **0.4577** | **0.1909** | **0.7136** | **0.3880** |
| **±Std** | 0.0214 | 0.0888 | 0.0250 | 0.0449 | 0.0389 |

CrossAttn best val AUC per fold: Fold1=0.8089, Fold2=0.8609, Fold3=0.8154, Fold4=0.8553, Fold5=0.8483

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6340 | 0.2311 | 0.1899 | 0.7216 | 0.2366 |
| CrossAttn | 0.7718 | 0.2597 | 0.2136 | 0.6902 | 0.3130 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7263 | 0.3121 | 0.1987 | 0.7113 | 0.3636 |
| F | 158 | 0.5146 | 0.1681 | 0.1845 | 0.7278 | 0.1224 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7900 | 0.3130 | 0.2728 | 0.5979 | 0.3607 |
| F | 158 | 0.7220 | 0.1849 | 0.1773 | 0.7468 | 0.2593 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 173 | 55 |
| **True: Sarco**  | 16 | 11 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 158 | 70 |
| **True: Sarco**  | 9 | 18 |

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
