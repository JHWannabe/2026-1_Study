# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:07  |  5-Fold CV  |  Median best epoch: 5

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
| 1 | 0.6361 | 0.1643 | 0.2243 | 0.6765 | 0.2326 |
| 2 | 0.6120 | 0.1505 | 0.1989 | 0.7291 | 0.2029 |
| 3 | 0.7268 | 0.2145 | 0.1758 | 0.7340 | 0.3077 |
| 4 | 0.6439 | 0.2484 | 0.1827 | 0.7438 | 0.3158 |
| 5 | 0.6012 | 0.2122 | 0.2236 | 0.7143 | 0.2927 |
| **Mean** | **0.6440** | **0.1980** | **0.2011** | **0.7195** | **0.2703** |
| **±Std** | 0.0442 | 0.0358 | 0.0201 | 0.0235 | 0.0446 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8487 | 0.4443 | 0.1714 | 0.7353 | 0.4255 |
| 2 | 0.8719 | 0.4883 | 0.1649 | 0.7438 | 0.4222 |
| 3 | 0.8320 | 0.3765 | 0.1833 | 0.6946 | 0.3261 |
| 4 | 0.8463 | 0.5969 | 0.2024 | 0.7094 | 0.3789 |
| 5 | 0.8112 | 0.3397 | 0.1703 | 0.7586 | 0.4235 |
| **Mean** | **0.8420** | **0.4491** | **0.1784** | **0.7283** | **0.3953** |
| **±Std** | 0.0201 | 0.0901 | 0.0134 | 0.0233 | 0.0387 |

CrossAttn best val AUC per fold: Fold1=0.8487, Fold2=0.8719, Fold3=0.8320, Fold4=0.8463, Fold5=0.8112

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5570 | 0.1799 | 0.2139 | 0.7176 | 0.2500 |
| CrossAttn | 0.7606 | 0.2905 | 0.1845 | 0.7176 | 0.3208 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.5456 | 0.2700 | 0.2353 | 0.7216 | 0.3415 |
| F | 158 | 0.5740 | 0.1170 | 0.2008 | 0.7152 | 0.1818 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7556 | 0.3605 | 0.2330 | 0.6598 | 0.4000 |
| F | 158 | 0.7379 | 0.2490 | 0.1547 | 0.7532 | 0.2353 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 171 | 57 |
| **True: Sarco**  | 15 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 166 | 62 |
| **True: Sarco**  | 10 | 17 |

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
