# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:14  |  5-Fold CV  |  Median best epoch: 9

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
| 1 | 0.6044 | 0.1659 | 0.2164 | 0.6814 | 0.2353 |
| 2 | 0.6665 | 0.2144 | 0.1956 | 0.7094 | 0.2532 |
| 3 | 0.7376 | 0.2811 | 0.1792 | 0.7438 | 0.3333 |
| 4 | 0.7368 | 0.2831 | 0.2088 | 0.6995 | 0.3146 |
| 5 | 0.7019 | 0.2282 | 0.2019 | 0.7192 | 0.2963 |
| **Mean** | **0.6894** | **0.2346** | **0.2004** | **0.7107** | **0.2865** |
| **±Std** | 0.0500 | 0.0440 | 0.0127 | 0.0208 | 0.0369 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8014 | 0.4087 | 0.2179 | 0.6814 | 0.3299 |
| 2 | 0.8644 | 0.4398 | 0.1375 | 0.7980 | 0.4810 |
| 3 | 0.8124 | 0.3577 | 0.1143 | 0.8522 | 0.4000 |
| 4 | 0.8692 | 0.6087 | 0.2250 | 0.6601 | 0.3551 |
| 5 | 0.8149 | 0.3410 | 0.2077 | 0.6650 | 0.3585 |
| **Mean** | **0.8325** | **0.4312** | **0.1805** | **0.7313** | **0.3849** |
| **±Std** | 0.0284 | 0.0955 | 0.0455 | 0.0788 | 0.0531 |

CrossAttn best val AUC per fold: Fold1=0.8014, Fold2=0.8644, Fold3=0.8124, Fold4=0.8692, Fold5=0.8149

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6144 | 0.1632 | 0.2056 | 0.6941 | 0.2200 |
| CrossAttn | 0.7529 | 0.2233 | 0.2301 | 0.6588 | 0.3256 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6919 | 0.2599 | 0.2082 | 0.6907 | 0.2857 |
| F | 158 | 0.5507 | 0.1256 | 0.2040 | 0.6962 | 0.1724 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7487 | 0.2624 | 0.2751 | 0.6186 | 0.3729 |
| F | 158 | 0.7178 | 0.1805 | 0.2025 | 0.6835 | 0.2857 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 166 | 62 |
| **True: Sarco**  | 16 | 11 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 147 | 81 |
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
