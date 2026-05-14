# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:35  |  5-Fold CV  |  Median best epoch: 12

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
| 1 | 0.7955 | 0.3945 | 0.1654 | 0.7549 | 0.3590 |
| 2 | 0.8066 | 0.3403 | 0.1599 | 0.7931 | 0.4167 |
| 3 | 0.8343 | 0.3084 | 0.1584 | 0.7734 | 0.3784 |
| 4 | 0.8144 | 0.4251 | 0.1881 | 0.7094 | 0.3656 |
| 5 | 0.8021 | 0.3915 | 0.1769 | 0.7635 | 0.4000 |
| **Mean** | **0.8106** | **0.3720** | **0.1697** | **0.7589** | **0.3839** |
| **±Std** | 0.0134 | 0.0419 | 0.0113 | 0.0278 | 0.0215 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8452 | 0.4758 | 0.1188 | 0.8235 | 0.4375 |
| 2 | 0.8893 | 0.6117 | 0.1689 | 0.7192 | 0.4000 |
| 3 | 0.8159 | 0.3731 | 0.1335 | 0.8227 | 0.4545 |
| 4 | 0.8506 | 0.5819 | 0.1707 | 0.7340 | 0.4130 |
| 5 | 0.8425 | 0.3639 | 0.1689 | 0.7635 | 0.4286 |
| **Mean** | **0.8487** | **0.4813** | **0.1522** | **0.7726** | **0.4267** |
| **±Std** | 0.0235 | 0.1026 | 0.0217 | 0.0436 | 0.0189 |

CrossAttn best val AUC per fold: Fold1=0.8452, Fold2=0.8893, Fold3=0.8159, Fold4=0.8506, Fold5=0.8425

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7471 | 0.2466 | 0.1775 | 0.7529 | 0.2759 |
| CrossAttn | 0.7318 | 0.2520 | 0.1673 | 0.7569 | 0.3261 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7608 | 0.3303 | 0.2309 | 0.6804 | 0.3922 |
| F | 158 | 0.7066 | 0.1543 | 0.1448 | 0.7975 | 0.1111 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7401 | 0.3350 | 0.2186 | 0.6392 | 0.3636 |
| F | 158 | 0.7162 | 0.1792 | 0.1358 | 0.8291 | 0.2703 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 180 | 48 |
| **True: Sarco**  | 15 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 178 | 50 |
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
