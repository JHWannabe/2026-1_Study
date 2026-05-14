# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:26  |  5-Fold CV  |  Median best epoch: 15

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
| 1 | 0.7677 | 0.2492 | 0.1764 | 0.7696 | 0.3733 |
| 2 | 0.6411 | 0.1677 | 0.1814 | 0.7635 | 0.2500 |
| 3 | 0.7823 | 0.2600 | 0.1760 | 0.7488 | 0.3544 |
| 4 | 0.6655 | 0.2916 | 0.1767 | 0.7586 | 0.3288 |
| 5 | 0.5917 | 0.2201 | 0.2263 | 0.6798 | 0.2529 |
| **Mean** | **0.6897** | **0.2377** | **0.1874** | **0.7441** | **0.3119** |
| **±Std** | 0.0738 | 0.0418 | 0.0196 | 0.0328 | 0.0513 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8412 | 0.4507 | 0.1903 | 0.7108 | 0.3656 |
| 2 | 0.8626 | 0.4977 | 0.1527 | 0.7586 | 0.4368 |
| 3 | 0.8212 | 0.3646 | 0.1581 | 0.7586 | 0.3797 |
| 4 | 0.8413 | 0.4809 | 0.2286 | 0.6453 | 0.3333 |
| 5 | 0.8310 | 0.3543 | 0.2457 | 0.6059 | 0.3220 |
| **Mean** | **0.8395** | **0.4297** | **0.1951** | **0.6959** | **0.3675** |
| **±Std** | 0.0138 | 0.0593 | 0.0371 | 0.0612 | 0.0405 |

CrossAttn best val AUC per fold: Fold1=0.8412, Fold2=0.8626, Fold3=0.8212, Fold4=0.8413, Fold5=0.8310

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6272 | 0.2081 | 0.2143 | 0.6902 | 0.2330 |
| CrossAttn | 0.7390 | 0.2594 | 0.1662 | 0.7529 | 0.3077 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6497 | 0.3554 | 0.2079 | 0.6907 | 0.3182 |
| F | 158 | 0.6037 | 0.1361 | 0.2182 | 0.6899 | 0.1695 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7367 | 0.3164 | 0.2251 | 0.6392 | 0.3636 |
| F | 158 | 0.7151 | 0.2367 | 0.1300 | 0.8228 | 0.2222 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 164 | 64 |
| **True: Sarco**  | 15 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 178 | 50 |
| **True: Sarco**  | 13 | 14 |

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
