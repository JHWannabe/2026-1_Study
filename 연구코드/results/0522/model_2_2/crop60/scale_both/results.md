# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:34  |  5-Fold CV  |  Median best epoch: 9

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
| 1 | 0.7740 | 0.3092 | 0.1686 | 0.7647 | 0.4000 |
| 2 | 0.8270 | 0.4334 | 0.1454 | 0.7980 | 0.4384 |
| 3 | 0.8697 | 0.3773 | 0.1457 | 0.8177 | 0.4932 |
| 4 | 0.8453 | 0.3920 | 0.1750 | 0.7488 | 0.4138 |
| 5 | 0.8227 | 0.3715 | 0.1762 | 0.7340 | 0.4000 |
| **Mean** | **0.8277** | **0.3767** | **0.1622** | **0.7726** | **0.4291** |
| **±Std** | 0.0316 | 0.0401 | 0.0138 | 0.0310 | 0.0350 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8077 | 0.4025 | 0.1948 | 0.7108 | 0.3789 |
| 2 | 0.8418 | 0.4586 | 0.1429 | 0.7833 | 0.4211 |
| 3 | 0.8222 | 0.3539 | 0.1384 | 0.8325 | 0.3929 |
| 4 | 0.8360 | 0.6144 | 0.1797 | 0.7291 | 0.3956 |
| 5 | 0.8348 | 0.4124 | 0.2591 | 0.6010 | 0.3193 |
| **Mean** | **0.8285** | **0.4484** | **0.1830** | **0.7313** | **0.3816** |
| **±Std** | 0.0122 | 0.0894 | 0.0437 | 0.0779 | 0.0340 |

CrossAttn best val AUC per fold: Fold1=0.8077, Fold2=0.8418, Fold3=0.8222, Fold4=0.8360, Fold5=0.8348

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7060 | 0.2210 | 0.1773 | 0.7529 | 0.2921 |
| CrossAttn | 0.7853 | 0.2763 | 0.1965 | 0.7373 | 0.3738 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7418 | 0.2927 | 0.2399 | 0.6804 | 0.4151 |
| F | 158 | 0.6504 | 0.1318 | 0.1389 | 0.7975 | 0.1111 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7960 | 0.3384 | 0.2407 | 0.6907 | 0.4444 |
| F | 158 | 0.7507 | 0.2046 | 0.1693 | 0.7658 | 0.3019 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 179 | 49 |
| **True: Sarco**  | 14 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 60 |
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
