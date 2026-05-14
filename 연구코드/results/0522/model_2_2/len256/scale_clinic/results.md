# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:49  |  5-Fold CV  |  Median best epoch: 12

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
| 1 | 0.4618 | 0.1180 | 0.2288 | 0.6863 | 0.1795 |
| 2 | 0.6231 | 0.1657 | 0.1929 | 0.7143 | 0.2564 |
| 3 | 0.6148 | 0.1938 | 0.2137 | 0.6798 | 0.2529 |
| 4 | 0.6321 | 0.2775 | 0.2294 | 0.7094 | 0.3059 |
| 5 | 0.6220 | 0.1459 | 0.2215 | 0.6798 | 0.1975 |
| **Mean** | **0.5908** | **0.1802** | **0.2173** | **0.6939** | **0.2384** |
| **±Std** | 0.0647 | 0.0546 | 0.0134 | 0.0149 | 0.0452 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7962 | 0.3783 | 0.1864 | 0.7157 | 0.3556 |
| 2 | 0.8805 | 0.4878 | 0.1127 | 0.8473 | 0.5231 |
| 3 | 0.8255 | 0.3607 | 0.1749 | 0.7833 | 0.4211 |
| 4 | 0.8719 | 0.6424 | 0.1869 | 0.7291 | 0.4086 |
| 5 | 0.8162 | 0.3104 | 0.1491 | 0.7980 | 0.4533 |
| **Mean** | **0.8380** | **0.4359** | **0.1620** | **0.7747** | **0.4323** |
| **±Std** | 0.0327 | 0.1184 | 0.0282 | 0.0479 | 0.0552 |

CrossAttn best val AUC per fold: Fold1=0.7962, Fold2=0.8805, Fold3=0.8255, Fold4=0.8719, Fold5=0.8162

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5031 | 0.1461 | 0.2283 | 0.6745 | 0.1443 |
| CrossAttn | 0.7817 | 0.2414 | 0.1730 | 0.7608 | 0.3579 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6179 | 0.2508 | 0.2178 | 0.6907 | 0.2105 |
| F | 158 | 0.4080 | 0.1084 | 0.2347 | 0.6646 | 0.1017 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7676 | 0.2724 | 0.2265 | 0.7113 | 0.4615 |
| F | 158 | 0.7475 | 0.1915 | 0.1401 | 0.7911 | 0.2326 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 165 | 63 |
| **True: Sarco**  | 20 | 7 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 177 | 51 |
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
