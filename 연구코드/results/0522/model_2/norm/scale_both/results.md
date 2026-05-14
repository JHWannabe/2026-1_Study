# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:53  |  5-Fold CV  |  Median best epoch: 6

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
| 1 | 0.7585 | 0.3812 | 0.1731 | 0.7402 | 0.3457 |
| 2 | 0.7901 | 0.3062 | 0.1571 | 0.7734 | 0.3784 |
| 3 | 0.7582 | 0.2777 | 0.1636 | 0.7537 | 0.3421 |
| 4 | 0.7524 | 0.2752 | 0.1771 | 0.7635 | 0.3846 |
| 5 | 0.7343 | 0.3074 | 0.1796 | 0.7586 | 0.3467 |
| **Mean** | **0.7587** | **0.3095** | **0.1701** | **0.7579** | **0.3595** |
| **±Std** | 0.0180 | 0.0383 | 0.0085 | 0.0110 | 0.0181 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8329 | 0.4418 | 0.1689 | 0.7647 | 0.4146 |
| 2 | 0.8689 | 0.4827 | 0.2284 | 0.6305 | 0.3478 |
| 3 | 0.8209 | 0.3161 | 0.1608 | 0.7635 | 0.3846 |
| 4 | 0.8446 | 0.5144 | 0.1675 | 0.7783 | 0.4444 |
| 5 | 0.8461 | 0.3764 | 0.2367 | 0.6749 | 0.3889 |
| **Mean** | **0.8427** | **0.4263** | **0.1924** | **0.7224** | **0.3961** |
| **±Std** | 0.0159 | 0.0718 | 0.0330 | 0.0588 | 0.0322 |

CrossAttn best val AUC per fold: Fold1=0.8329, Fold2=0.8689, Fold3=0.8209, Fold4=0.8446, Fold5=0.8461

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6897 | 0.2060 | 0.1762 | 0.7608 | 0.2989 |
| CrossAttn | 0.7604 | 0.2776 | 0.2189 | 0.6588 | 0.3256 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6747 | 0.2399 | 0.2381 | 0.6701 | 0.3600 |
| F | 158 | 0.6690 | 0.2095 | 0.1383 | 0.8165 | 0.2162 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7642 | 0.3669 | 0.2562 | 0.6495 | 0.3929 |
| F | 158 | 0.7448 | 0.1677 | 0.1960 | 0.6646 | 0.2740 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 181 | 47 |
| **True: Sarco**  | 14 | 13 |

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
