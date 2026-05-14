# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:14  |  5-Fold CV  |  Median best epoch: 18

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
| 1 | 0.6988 | 0.2131 | 0.2095 | 0.6961 | 0.2791 |
| 2 | 0.6497 | 0.1758 | 0.2096 | 0.7044 | 0.2105 |
| 3 | 0.7275 | 0.2299 | 0.1847 | 0.7192 | 0.3133 |
| 4 | 0.6308 | 0.1969 | 0.2021 | 0.7192 | 0.2963 |
| 5 | 0.6645 | 0.2732 | 0.2007 | 0.7044 | 0.2857 |
| **Mean** | **0.6743** | **0.2178** | **0.2013** | **0.7087** | **0.2770** |
| **±Std** | 0.0347 | 0.0330 | 0.0091 | 0.0091 | 0.0352 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8506 | 0.4509 | 0.1410 | 0.7941 | 0.4000 |
| 2 | 0.8604 | 0.4661 | 0.1835 | 0.7094 | 0.4040 |
| 3 | 0.8147 | 0.3533 | 0.1466 | 0.7980 | 0.4225 |
| 4 | 0.8287 | 0.5534 | 0.1364 | 0.7783 | 0.3836 |
| 5 | 0.8199 | 0.3279 | 0.1797 | 0.7783 | 0.4444 |
| **Mean** | **0.8349** | **0.4303** | **0.1574** | **0.7716** | **0.4109** |
| **±Std** | 0.0177 | 0.0816 | 0.0200 | 0.0322 | 0.0209 |

CrossAttn best val AUC per fold: Fold1=0.8506, Fold2=0.8604, Fold3=0.8147, Fold4=0.8287, Fold5=0.8199

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6298 | 0.1686 | 0.2047 | 0.6824 | 0.2430 |
| CrossAttn | 0.7445 | 0.2869 | 0.1480 | 0.7882 | 0.3250 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6403 | 0.2287 | 0.2353 | 0.6392 | 0.2857 |
| F | 158 | 0.5979 | 0.1284 | 0.1860 | 0.7089 | 0.2069 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7392 | 0.3656 | 0.2207 | 0.6598 | 0.3529 |
| F | 158 | 0.7215 | 0.2622 | 0.1034 | 0.8671 | 0.2759 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 161 | 67 |
| **True: Sarco**  | 14 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 188 | 40 |
| **True: Sarco**  | 14 | 13 |

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
