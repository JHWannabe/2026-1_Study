# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:31  |  5-Fold CV  |  Median best epoch: 10

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
| 1 | 0.6126 | 0.1864 | 0.2390 | 0.6716 | 0.2299 |
| 2 | 0.5849 | 0.1321 | 0.2083 | 0.7291 | 0.1270 |
| 3 | 0.6969 | 0.2058 | 0.1862 | 0.7340 | 0.2286 |
| 4 | 0.6607 | 0.2358 | 0.1688 | 0.7488 | 0.3014 |
| 5 | 0.6328 | 0.2954 | 0.2134 | 0.7094 | 0.2532 |
| **Mean** | **0.6376** | **0.2111** | **0.2032** | **0.7186** | **0.2280** |
| **±Std** | 0.0387 | 0.0540 | 0.0240 | 0.0267 | 0.0570 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8511 | 0.5168 | 0.1257 | 0.8088 | 0.4179 |
| 2 | 0.8636 | 0.5274 | 0.1900 | 0.7044 | 0.3878 |
| 3 | 0.8076 | 0.3682 | 0.2225 | 0.6355 | 0.3148 |
| 4 | 0.8343 | 0.5247 | 0.1584 | 0.7488 | 0.4138 |
| 5 | 0.8222 | 0.3546 | 0.1999 | 0.7192 | 0.4000 |
| **Mean** | **0.8358** | **0.4584** | **0.1793** | **0.7233** | **0.3869** |
| **±Std** | 0.0200 | 0.0793 | 0.0338 | 0.0567 | 0.0376 |

CrossAttn best val AUC per fold: Fold1=0.8511, Fold2=0.8636, Fold3=0.8076, Fold4=0.8343, Fold5=0.8222

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5755 | 0.1857 | 0.2294 | 0.6667 | 0.2202 |
| CrossAttn | 0.7609 | 0.2816 | 0.1848 | 0.7176 | 0.3077 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.5491 | 0.2054 | 0.2715 | 0.6495 | 0.2609 |
| F | 158 | 0.5915 | 0.1744 | 0.2035 | 0.6772 | 0.1905 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7668 | 0.3520 | 0.2383 | 0.6392 | 0.4068 |
| F | 158 | 0.7199 | 0.2248 | 0.1520 | 0.7658 | 0.1778 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 158 | 70 |
| **True: Sarco**  | 15 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 61 |
| **True: Sarco**  | 11 | 16 |

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
