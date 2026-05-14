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
| 1 | 0.5649 | 0.1665 | 0.2459 | 0.6520 | 0.1647 |
| 2 | 0.5487 | 0.1190 | 0.2291 | 0.7241 | 0.1250 |
| 3 | 0.6783 | 0.1815 | 0.1958 | 0.7044 | 0.2105 |
| 4 | 0.6253 | 0.2429 | 0.1783 | 0.7438 | 0.2778 |
| 5 | 0.5972 | 0.2641 | 0.2284 | 0.6897 | 0.2588 |
| **Mean** | **0.6029** | **0.1948** | **0.2155** | **0.7028** | **0.2074** |
| **±Std** | 0.0460 | 0.0526 | 0.0247 | 0.0313 | 0.0570 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8651 | 0.4612 | 0.1262 | 0.8284 | 0.5070 |
| 2 | 0.8677 | 0.5139 | 0.2025 | 0.6847 | 0.3725 |
| 3 | 0.8282 | 0.3410 | 0.1572 | 0.7488 | 0.3704 |
| 4 | 0.8448 | 0.4595 | 0.2172 | 0.6502 | 0.3486 |
| 5 | 0.8225 | 0.3363 | 0.1919 | 0.7537 | 0.4318 |
| **Mean** | **0.8457** | **0.4224** | **0.1790** | **0.7332** | **0.4061** |
| **±Std** | 0.0185 | 0.0711 | 0.0330 | 0.0616 | 0.0575 |

CrossAttn best val AUC per fold: Fold1=0.8651, Fold2=0.8677, Fold3=0.8282, Fold4=0.8448, Fold5=0.8225

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5302 | 0.1915 | 0.2329 | 0.6588 | 0.1869 |
| CrossAttn | 0.7705 | 0.2887 | 0.1937 | 0.7098 | 0.2885 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.4828 | 0.2251 | 0.2916 | 0.5670 | 0.1923 |
| F | 158 | 0.5655 | 0.1743 | 0.1968 | 0.7152 | 0.1818 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7788 | 0.3636 | 0.2655 | 0.5876 | 0.3548 |
| F | 158 | 0.7379 | 0.2115 | 0.1497 | 0.7848 | 0.1905 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 158 | 70 |
| **True: Sarco**  | 17 | 10 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 166 | 62 |
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
