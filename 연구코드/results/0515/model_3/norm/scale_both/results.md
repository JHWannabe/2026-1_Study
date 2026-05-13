# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:41  |  5-Fold CV  |  Median best epoch: 5

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 357 | 297 | 83.2% | 60 | 16.8% |
| Train | F | 660 | 609 | 92.3% | 51 | 7.7% |
| Train | **All** | **1017** | **906** | **89.1%** | **111** | **10.9%** |
| Test | M | 99 | 82 | 82.8% | 17 | 17.2% |
| Test | F | 156 | 145 | 92.9% | 11 | 7.1% |
| Test | **All** | **255** | **227** | **89.0%** | **28** | **11.0%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 357 | 59.35 ± 12.80 | 18.00 | 60.00 | 88.00 |
| Train | F | 660 | 55.60 ± 11.74 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **1017** | **56.92 ± 12.25** | **14.00** | **57.00** | **91.00** |
| Test | M | 99 | 61.58 ± 10.97 | 34.00 | 61.00 | 89.00 |
| Test | F | 156 | 55.71 ± 13.31 | 11.00 | 55.00 | 86.00 |
| Test | **All** | **255** | **57.98 ± 12.78** | **11.00** | **59.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 357 | 24.17 ± 3.31 | 14.48 | 24.16 | 36.76 |
| Train | F | 660 | 23.01 ± 3.24 | 15.62 | 22.69 | 34.61 |
| Train | **All** | **1017** | **23.42 ± 3.31** | **14.48** | **23.24** | **36.76** |
| Test | M | 99 | 24.03 ± 3.22 | 16.80 | 24.16 | 33.87 |
| Test | F | 156 | 23.22 ± 3.52 | 14.40 | 22.71 | 36.24 |
| Test | **All** | **255** | **23.54 ± 3.43** | **14.40** | **23.53** | **36.24** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.6733 | 0.1862 | 0.1776 | 0.7696 | 0.2769 |
| 2 | 0.6385 | 0.2550 | 0.1931 | 0.7206 | 0.2192 |
| 3 | 0.7290 | 0.2093 | 0.2000 | 0.7192 | 0.3294 |
| 4 | 0.7888 | 0.3843 | 0.1745 | 0.7586 | 0.3797 |
| 5 | 0.7913 | 0.3418 | 0.1807 | 0.7291 | 0.3529 |
| **Mean** | **0.7242** | **0.2753** | **0.1852** | **0.7394** | **0.3116** |
| **±Std** | 0.0611 | 0.0762 | 0.0097 | 0.0207 | 0.0573 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8067 | 0.4233 | 0.2783 | 0.5588 | 0.3077 |
| 2 | 0.6680 | 0.2895 | 0.1762 | 0.7206 | 0.2785 |
| 3 | 0.7946 | 0.3591 | 0.2257 | 0.6502 | 0.3364 |
| 4 | 0.8177 | 0.4226 | 0.2755 | 0.5419 | 0.3008 |
| 5 | 0.8918 | 0.4553 | 0.1496 | 0.8030 | 0.5122 |
| **Mean** | **0.7957** | **0.3899** | **0.2210** | **0.6549** | **0.3471** |
| **±Std** | 0.0723 | 0.0591 | 0.0517 | 0.0982 | 0.0846 |

CrossAttn best val AUC per fold: Fold1=0.8067, Fold2=0.6680, Fold3=0.7946, Fold4=0.8177, Fold5=0.8918

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7473 | 0.2830 | 0.1869 | 0.7333 | 0.3462 |
| CrossAttn | 0.8117 | 0.3233 | 0.1689 | 0.7765 | 0.4000 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6872 | 0.3044 | 0.2603 | 0.5960 | 0.3548 |
| F | 156 | 0.7624 | 0.3070 | 0.1404 | 0.8205 | 0.3333 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7439 | 0.3491 | 0.2312 | 0.6667 | 0.4211 |
| F | 156 | 0.8276 | 0.3782 | 0.1294 | 0.8462 | 0.3684 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 169 | 58 |
| **True: Sarco**  | 10 | 18 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 179 | 48 |
| **True: Sarco**  | 9 | 19 |

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
