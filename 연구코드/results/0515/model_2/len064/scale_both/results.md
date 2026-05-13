# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 16:29  |  5-Fold CV  |  Median best epoch: 11

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
| 1 | 0.7500 | 0.2748 | 0.1760 | 0.7598 | 0.3467 |
| 2 | 0.6680 | 0.2693 | 0.1931 | 0.7255 | 0.2432 |
| 3 | 0.7639 | 0.2453 | 0.1981 | 0.6798 | 0.3434 |
| 4 | 0.8430 | 0.4831 | 0.1639 | 0.7882 | 0.4416 |
| 5 | 0.8594 | 0.4762 | 0.1825 | 0.7192 | 0.3871 |
| **Mean** | **0.7769** | **0.3497** | **0.1827** | **0.7345** | **0.3524** |
| **±Std** | 0.0692 | 0.1066 | 0.0122 | 0.0370 | 0.0651 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8109 | 0.3477 | 0.1916 | 0.7206 | 0.3294 |
| 2 | 0.6971 | 0.2363 | 0.2140 | 0.6520 | 0.2828 |
| 3 | 0.8169 | 0.3246 | 0.2253 | 0.6108 | 0.3248 |
| 4 | 0.7991 | 0.4117 | 0.1764 | 0.6995 | 0.3579 |
| 5 | 0.8958 | 0.6362 | 0.1632 | 0.7291 | 0.4211 |
| **Mean** | **0.8040** | **0.3913** | **0.1941** | **0.6824** | **0.3432** |
| **±Std** | 0.0634 | 0.1348 | 0.0230 | 0.0447 | 0.0457 |

CrossAttn best val AUC per fold: Fold1=0.8109, Fold2=0.6971, Fold3=0.8169, Fold4=0.7991, Fold5=0.8958

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8147 | 0.3195 | 0.1846 | 0.7412 | 0.4000 |
| CrossAttn | 0.8137 | 0.3500 | 0.1855 | 0.6980 | 0.3529 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7554 | 0.3357 | 0.2536 | 0.6364 | 0.4706 |
| F | 156 | 0.8107 | 0.3734 | 0.1408 | 0.8077 | 0.2857 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7726 | 0.3984 | 0.2426 | 0.6364 | 0.4375 |
| F | 156 | 0.8150 | 0.3190 | 0.1493 | 0.7372 | 0.2545 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 60 |
| **True: Sarco**  | 6 | 22 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 157 | 70 |
| **True: Sarco**  | 7 | 21 |

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
