# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:38  |  5-Fold CV  |  Median best epoch: 5

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
| 1 | 0.7637 | 0.3987 | 0.1628 | 0.7549 | 0.3056 |
| 2 | 0.6527 | 0.2319 | 0.1999 | 0.7157 | 0.2564 |
| 3 | 0.7079 | 0.2130 | 0.2092 | 0.6749 | 0.2979 |
| 4 | 0.6848 | 0.2098 | 0.1919 | 0.7291 | 0.2667 |
| 5 | 0.7253 | 0.3472 | 0.1919 | 0.7438 | 0.3500 |
| **Mean** | **0.7069** | **0.2801** | **0.1911** | **0.7237** | **0.2953** |
| **±Std** | 0.0374 | 0.0779 | 0.0155 | 0.0278 | 0.0330 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8187 | 0.3258 | 0.2083 | 0.6716 | 0.3495 |
| 2 | 0.7033 | 0.2299 | 0.2224 | 0.6863 | 0.3191 |
| 3 | 0.8147 | 0.3630 | 0.2030 | 0.6453 | 0.3333 |
| 4 | 0.7803 | 0.3445 | 0.1646 | 0.7537 | 0.3902 |
| 5 | 0.9156 | 0.6353 | 0.1633 | 0.7241 | 0.4286 |
| **Mean** | **0.8065** | **0.3797** | **0.1923** | **0.6962** | **0.3642** |
| **±Std** | 0.0685 | 0.1358 | 0.0240 | 0.0384 | 0.0400 |

CrossAttn best val AUC per fold: Fold1=0.8187, Fold2=0.7033, Fold3=0.8147, Fold4=0.7803, Fold5=0.9156

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8288 | 0.3188 | 0.1886 | 0.7098 | 0.3729 |
| CrossAttn | 0.8150 | 0.3376 | 0.2468 | 0.6196 | 0.3490 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7238 | 0.3057 | 0.2738 | 0.5556 | 0.3889 |
| F | 156 | 0.8821 | 0.3733 | 0.1345 | 0.8077 | 0.3478 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7618 | 0.3864 | 0.2930 | 0.5960 | 0.4444 |
| F | 156 | 0.8270 | 0.3733 | 0.2175 | 0.6346 | 0.2597 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 159 | 68 |
| **True: Sarco**  | 6 | 22 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 132 | 95 |
| **True: Sarco**  | 2 | 26 |

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
