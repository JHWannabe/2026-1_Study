# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 16:58  |  5-Fold CV  |  Median best epoch: 4

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
| 1 | 0.4900 | 0.1315 | 0.2193 | 0.7059 | 0.1429 |
| 2 | 0.6020 | 0.1634 | 0.2158 | 0.6961 | 0.1842 |
| 3 | 0.5321 | 0.1174 | 0.2297 | 0.6700 | 0.1299 |
| 4 | 0.6818 | 0.2623 | 0.2146 | 0.7044 | 0.2857 |
| 5 | 0.6243 | 0.1859 | 0.2006 | 0.7389 | 0.2535 |
| **Mean** | **0.5861** | **0.1721** | **0.2160** | **0.7031** | **0.1992** |
| **±Std** | 0.0679 | 0.0510 | 0.0093 | 0.0221 | 0.0611 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8099 | 0.4543 | 0.2275 | 0.6716 | 0.3619 |
| 2 | 0.6572 | 0.2324 | 0.1727 | 0.7353 | 0.2895 |
| 3 | 0.8172 | 0.3312 | 0.2218 | 0.6601 | 0.3429 |
| 4 | 0.8119 | 0.3989 | 0.1902 | 0.7537 | 0.3902 |
| 5 | 0.9136 | 0.6154 | 0.1749 | 0.7291 | 0.4211 |
| **Mean** | **0.8020** | **0.4064** | **0.1974** | **0.7099** | **0.3611** |
| **±Std** | 0.0822 | 0.1280 | 0.0231 | 0.0371 | 0.0445 |

CrossAttn best val AUC per fold: Fold1=0.8099, Fold2=0.6572, Fold3=0.8172, Fold4=0.8119, Fold5=0.9136

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6449 | 0.2041 | 0.2092 | 0.7098 | 0.2600 |
| CrossAttn | 0.8197 | 0.3381 | 0.2222 | 0.6627 | 0.3676 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6055 | 0.2876 | 0.2654 | 0.6263 | 0.2745 |
| F | 156 | 0.6759 | 0.1811 | 0.1736 | 0.7628 | 0.2449 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7733 | 0.3735 | 0.2588 | 0.6162 | 0.4722 |
| F | 156 | 0.8025 | 0.3200 | 0.1989 | 0.6923 | 0.2500 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 59 |
| **True: Sarco**  | 15 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 144 | 83 |
| **True: Sarco**  | 3 | 25 |

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
