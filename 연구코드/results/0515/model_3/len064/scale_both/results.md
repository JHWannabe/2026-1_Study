# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:04  |  5-Fold CV  |  Median best epoch: 6

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
| 1 | 0.7577 | 0.2717 | 0.1763 | 0.7549 | 0.3590 |
| 2 | 0.6687 | 0.2694 | 0.1945 | 0.7255 | 0.2432 |
| 3 | 0.7662 | 0.2456 | 0.1964 | 0.7192 | 0.3871 |
| 4 | 0.8488 | 0.4820 | 0.1625 | 0.7783 | 0.4304 |
| 5 | 0.8398 | 0.4993 | 0.1801 | 0.7094 | 0.3656 |
| **Mean** | **0.7763** | **0.3536** | **0.1820** | **0.7375** | **0.3571** |
| **±Std** | 0.0653 | 0.1124 | 0.0125 | 0.0255 | 0.0621 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8224 | 0.3589 | 0.1652 | 0.7500 | 0.3855 |
| 2 | 0.7043 | 0.2626 | 0.1601 | 0.7745 | 0.3235 |
| 3 | 0.7903 | 0.3239 | 0.1702 | 0.7143 | 0.3256 |
| 4 | 0.8327 | 0.4333 | 0.1170 | 0.8227 | 0.3571 |
| 5 | 0.8933 | 0.5378 | 0.2236 | 0.6453 | 0.3684 |
| **Mean** | **0.8086** | **0.3833** | **0.1672** | **0.7414** | **0.3520** |
| **±Std** | 0.0619 | 0.0949 | 0.0339 | 0.0596 | 0.0242 |

CrossAttn best val AUC per fold: Fold1=0.8224, Fold2=0.7043, Fold3=0.7903, Fold4=0.8327, Fold5=0.8933

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8164 | 0.3305 | 0.1856 | 0.7373 | 0.3964 |
| CrossAttn | 0.8310 | 0.3837 | 0.1693 | 0.7608 | 0.4299 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7554 | 0.3521 | 0.2566 | 0.6263 | 0.4478 |
| F | 156 | 0.8157 | 0.3776 | 0.1406 | 0.8077 | 0.3182 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7819 | 0.4126 | 0.2207 | 0.6869 | 0.4918 |
| F | 156 | 0.8313 | 0.4014 | 0.1367 | 0.8077 | 0.3478 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 166 | 61 |
| **True: Sarco**  | 6 | 22 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 171 | 56 |
| **True: Sarco**  | 5 | 23 |

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
