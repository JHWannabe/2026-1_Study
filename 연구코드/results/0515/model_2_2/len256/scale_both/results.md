# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:41  |  5-Fold CV  |  Median best epoch: 3

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
| 1 | 0.8039 | 0.4800 | 0.1678 | 0.7500 | 0.3200 |
| 2 | 0.6687 | 0.2422 | 0.1932 | 0.7304 | 0.3038 |
| 3 | 0.7220 | 0.2466 | 0.2046 | 0.6601 | 0.2887 |
| 4 | 0.7762 | 0.3180 | 0.1787 | 0.7537 | 0.3590 |
| 5 | 0.8338 | 0.4121 | 0.1694 | 0.7833 | 0.4500 |
| **Mean** | **0.7609** | **0.3398** | **0.1827** | **0.7355** | **0.3443** |
| **±Std** | 0.0590 | 0.0933 | 0.0142 | 0.0413 | 0.0578 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7907 | 0.3950 | 0.1787 | 0.7059 | 0.3182 |
| 2 | 0.6981 | 0.2122 | 0.2599 | 0.6078 | 0.3103 |
| 3 | 0.8054 | 0.3496 | 0.1666 | 0.7291 | 0.3678 |
| 4 | 0.7908 | 0.3759 | 0.2757 | 0.4729 | 0.2819 |
| 5 | 0.9028 | 0.6293 | 0.1532 | 0.7389 | 0.4301 |
| **Mean** | **0.7976** | **0.3924** | **0.2068** | **0.6509** | **0.3417** |
| **±Std** | 0.0650 | 0.1347 | 0.0507 | 0.1004 | 0.0522 |

CrossAttn best val AUC per fold: Fold1=0.7907, Fold2=0.6981, Fold3=0.8054, Fold4=0.7908, Fold5=0.9028

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8161 | 0.3069 | 0.1893 | 0.7333 | 0.3929 |
| CrossAttn | 0.7873 | 0.2822 | 0.1801 | 0.6784 | 0.3492 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7453 | 0.3149 | 0.2761 | 0.5960 | 0.4286 |
| F | 156 | 0.8301 | 0.3159 | 0.1343 | 0.8205 | 0.3333 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7425 | 0.2995 | 0.2045 | 0.6768 | 0.4483 |
| F | 156 | 0.8044 | 0.3680 | 0.1646 | 0.6795 | 0.2647 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 165 | 62 |
| **True: Sarco**  | 6 | 22 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 151 | 76 |
| **True: Sarco**  | 6 | 22 |

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
