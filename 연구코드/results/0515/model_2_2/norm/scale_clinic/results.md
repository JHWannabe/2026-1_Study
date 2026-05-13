# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:29  |  5-Fold CV  |  Median best epoch: 10

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
| 1 | 0.7805 | 0.4182 | 0.1630 | 0.7500 | 0.3014 |
| 2 | 0.6579 | 0.2027 | 0.2009 | 0.7108 | 0.2532 |
| 3 | 0.7338 | 0.2711 | 0.2033 | 0.6552 | 0.2857 |
| 4 | 0.7243 | 0.2396 | 0.1860 | 0.7438 | 0.2973 |
| 5 | 0.7607 | 0.4132 | 0.1846 | 0.7488 | 0.3544 |
| **Mean** | **0.7314** | **0.3090** | **0.1876** | **0.7217** | **0.2984** |
| **±Std** | 0.0418 | 0.0898 | 0.0144 | 0.0362 | 0.0327 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8169 | 0.3438 | 0.2210 | 0.6422 | 0.3178 |
| 2 | 0.6925 | 0.2239 | 0.2233 | 0.6520 | 0.2970 |
| 3 | 0.7968 | 0.3181 | 0.2181 | 0.6207 | 0.3186 |
| 4 | 0.8036 | 0.4545 | 0.1261 | 0.8374 | 0.4407 |
| 5 | 0.9309 | 0.6845 | 0.1811 | 0.7094 | 0.4158 |
| **Mean** | **0.8082** | **0.4050** | **0.1939** | **0.6923** | **0.3580** |
| **±Std** | 0.0757 | 0.1579 | 0.0373 | 0.0783 | 0.0584 |

CrossAttn best val AUC per fold: Fold1=0.8169, Fold2=0.6925, Fold3=0.7968, Fold4=0.8036, Fold5=0.9309

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8357 | 0.3122 | 0.1885 | 0.7176 | 0.3898 |
| CrossAttn | 0.8038 | 0.3033 | 0.2126 | 0.6784 | 0.3492 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7367 | 0.3076 | 0.2747 | 0.5758 | 0.4167 |
| F | 156 | 0.8828 | 0.4000 | 0.1338 | 0.8077 | 0.3478 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7496 | 0.3283 | 0.2671 | 0.6263 | 0.4308 |
| F | 156 | 0.8075 | 0.3257 | 0.1780 | 0.7115 | 0.2623 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 160 | 67 |
| **True: Sarco**  | 5 | 23 |

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
