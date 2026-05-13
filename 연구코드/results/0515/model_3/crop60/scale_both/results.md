# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:22  |  5-Fold CV  |  Median best epoch: 7

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
| 1 | 0.7927 | 0.3249 | 0.1767 | 0.7402 | 0.3291 |
| 2 | 0.6704 | 0.2695 | 0.1898 | 0.7255 | 0.2432 |
| 3 | 0.7564 | 0.2192 | 0.1984 | 0.7094 | 0.3516 |
| 4 | 0.8343 | 0.4848 | 0.1654 | 0.7537 | 0.3902 |
| 5 | 0.8390 | 0.4907 | 0.1779 | 0.6946 | 0.3404 |
| **Mean** | **0.7786** | **0.3578** | **0.1817** | **0.7247** | **0.3309** |
| **±Std** | 0.0619 | 0.1112 | 0.0114 | 0.0211 | 0.0484 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8079 | 0.3805 | 0.2077 | 0.7108 | 0.3516 |
| 2 | 0.6904 | 0.2518 | 0.2232 | 0.6912 | 0.3077 |
| 3 | 0.7903 | 0.2958 | 0.2455 | 0.6207 | 0.3186 |
| 4 | 0.8076 | 0.3430 | 0.1911 | 0.7044 | 0.3478 |
| 5 | 0.9028 | 0.6004 | 0.2300 | 0.6158 | 0.3500 |
| **Mean** | **0.7998** | **0.3743** | **0.2195** | **0.6686** | **0.3352** |
| **±Std** | 0.0675 | 0.1211 | 0.0187 | 0.0416 | 0.0183 |

CrossAttn best val AUC per fold: Fold1=0.8079, Fold2=0.6904, Fold3=0.7903, Fold4=0.8076, Fold5=0.9028

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8092 | 0.3208 | 0.1887 | 0.7333 | 0.3929 |
| CrossAttn | 0.8205 | 0.3154 | 0.2149 | 0.6627 | 0.3485 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7446 | 0.3459 | 0.2581 | 0.6364 | 0.4545 |
| F | 156 | 0.8125 | 0.3687 | 0.1447 | 0.7949 | 0.3043 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7633 | 0.3262 | 0.2635 | 0.6061 | 0.4348 |
| F | 156 | 0.8339 | 0.4317 | 0.1841 | 0.6987 | 0.2540 |

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
| **True: Normal** | 146 | 81 |
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
