# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:30  |  5-Fold CV  |  Median best epoch: 3

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
| 1 | 0.7055 | 0.2042 | 0.1785 | 0.7696 | 0.3380 |
| 2 | 0.6418 | 0.2962 | 0.1936 | 0.7157 | 0.2368 |
| 3 | 0.7449 | 0.2163 | 0.1992 | 0.7241 | 0.3636 |
| 4 | 0.8177 | 0.4371 | 0.1699 | 0.7635 | 0.3846 |
| 5 | 0.8302 | 0.3542 | 0.1810 | 0.7537 | 0.4186 |
| **Mean** | **0.7480** | **0.3016** | **0.1844** | **0.7453** | **0.3483** |
| **±Std** | 0.0703 | 0.0871 | 0.0106 | 0.0215 | 0.0617 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8127 | 0.3726 | 0.1301 | 0.8431 | 0.3846 |
| 2 | 0.7067 | 0.2065 | 0.2269 | 0.6520 | 0.3238 |
| 3 | 0.8049 | 0.3264 | 0.2722 | 0.5616 | 0.2992 |
| 4 | 0.7936 | 0.4124 | 0.1931 | 0.7340 | 0.4000 |
| 5 | 0.9058 | 0.6163 | 0.1718 | 0.7438 | 0.4468 |
| **Mean** | **0.8047** | **0.3868** | **0.1988** | **0.7069** | **0.3709** |
| **±Std** | 0.0633 | 0.1339 | 0.0483 | 0.0947 | 0.0532 |

CrossAttn best val AUC per fold: Fold1=0.8127, Fold2=0.7067, Fold3=0.8049, Fold4=0.7936, Fold5=0.9058

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7741 | 0.2954 | 0.1843 | 0.7255 | 0.3396 |
| CrossAttn | 0.8290 | 0.3217 | 0.1741 | 0.7373 | 0.3853 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7138 | 0.3190 | 0.2541 | 0.5859 | 0.3492 |
| F | 156 | 0.7856 | 0.3256 | 0.1399 | 0.8141 | 0.3256 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7597 | 0.3430 | 0.2322 | 0.6465 | 0.4262 |
| F | 156 | 0.8414 | 0.4155 | 0.1372 | 0.7949 | 0.3333 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 60 |
| **True: Sarco**  | 10 | 18 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 60 |
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
