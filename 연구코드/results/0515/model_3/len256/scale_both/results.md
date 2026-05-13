# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:42  |  5-Fold CV  |  Median best epoch: 5

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
| 1 | 0.7300 | 0.2499 | 0.1763 | 0.7549 | 0.3243 |
| 2 | 0.6649 | 0.2809 | 0.1913 | 0.7255 | 0.2432 |
| 3 | 0.7423 | 0.2221 | 0.2008 | 0.6798 | 0.3158 |
| 4 | 0.8300 | 0.4650 | 0.1644 | 0.7635 | 0.3846 |
| 5 | 0.8297 | 0.4786 | 0.1788 | 0.7094 | 0.3516 |
| **Mean** | **0.7594** | **0.3393** | **0.1823** | **0.7266** | **0.3239** |
| **±Std** | 0.0633 | 0.1099 | 0.0126 | 0.0305 | 0.0470 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8207 | 0.4250 | 0.1448 | 0.7990 | 0.3881 |
| 2 | 0.6851 | 0.2636 | 0.2063 | 0.6765 | 0.2826 |
| 3 | 0.7647 | 0.3080 | 0.1850 | 0.7291 | 0.3373 |
| 4 | 0.8235 | 0.4226 | 0.2006 | 0.7241 | 0.3913 |
| 5 | 0.8945 | 0.5625 | 0.1598 | 0.7537 | 0.4565 |
| **Mean** | **0.7977** | **0.3963** | **0.1793** | **0.7365** | **0.3712** |
| **±Std** | 0.0698 | 0.1045 | 0.0236 | 0.0400 | 0.0582 |

CrossAttn best val AUC per fold: Fold1=0.8207, Fold2=0.6851, Fold3=0.7647, Fold4=0.8235, Fold5=0.8945

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8076 | 0.3054 | 0.1850 | 0.7373 | 0.3853 |
| CrossAttn | 0.8257 | 0.3686 | 0.1726 | 0.7412 | 0.3889 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7432 | 0.3159 | 0.2542 | 0.6162 | 0.4242 |
| F | 156 | 0.8088 | 0.3843 | 0.1411 | 0.8141 | 0.3256 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7661 | 0.3700 | 0.2266 | 0.6465 | 0.4444 |
| F | 156 | 0.8533 | 0.3873 | 0.1383 | 0.8013 | 0.3111 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 60 |
| **True: Sarco**  | 7 | 21 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 59 |
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
