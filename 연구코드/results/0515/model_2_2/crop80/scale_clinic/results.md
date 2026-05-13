# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:50  |  5-Fold CV  |  Median best epoch: 6

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
| 1 | 0.7033 | 0.2206 | 0.1868 | 0.7206 | 0.2597 |
| 2 | 0.5686 | 0.1388 | 0.2197 | 0.6618 | 0.1882 |
| 3 | 0.5427 | 0.1595 | 0.2539 | 0.6207 | 0.1720 |
| 4 | 0.4877 | 0.1228 | 0.2406 | 0.6847 | 0.1795 |
| 5 | 0.6113 | 0.1835 | 0.2252 | 0.6995 | 0.2278 |
| **Mean** | **0.5827** | **0.1650** | **0.2252** | **0.6775** | **0.2055** |
| **±Std** | 0.0724 | 0.0345 | 0.0226 | 0.0343 | 0.0333 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8097 | 0.3781 | 0.1929 | 0.7255 | 0.3488 |
| 2 | 0.7057 | 0.2512 | 0.1839 | 0.7206 | 0.2785 |
| 3 | 0.8001 | 0.3232 | 0.2107 | 0.6355 | 0.3273 |
| 4 | 0.7589 | 0.4015 | 0.1658 | 0.7291 | 0.3210 |
| 5 | 0.8968 | 0.5664 | 0.2259 | 0.6158 | 0.3390 |
| **Mean** | **0.7942** | **0.3841** | **0.1958** | **0.6853** | **0.3229** |
| **±Std** | 0.0630 | 0.1048 | 0.0209 | 0.0492 | 0.0242 |

CrossAttn best val AUC per fold: Fold1=0.8097, Fold2=0.7057, Fold3=0.8001, Fold4=0.7589, Fold5=0.8968

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7212 | 0.2281 | 0.2122 | 0.6980 | 0.3304 |
| CrossAttn | 0.7934 | 0.3553 | 0.2045 | 0.6314 | 0.2985 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6535 | 0.2499 | 0.2554 | 0.6566 | 0.3704 |
| F | 156 | 0.7712 | 0.2313 | 0.1848 | 0.7244 | 0.2951 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7633 | 0.4077 | 0.2263 | 0.6263 | 0.3934 |
| F | 156 | 0.7987 | 0.3274 | 0.1907 | 0.6346 | 0.2192 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 159 | 68 |
| **True: Sarco**  | 9 | 19 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 141 | 86 |
| **True: Sarco**  | 8 | 20 |

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
