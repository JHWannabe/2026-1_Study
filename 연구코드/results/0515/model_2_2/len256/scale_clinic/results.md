# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:32  |  5-Fold CV  |  Median best epoch: 3

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
| 1 | 0.6259 | 0.2018 | 0.1951 | 0.7402 | 0.2740 |
| 2 | 0.5378 | 0.1267 | 0.2166 | 0.6814 | 0.1333 |
| 3 | 0.4890 | 0.1172 | 0.2816 | 0.6108 | 0.1505 |
| 4 | 0.4661 | 0.1119 | 0.2406 | 0.6995 | 0.2078 |
| 5 | 0.5954 | 0.1917 | 0.2252 | 0.6847 | 0.2381 |
| **Mean** | **0.5428** | **0.1499** | **0.2318** | **0.6833** | **0.2007** |
| **±Std** | 0.0608 | 0.0387 | 0.0289 | 0.0418 | 0.0527 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8094 | 0.3306 | 0.2037 | 0.6569 | 0.3396 |
| 2 | 0.6868 | 0.2196 | 0.2192 | 0.6667 | 0.3200 |
| 3 | 0.8089 | 0.3242 | 0.1790 | 0.7488 | 0.3704 |
| 4 | 0.7559 | 0.3466 | 0.2746 | 0.4433 | 0.2803 |
| 5 | 0.9001 | 0.5496 | 0.1625 | 0.7438 | 0.4348 |
| **Mean** | **0.7922** | **0.3541** | **0.2078** | **0.6519** | **0.3490** |
| **±Std** | 0.0702 | 0.1075 | 0.0387 | 0.1110 | 0.0519 |

CrossAttn best val AUC per fold: Fold1=0.8094, Fold2=0.6868, Fold3=0.8089, Fold4=0.7559, Fold5=0.9001

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6616 | 0.2027 | 0.2279 | 0.6627 | 0.2586 |
| CrossAttn | 0.8093 | 0.3370 | 0.2003 | 0.6510 | 0.3308 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.5165 | 0.2360 | 0.3113 | 0.4949 | 0.1667 |
| F | 156 | 0.8301 | 0.2750 | 0.1749 | 0.7692 | 0.3571 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7747 | 0.3740 | 0.2381 | 0.6162 | 0.4242 |
| F | 156 | 0.7843 | 0.3080 | 0.1764 | 0.6731 | 0.2388 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 154 | 73 |
| **True: Sarco**  | 13 | 15 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 144 | 83 |
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
