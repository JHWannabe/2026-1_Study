# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:32  |  5-Fold CV  |  Median best epoch: 17

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
| 1 | 0.4950 | 0.1336 | 0.2213 | 0.7059 | 0.1667 |
| 2 | 0.6219 | 0.1671 | 0.1996 | 0.7255 | 0.2000 |
| 3 | 0.4907 | 0.1092 | 0.2274 | 0.6798 | 0.1096 |
| 4 | 0.7007 | 0.2763 | 0.2034 | 0.7291 | 0.3210 |
| 5 | 0.6313 | 0.1865 | 0.1985 | 0.7438 | 0.2778 |
| **Mean** | **0.5879** | **0.1745** | **0.2100** | **0.7168** | **0.2150** |
| **±Std** | 0.0823 | 0.0574 | 0.0120 | 0.0221 | 0.0760 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8384 | 0.3607 | 0.2095 | 0.7304 | 0.4086 |
| 2 | 0.7336 | 0.2532 | 0.2015 | 0.7108 | 0.3516 |
| 3 | 0.8086 | 0.3096 | 0.2630 | 0.5271 | 0.2836 |
| 4 | 0.8071 | 0.4440 | 0.1963 | 0.6798 | 0.3299 |
| 5 | 0.9124 | 0.6573 | 0.1841 | 0.6995 | 0.3960 |
| **Mean** | **0.8200** | **0.4050** | **0.2109** | **0.6695** | **0.3540** |
| **±Std** | 0.0577 | 0.1409 | 0.0274 | 0.0731 | 0.0454 |

CrossAttn best val AUC per fold: Fold1=0.8384, Fold2=0.7336, Fold3=0.8086, Fold4=0.8071, Fold5=0.9124

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6070 | 0.1929 | 0.2061 | 0.7176 | 0.2340 |
| CrossAttn | 0.8269 | 0.4135 | 0.1920 | 0.6863 | 0.3333 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.5617 | 0.2813 | 0.2652 | 0.6263 | 0.2449 |
| F | 156 | 0.6489 | 0.1625 | 0.1686 | 0.7756 | 0.2222 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.8034 | 0.4532 | 0.2067 | 0.6566 | 0.4333 |
| F | 156 | 0.8351 | 0.3752 | 0.1827 | 0.7051 | 0.2333 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 172 | 55 |
| **True: Sarco**  | 17 | 11 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 155 | 72 |
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
