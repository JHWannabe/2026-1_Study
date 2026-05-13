# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:12  |  5-Fold CV  |  Median best epoch: 5

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
| 1 | 0.6201 | 0.1876 | 0.2070 | 0.7010 | 0.1644 |
| 2 | 0.6010 | 0.1977 | 0.1874 | 0.7402 | 0.2535 |
| 3 | 0.6371 | 0.1547 | 0.2201 | 0.6798 | 0.2857 |
| 4 | 0.6688 | 0.2829 | 0.1991 | 0.6995 | 0.2651 |
| 5 | 0.7022 | 0.2882 | 0.1815 | 0.7488 | 0.3377 |
| **Mean** | **0.6458** | **0.2222** | **0.1990** | **0.7139** | **0.2613** |
| **±Std** | 0.0359 | 0.0537 | 0.0138 | 0.0262 | 0.0564 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8072 | 0.4060 | 0.1544 | 0.7843 | 0.3889 |
| 2 | 0.6933 | 0.2986 | 0.2156 | 0.6471 | 0.2800 |
| 3 | 0.7745 | 0.3076 | 0.1806 | 0.7882 | 0.3582 |
| 4 | 0.8159 | 0.4033 | 0.2338 | 0.6207 | 0.3186 |
| 5 | 0.9073 | 0.6788 | 0.1411 | 0.7734 | 0.4651 |
| **Mean** | **0.7996** | **0.4188** | **0.1851** | **0.7227** | **0.3622** |
| **±Std** | 0.0691 | 0.1377 | 0.0352 | 0.0732 | 0.0632 |

CrossAttn best val AUC per fold: Fold1=0.8072, Fold2=0.6933, Fold3=0.7745, Fold4=0.8159, Fold5=0.9073

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7486 | 0.2695 | 0.1946 | 0.7373 | 0.3366 |
| CrossAttn | 0.8263 | 0.3723 | 0.1720 | 0.7451 | 0.3810 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7209 | 0.3287 | 0.2223 | 0.7071 | 0.4082 |
| F | 156 | 0.7705 | 0.2737 | 0.1770 | 0.7564 | 0.2692 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7740 | 0.3907 | 0.2121 | 0.6970 | 0.4643 |
| F | 156 | 0.8408 | 0.4198 | 0.1465 | 0.7756 | 0.2857 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 171 | 56 |
| **True: Sarco**  | 11 | 17 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 170 | 57 |
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
