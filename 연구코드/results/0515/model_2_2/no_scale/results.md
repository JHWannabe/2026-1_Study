# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 16:45  |  5-Fold CV  |  Median best epoch: 130

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 402 | 332 | 82.6% | 70 | 17.4% |
| Train | F | 695 | 645 | 92.8% | 50 | 7.2% |
| Train | **All** | **1097** | **977** | **89.1%** | **120** | **10.9%** |
| Test | M | 112 | 95 | 84.8% | 17 | 15.2% |
| Test | F | 163 | 150 | 92.0% | 13 | 8.0% |
| Test | **All** | **275** | **245** | **89.1%** | **30** | **10.9%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 402 | 59.81 ± 12.51 | 18.00 | 60.00 | 89.00 |
| Train | F | 695 | 55.36 ± 12.15 | 11.00 | 55.00 | 91.00 |
| Train | **All** | **1097** | **56.99 ± 12.47** | **11.00** | **58.00** | **91.00** |
| Test | M | 112 | 59.05 ± 12.52 | 23.00 | 59.50 | 84.00 |
| Test | F | 163 | 56.52 ± 12.29 | 22.00 | 56.00 | 87.00 |
| Test | **All** | **275** | **57.55 ± 12.45** | **22.00** | **58.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 402 | 24.22 ± 3.26 | 14.48 | 24.19 | 36.76 |
| Train | F | 695 | 23.09 ± 3.43 | 14.40 | 22.70 | 39.49 |
| Train | **All** | **1097** | **23.51 ± 3.41** | **14.40** | **23.30** | **39.49** |
| Test | M | 112 | 24.07 ± 3.30 | 16.44 | 24.16 | 35.20 |
| Test | F | 163 | 22.99 ± 3.19 | 16.06 | 22.83 | 34.23 |
| Test | **All** | **275** | **23.43 ± 3.28** | **16.06** | **23.44** | **35.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.6144 | 0.1641 | 0.2094 | 0.7182 | 0.2250 |
| 2 | 0.6386 | 0.1758 | 0.2066 | 0.7091 | 0.2195 |
| 3 | 0.5829 | 0.1611 | 0.2081 | 0.6986 | 0.2143 |
| 4 | 0.7147 | 0.2185 | 0.1734 | 0.7808 | 0.3143 |
| 5 | 0.7160 | 0.2467 | 0.2096 | 0.7078 | 0.3043 |
| **Mean** | **0.6533** | **0.1932** | **0.2014** | **0.7229** | **0.2555** |
| **±Std** | 0.0537 | 0.0337 | 0.0141 | 0.0296 | 0.0442 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8335 | 0.4235 | 0.1913 | 0.7000 | 0.3654 |
| 2 | 0.8386 | 0.3855 | 0.1378 | 0.8318 | 0.4308 |
| 3 | 0.7374 | 0.2623 | 0.2002 | 0.6941 | 0.3366 |
| 4 | 0.7682 | 0.2711 | 0.1571 | 0.7945 | 0.4000 |
| 5 | 0.8173 | 0.5119 | 0.2656 | 0.5160 | 0.2933 |
| **Mean** | **0.7990** | **0.3708** | **0.1904** | **0.7073** | **0.3652** |
| **±Std** | 0.0396 | 0.0945 | 0.0439 | 0.1095 | 0.0479 |

CrossAttn best val AUC per fold: Fold1=0.8335, Fold2=0.8386, Fold3=0.7374, Fold4=0.7682, Fold5=0.8173

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6189 | 0.1600 | 0.2125 | 0.7127 | 0.2617 |
| CrossAttn | 0.7140 | 0.3629 | 0.1843 | 0.7636 | 0.3564 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.5895 | 0.2297 | 0.2262 | 0.6875 | 0.2857 |
| F | 163 | 0.6441 | 0.1325 | 0.2031 | 0.7301 | 0.2414 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7059 | 0.3746 | 0.2677 | 0.5982 | 0.3284 |
| F | 163 | 0.6882 | 0.3553 | 0.1269 | 0.8773 | 0.4118 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 182 | 63 |
| **True: Sarco**  | 16 | 14 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 192 | 53 |
| **True: Sarco**  | 12 | 18 |

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
