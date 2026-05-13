# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:14  |  5-Fold CV  |  Median best epoch: 14

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
| 1 | 0.5360 | 0.1598 | 0.2166 | 0.7206 | 0.1739 |
| 2 | 0.6214 | 0.1766 | 0.2139 | 0.7108 | 0.1918 |
| 3 | 0.5841 | 0.1497 | 0.2163 | 0.6847 | 0.2195 |
| 4 | 0.7330 | 0.3179 | 0.1905 | 0.7389 | 0.3457 |
| 5 | 0.6796 | 0.2358 | 0.1952 | 0.7192 | 0.2597 |
| **Mean** | **0.6308** | **0.2080** | **0.2065** | **0.7148** | **0.2381** |
| **±Std** | 0.0694 | 0.0625 | 0.0113 | 0.0176 | 0.0611 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8139 | 0.3417 | 0.2210 | 0.6520 | 0.3238 |
| 2 | 0.7122 | 0.2096 | 0.1912 | 0.7206 | 0.3448 |
| 3 | 0.8079 | 0.3187 | 0.2073 | 0.6650 | 0.3462 |
| 4 | 0.8071 | 0.3841 | 0.1981 | 0.6749 | 0.3265 |
| 5 | 0.9111 | 0.6647 | 0.1845 | 0.6847 | 0.3962 |
| **Mean** | **0.8105** | **0.3838** | **0.2004** | **0.6794** | **0.3475** |
| **±Std** | 0.0629 | 0.1519 | 0.0128 | 0.0233 | 0.0260 |

CrossAttn best val AUC per fold: Fold1=0.8139, Fold2=0.7122, Fold3=0.8079, Fold4=0.8071, Fold5=0.9111

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6539 | 0.2178 | 0.2047 | 0.7373 | 0.2947 |
| CrossAttn | 0.8287 | 0.3891 | 0.1799 | 0.7216 | 0.3604 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6083 | 0.3052 | 0.2690 | 0.6364 | 0.3077 |
| F | 156 | 0.6903 | 0.1971 | 0.1639 | 0.8013 | 0.2791 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.8020 | 0.4651 | 0.2153 | 0.6162 | 0.4062 |
| F | 156 | 0.8100 | 0.2868 | 0.1575 | 0.7885 | 0.2979 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 174 | 53 |
| **True: Sarco**  | 14 | 14 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 164 | 63 |
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
