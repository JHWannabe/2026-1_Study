# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 18:55  |  5-Fold CV  |  Median best epoch: 7

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
| 1 | 0.6618 | 0.3026 | 0.1878 | 0.7108 | 0.2532 |
| 2 | 0.6363 | 0.2127 | 0.2083 | 0.6961 | 0.2439 |
| 3 | 0.6017 | 0.1759 | 0.2287 | 0.6256 | 0.2083 |
| 4 | 0.5816 | 0.1671 | 0.2323 | 0.6700 | 0.2299 |
| 5 | 0.6617 | 0.2483 | 0.2076 | 0.7143 | 0.2750 |
| **Mean** | **0.6286** | **0.2213** | **0.2130** | **0.6833** | **0.2421** |
| **±Std** | 0.0322 | 0.0498 | 0.0162 | 0.0328 | 0.0223 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8204 | 0.3038 | 0.2379 | 0.6225 | 0.3419 |
| 2 | 0.6906 | 0.2171 | 0.2158 | 0.6569 | 0.3000 |
| 3 | 0.8094 | 0.3101 | 0.2151 | 0.6552 | 0.3269 |
| 4 | 0.7654 | 0.4075 | 0.2066 | 0.6798 | 0.3299 |
| 5 | 0.8900 | 0.6102 | 0.1276 | 0.8128 | 0.4722 |
| **Mean** | **0.7952** | **0.3698** | **0.2006** | **0.6854** | **0.3542** |
| **±Std** | 0.0658 | 0.1345 | 0.0379 | 0.0662 | 0.0606 |

CrossAttn best val AUC per fold: Fold1=0.8204, Fold2=0.6906, Fold3=0.8094, Fold4=0.7654, Fold5=0.8900

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7574 | 0.3250 | 0.2058 | 0.6706 | 0.3226 |
| CrossAttn | 0.8037 | 0.3088 | 0.1858 | 0.6980 | 0.3529 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6456 | 0.3048 | 0.2625 | 0.5859 | 0.3279 |
| F | 156 | 0.8639 | 0.4527 | 0.1698 | 0.7244 | 0.3175 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7511 | 0.3271 | 0.2314 | 0.6566 | 0.4333 |
| F | 156 | 0.8107 | 0.3790 | 0.1569 | 0.7244 | 0.2712 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 151 | 76 |
| **True: Sarco**  | 8 | 20 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 157 | 70 |
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
