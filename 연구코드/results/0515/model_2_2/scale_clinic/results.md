# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 16:51  |  5-Fold CV  |  Median best epoch: 2

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
| 1 | 0.5793 | 0.1541 | 0.2214 | 0.6909 | 0.2093 |
| 2 | 0.5770 | 0.1501 | 0.2280 | 0.6864 | 0.1882 |
| 3 | 0.5496 | 0.1314 | 0.2157 | 0.7123 | 0.2222 |
| 4 | 0.6019 | 0.1357 | 0.1984 | 0.7078 | 0.2000 |
| 5 | 0.6256 | 0.2091 | 0.2214 | 0.7123 | 0.2921 |
| **Mean** | **0.5867** | **0.1561** | **0.2170** | **0.7019** | **0.2224** |
| **±Std** | 0.0256 | 0.0278 | 0.0101 | 0.0111 | 0.0366 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8512 | 0.4125 | 0.1646 | 0.7818 | 0.4146 |
| 2 | 0.8263 | 0.4156 | 0.1504 | 0.7682 | 0.3704 |
| 3 | 0.7669 | 0.3371 | 0.2514 | 0.6073 | 0.3175 |
| 4 | 0.7714 | 0.2671 | 0.2348 | 0.6301 | 0.3306 |
| 5 | 0.8476 | 0.4547 | 0.1706 | 0.7443 | 0.4167 |
| **Mean** | **0.8127** | **0.3774** | **0.1944** | **0.7063** | **0.3699** |
| **±Std** | 0.0366 | 0.0670 | 0.0407 | 0.0729 | 0.0412 |

CrossAttn best val AUC per fold: Fold1=0.8512, Fold2=0.8263, Fold3=0.7669, Fold4=0.7714, Fold5=0.8476

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5974 | 0.1346 | 0.2237 | 0.6764 | 0.2124 |
| CrossAttn | 0.7420 | 0.2902 | 0.2688 | 0.5818 | 0.2945 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.5653 | 0.1875 | 0.2629 | 0.6339 | 0.2807 |
| F | 163 | 0.5892 | 0.1000 | 0.1967 | 0.7055 | 0.1429 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7201 | 0.3337 | 0.3509 | 0.4554 | 0.3146 |
| F | 163 | 0.7421 | 0.2715 | 0.2123 | 0.6687 | 0.2703 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 174 | 71 |
| **True: Sarco**  | 18 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 136 | 109 |
| **True: Sarco**  | 6 | 24 |

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
