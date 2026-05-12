# SMI Binary Classification — Results

Generated: 2026-05-12 16:43  |  5-Fold CV  |  ResNet1D median best epoch: 15

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
| 1 | 0.8342 | 0.3660 | 0.1961 | 0.6864 | 0.3670 |
| 2 | 0.8099 | 0.3375 | 0.1609 | 0.7636 | 0.3810 |
| 3 | 0.7778 | 0.3408 | 0.2042 | 0.6758 | 0.3107 |
| 4 | 0.7216 | 0.2311 | 0.1834 | 0.7260 | 0.3023 |
| 5 | 0.8511 | 0.5248 | 0.1722 | 0.7489 | 0.3956 |
| **Mean** | **0.7989** | **0.3600** | **0.1833** | **0.7201** | **0.3513** |
| **±Std** | 0.0459 | 0.0945 | 0.0157 | 0.0342 | 0.0378 |

### ResNet1D

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8569 | 0.4305 | 0.1990 | 0.6818 | 0.3750 |
| 2 | 0.8442 | 0.4189 | 0.1546 | 0.7455 | 0.4043 |
| 3 | 0.8077 | 0.3580 | 0.1933 | 0.6895 | 0.3704 |
| 4 | 0.7942 | 0.3157 | 0.1861 | 0.7534 | 0.3721 |
| 5 | 0.8551 | 0.5568 | 0.1732 | 0.8128 | 0.4675 |
| **Mean** | **0.8316** | **0.4160** | **0.1812** | **0.7366** | **0.3979** |
| **±Std** | 0.0258 | 0.0818 | 0.0159 | 0.0477 | 0.0370 |

ResNet1D best val AUC per fold: Fold1=0.8569, Fold2=0.8442, Fold3=0.8077, Fold4=0.7942, Fold5=0.8551

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7531 | 0.3357 | 0.1992 | 0.7055 | 0.3306 |
| ResNet1D  | 0.7676 | 0.3389 | 0.2024 | 0.7236 | 0.3448 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7257 | 0.4184 | 0.2805 | 0.5268 | 0.3291 |
| F | 163 | 0.7687 | 0.2505 | 0.1433 | 0.8282 | 0.3333 |

#### ResNet1D

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7412 | 0.4243 | 0.2753 | 0.5982 | 0.3662 |
| F | 163 | 0.7831 | 0.2582 | 0.1522 | 0.8098 | 0.3111 |

---

## 3. Confusion Matrices (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 174 | 71 |
| **True: Sarco**  | 10 | 20 |

### ResNet1D

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 179 | 66 |
| **True: Sarco**  | 10 | 20 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR & ResNet1D) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `confusion_matrices.png` | Test-set confusion matrices (overall) |
| `confusion_matrices_by_sex.png` | Test-set confusion matrices split by sex |
| `training_curves.png` | ResNet1D loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curves (overall) |
| `test_roc_by_sex.png` | Final test-set ROC curves split by sex |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
