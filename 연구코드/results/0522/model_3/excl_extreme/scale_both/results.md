# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 20:11  |  5-Fold CV  |  Median best epoch: 6

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 309 | 261 | 84.5% | 48 | 15.5% |
| Train | F | 605 | 559 | 92.4% | 46 | 7.6% |
| Train | **All** | **914** | **820** | **89.7%** | **94** | **10.3%** |
| Test | M | 83 | 70 | 84.3% | 13 | 15.7% |
| Test | F | 146 | 133 | 91.1% | 13 | 8.9% |
| Test | **All** | **229** | **203** | **88.6%** | **26** | **11.4%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 309 | 59.60 ± 12.50 | 18.00 | 60.00 | 89.00 |
| Train | F | 605 | 55.60 ± 11.93 | 18.00 | 55.00 | 91.00 |
| Train | **All** | **914** | **56.95 ± 12.27** | **18.00** | **57.00** | **91.00** |
| Test | M | 83 | 59.20 ± 12.71 | 28.00 | 60.00 | 88.00 |
| Test | F | 146 | 54.76 ± 11.42 | 23.00 | 55.00 | 86.00 |
| Test | **All** | **229** | **56.37 ± 12.09** | **23.00** | **57.00** | **88.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 309 | 24.01 ± 3.23 | 14.48 | 24.07 | 35.20 |
| Train | F | 605 | 23.00 ± 3.25 | 14.40 | 22.72 | 36.24 |
| Train | **All** | **914** | **23.34 ± 3.27** | **14.40** | **23.24** | **36.24** |
| Test | M | 83 | 24.36 ± 2.96 | 18.37 | 24.39 | 33.87 |
| Test | F | 146 | 22.97 ± 3.08 | 16.87 | 22.65 | 34.23 |
| Test | **All** | **229** | **23.47 ± 3.11** | **16.87** | **23.28** | **34.23** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7413 | 0.2410 | 0.1472 | 0.7869 | 0.2909 |
| 2 | 0.7757 | 0.3661 | 0.1746 | 0.7596 | 0.3333 |
| 3 | 0.8383 | 0.4290 | 0.1569 | 0.7650 | 0.4110 |
| 4 | 0.7747 | 0.3907 | 0.1797 | 0.7213 | 0.3377 |
| 5 | 0.7060 | 0.2605 | 0.1908 | 0.6978 | 0.2667 |
| **Mean** | **0.7672** | **0.3375** | **0.1698** | **0.7461** | **0.3279** |
| **±Std** | 0.0438 | 0.0738 | 0.0158 | 0.0321 | 0.0493 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7981 | 0.3939 | 0.1971 | 0.7213 | 0.3704 |
| 2 | 0.8623 | 0.4464 | 0.1725 | 0.7541 | 0.4156 |
| 3 | 0.9076 | 0.6293 | 0.1457 | 0.7869 | 0.4658 |
| 4 | 0.7949 | 0.3445 | 0.2462 | 0.6612 | 0.3542 |
| 5 | 0.8692 | 0.3554 | 0.1389 | 0.8187 | 0.4762 |
| **Mean** | **0.8464** | **0.4339** | **0.1801** | **0.7484** | **0.4164** |
| **±Std** | 0.0436 | 0.1040 | 0.0390 | 0.0544 | 0.0490 |

CrossAttn best val AUC per fold: Fold1=0.7981, Fold2=0.8623, Fold3=0.9076, Fold4=0.7949, Fold5=0.8692

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7116 | 0.2666 | 0.1789 | 0.7467 | 0.3095 |
| CrossAttn | 0.7232 | 0.2703 | 0.1822 | 0.7336 | 0.2824 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 83 | 0.7242 | 0.3781 | 0.2152 | 0.6988 | 0.3902 |
| F | 146 | 0.6842 | 0.1711 | 0.1582 | 0.7740 | 0.2326 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 83 | 0.7363 | 0.3644 | 0.2293 | 0.6506 | 0.3256 |
| F | 146 | 0.6865 | 0.1955 | 0.1554 | 0.7808 | 0.2381 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 158 | 45 |
| **True: Sarco**  | 13 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 156 | 47 |
| **True: Sarco**  | 14 | 12 |

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
