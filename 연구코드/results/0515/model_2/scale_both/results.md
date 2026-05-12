# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 17:04  |  5-Fold CV  |  Median best epoch: 5

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
| 1 | 0.7328 | 0.2586 | 0.2129 | 0.6818 | 0.3269 |
| 2 | 0.7747 | 0.3110 | 0.1642 | 0.7636 | 0.3333 |
| 3 | 0.7699 | 0.3390 | 0.1970 | 0.7169 | 0.3542 |
| 4 | 0.7331 | 0.2289 | 0.1732 | 0.7580 | 0.3291 |
| 5 | 0.8434 | 0.4530 | 0.1601 | 0.7808 | 0.4286 |
| **Mean** | **0.7708** | **0.3181** | **0.1815** | **0.7402** | **0.3544** |
| **±Std** | 0.0404 | 0.0777 | 0.0202 | 0.0360 | 0.0383 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8380 | 0.4015 | 0.2434 | 0.6045 | 0.3459 |
| 2 | 0.8250 | 0.4210 | 0.1567 | 0.7273 | 0.4000 |
| 3 | 0.7816 | 0.3119 | 0.2162 | 0.6575 | 0.3590 |
| 4 | 0.7600 | 0.3131 | 0.1472 | 0.7717 | 0.3750 |
| 5 | 0.8722 | 0.5599 | 0.2163 | 0.6164 | 0.3538 |
| **Mean** | **0.8154** | **0.4015** | **0.1959** | **0.6755** | **0.3667** |
| **±Std** | 0.0401 | 0.0909 | 0.0374 | 0.0645 | 0.0192 |

CrossAttn best val AUC per fold: Fold1=0.8380, Fold2=0.8250, Fold3=0.7816, Fold4=0.7600, Fold5=0.8722

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7388 | 0.3377 | 0.1919 | 0.7273 | 0.3590 |
| CrossAttn | 0.7527 | 0.3190 | 0.2794 | 0.4582 | 0.2513 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7659 | 0.4854 | 0.2643 | 0.5893 | 0.3784 |
| F | 163 | 0.7169 | 0.1874 | 0.1421 | 0.8221 | 0.3256 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7418 | 0.3963 | 0.3252 | 0.4554 | 0.3146 |
| F | 163 | 0.7600 | 0.2669 | 0.2480 | 0.4601 | 0.2000 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 179 | 66 |
| **True: Sarco**  | 9 | 21 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 101 | 144 |
| **True: Sarco**  | 5 | 25 |

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
