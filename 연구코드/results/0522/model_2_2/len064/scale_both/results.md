# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:22  |  5-Fold CV  |  Median best epoch: 5

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 355 | 292 | 82.3% | 63 | 17.7% |
| Train | F | 661 | 614 | 92.9% | 47 | 7.1% |
| Train | **All** | **1016** | **906** | **89.2%** | **110** | **10.8%** |
| Test | M | 97 | 83 | 85.6% | 14 | 14.4% |
| Test | F | 158 | 145 | 91.8% | 13 | 8.2% |
| Test | **All** | **255** | **228** | **89.4%** | **27** | **10.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 355 | 59.92 ± 12.67 | 18.00 | 60.00 | 89.00 |
| Train | F | 661 | 55.55 ± 11.94 | 18.00 | 55.00 | 91.00 |
| Train | **All** | **1016** | **57.07 ± 12.38** | **18.00** | **57.00** | **91.00** |
| Test | M | 97 | 58.63 ± 12.43 | 28.00 | 59.00 | 88.00 |
| Test | F | 158 | 55.27 ± 11.46 | 23.00 | 56.00 | 86.00 |
| Test | **All** | **255** | **56.55 ± 11.95** | **23.00** | **57.00** | **88.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 355 | 24.22 ± 3.38 | 14.48 | 24.16 | 36.76 |
| Train | F | 661 | 23.14 ± 3.39 | 14.40 | 22.83 | 36.24 |
| Train | **All** | **1016** | **23.52 ± 3.42** | **14.40** | **23.37** | **36.76** |
| Test | M | 97 | 24.50 ± 3.14 | 18.37 | 24.49 | 35.68 |
| Test | F | 158 | 23.11 ± 3.24 | 16.87 | 22.72 | 34.23 |
| Test | **All** | **255** | **23.64 ± 3.27** | **16.87** | **23.34** | **35.68** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7582 | 0.2986 | 0.1611 | 0.7696 | 0.3733 |
| 2 | 0.8257 | 0.4328 | 0.1489 | 0.7980 | 0.4384 |
| 3 | 0.8684 | 0.3753 | 0.1500 | 0.8079 | 0.4800 |
| 4 | 0.8451 | 0.4329 | 0.1746 | 0.7734 | 0.4524 |
| 5 | 0.8147 | 0.3728 | 0.1756 | 0.7192 | 0.3736 |
| **Mean** | **0.8224** | **0.3825** | **0.1621** | **0.7736** | **0.4235** |
| **±Std** | 0.0369 | 0.0495 | 0.0115 | 0.0308 | 0.0430 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8002 | 0.3625 | 0.1845 | 0.7010 | 0.3441 |
| 2 | 0.8669 | 0.5379 | 0.1326 | 0.8079 | 0.4507 |
| 3 | 0.8034 | 0.3490 | 0.1545 | 0.7783 | 0.3478 |
| 4 | 0.8586 | 0.6238 | 0.2144 | 0.6700 | 0.3619 |
| 5 | 0.8516 | 0.3896 | 0.1768 | 0.7389 | 0.4045 |
| **Mean** | **0.8361** | **0.4526** | **0.1726** | **0.7392** | **0.3818** |
| **±Std** | 0.0285 | 0.1090 | 0.0277 | 0.0500 | 0.0406 |

CrossAttn best val AUC per fold: Fold1=0.8002, Fold2=0.8669, Fold3=0.8034, Fold4=0.8586, Fold5=0.8516

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6983 | 0.1945 | 0.1803 | 0.7569 | 0.2791 |
| CrossAttn | 0.7420 | 0.2534 | 0.2202 | 0.6588 | 0.3150 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7255 | 0.2566 | 0.2391 | 0.6907 | 0.4000 |
| F | 158 | 0.6387 | 0.1292 | 0.1442 | 0.7975 | 0.1111 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7556 | 0.3203 | 0.2450 | 0.6598 | 0.3774 |
| F | 158 | 0.7082 | 0.1748 | 0.2049 | 0.6582 | 0.2703 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 181 | 47 |
| **True: Sarco**  | 15 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 148 | 80 |
| **True: Sarco**  | 7 | 20 |

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
