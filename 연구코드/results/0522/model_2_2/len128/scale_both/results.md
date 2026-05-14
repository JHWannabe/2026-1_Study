# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:40  |  5-Fold CV  |  Median best epoch: 4

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
| 1 | 0.7545 | 0.2977 | 0.1598 | 0.7598 | 0.3467 |
| 2 | 0.8257 | 0.4299 | 0.1469 | 0.8030 | 0.4444 |
| 3 | 0.8734 | 0.3775 | 0.1481 | 0.8177 | 0.4932 |
| 4 | 0.8483 | 0.4380 | 0.1713 | 0.7734 | 0.4524 |
| 5 | 0.8159 | 0.3803 | 0.1739 | 0.7438 | 0.3953 |
| **Mean** | **0.8236** | **0.3847** | **0.1600** | **0.7795** | **0.4264** |
| **±Std** | 0.0398 | 0.0500 | 0.0113 | 0.0272 | 0.0506 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8032 | 0.4040 | 0.1513 | 0.7745 | 0.4103 |
| 2 | 0.8521 | 0.4567 | 0.1478 | 0.7783 | 0.4304 |
| 3 | 0.8167 | 0.2918 | 0.2045 | 0.6897 | 0.3636 |
| 4 | 0.8584 | 0.6813 | 0.1887 | 0.7340 | 0.4000 |
| 5 | 0.8471 | 0.4008 | 0.1842 | 0.7241 | 0.4043 |
| **Mean** | **0.8355** | **0.4469** | **0.1753** | **0.7401** | **0.4017** |
| **±Std** | 0.0216 | 0.1289 | 0.0221 | 0.0331 | 0.0217 |

CrossAttn best val AUC per fold: Fold1=0.8032, Fold2=0.8521, Fold3=0.8167, Fold4=0.8584, Fold5=0.8471

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6927 | 0.1934 | 0.1799 | 0.7569 | 0.2619 |
| CrossAttn | 0.7537 | 0.2633 | 0.1829 | 0.7059 | 0.3119 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7203 | 0.2567 | 0.2393 | 0.6804 | 0.3673 |
| F | 158 | 0.6361 | 0.1282 | 0.1434 | 0.8038 | 0.1143 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7694 | 0.3279 | 0.2215 | 0.6598 | 0.3774 |
| F | 158 | 0.7263 | 0.1780 | 0.1593 | 0.7342 | 0.2500 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 182 | 46 |
| **True: Sarco**  | 16 | 11 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 163 | 65 |
| **True: Sarco**  | 10 | 17 |

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
