# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:52  |  5-Fold CV  |  Median best epoch: 5

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
| 1 | 0.6931 | 0.2168 | 0.1668 | 0.7843 | 0.3333 |
| 2 | 0.8307 | 0.4306 | 0.1375 | 0.7882 | 0.3944 |
| 3 | 0.8556 | 0.3678 | 0.1400 | 0.7931 | 0.4167 |
| 4 | 0.8338 | 0.4021 | 0.1692 | 0.7734 | 0.4103 |
| 5 | 0.7971 | 0.3046 | 0.1647 | 0.7833 | 0.4359 |
| **Mean** | **0.8020** | **0.3444** | **0.1556** | **0.7844** | **0.3981** |
| **±Std** | 0.0576 | 0.0764 | 0.0139 | 0.0065 | 0.0350 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8619 | 0.4317 | 0.1721 | 0.7206 | 0.3736 |
| 2 | 0.8634 | 0.3995 | 0.1602 | 0.7389 | 0.4176 |
| 3 | 0.8245 | 0.3528 | 0.1856 | 0.7143 | 0.3556 |
| 4 | 0.8631 | 0.5916 | 0.2159 | 0.6700 | 0.3738 |
| 5 | 0.8255 | 0.3289 | 0.2113 | 0.6946 | 0.3922 |
| **Mean** | **0.8477** | **0.4209** | **0.1890** | **0.7077** | **0.3826** |
| **±Std** | 0.0185 | 0.0925 | 0.0217 | 0.0236 | 0.0210 |

CrossAttn best val AUC per fold: Fold1=0.8619, Fold2=0.8634, Fold3=0.8245, Fold4=0.8631, Fold5=0.8255

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6407 | 0.1876 | 0.1736 | 0.7922 | 0.3457 |
| CrossAttn | 0.7815 | 0.2917 | 0.2253 | 0.6549 | 0.3231 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6463 | 0.2231 | 0.2486 | 0.7010 | 0.4082 |
| F | 158 | 0.5894 | 0.1517 | 0.1275 | 0.8481 | 0.2500 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7823 | 0.3712 | 0.3038 | 0.5361 | 0.3662 |
| F | 158 | 0.7692 | 0.2572 | 0.1771 | 0.7278 | 0.2712 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 188 | 40 |
| **True: Sarco**  | 13 | 14 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 146 | 82 |
| **True: Sarco**  | 6 | 21 |

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
