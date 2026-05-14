# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:17  |  5-Fold CV  |  Median best epoch: 10

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
| 1 | 0.8047 | 0.4232 | 0.1639 | 0.7647 | 0.3846 |
| 2 | 0.8104 | 0.3536 | 0.1588 | 0.7734 | 0.3947 |
| 3 | 0.8209 | 0.3110 | 0.1573 | 0.7882 | 0.3944 |
| 4 | 0.7893 | 0.4041 | 0.1881 | 0.7143 | 0.3556 |
| 5 | 0.8109 | 0.3878 | 0.1734 | 0.7685 | 0.4051 |
| **Mean** | **0.8072** | **0.3759** | **0.1683** | **0.7618** | **0.3869** |
| **±Std** | 0.0104 | 0.0397 | 0.0114 | 0.0251 | 0.0169 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8347 | 0.4117 | 0.1742 | 0.7304 | 0.3956 |
| 2 | 0.8639 | 0.4690 | 0.1675 | 0.7438 | 0.4091 |
| 3 | 0.8101 | 0.3153 | 0.1541 | 0.7685 | 0.2985 |
| 4 | 0.8493 | 0.5166 | 0.2292 | 0.6108 | 0.3248 |
| 5 | 0.8240 | 0.3460 | 0.1787 | 0.7586 | 0.4096 |
| **Mean** | **0.8364** | **0.4117** | **0.1807** | **0.7224** | **0.3675** |
| **±Std** | 0.0188 | 0.0747 | 0.0256 | 0.0573 | 0.0466 |

CrossAttn best val AUC per fold: Fold1=0.8347, Fold2=0.8639, Fold3=0.8101, Fold4=0.8493, Fold5=0.8240

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7313 | 0.2808 | 0.1767 | 0.7529 | 0.2921 |
| CrossAttn | 0.7640 | 0.2844 | 0.1326 | 0.8078 | 0.3797 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7608 | 0.3967 | 0.2268 | 0.6804 | 0.3922 |
| F | 158 | 0.6764 | 0.1632 | 0.1459 | 0.7975 | 0.1579 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7737 | 0.3579 | 0.1810 | 0.7113 | 0.4400 |
| F | 158 | 0.7247 | 0.2061 | 0.1029 | 0.8671 | 0.2759 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 179 | 49 |
| **True: Sarco**  | 14 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 191 | 37 |
| **True: Sarco**  | 12 | 15 |

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
