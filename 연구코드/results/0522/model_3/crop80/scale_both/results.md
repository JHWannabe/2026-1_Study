# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:19  |  5-Fold CV  |  Median best epoch: 14

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
| 1 | 0.8057 | 0.4244 | 0.1624 | 0.7549 | 0.3902 |
| 2 | 0.8157 | 0.3566 | 0.1582 | 0.7635 | 0.3846 |
| 3 | 0.8169 | 0.3117 | 0.1595 | 0.7980 | 0.4225 |
| 4 | 0.7926 | 0.4079 | 0.1859 | 0.7094 | 0.3516 |
| 5 | 0.8187 | 0.3989 | 0.1715 | 0.7635 | 0.4000 |
| **Mean** | **0.8099** | **0.3799** | **0.1675** | **0.7579** | **0.3898** |
| **±Std** | 0.0098 | 0.0408 | 0.0103 | 0.0284 | 0.0231 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8524 | 0.4057 | 0.1791 | 0.7402 | 0.4045 |
| 2 | 0.8619 | 0.4302 | 0.1836 | 0.7389 | 0.4045 |
| 3 | 0.8167 | 0.3541 | 0.2225 | 0.6650 | 0.3585 |
| 4 | 0.8719 | 0.6350 | 0.1671 | 0.7291 | 0.4086 |
| 5 | 0.8378 | 0.3994 | 0.2305 | 0.7143 | 0.3958 |
| **Mean** | **0.8481** | **0.4449** | **0.1965** | **0.7175** | **0.3944** |
| **±Std** | 0.0193 | 0.0982 | 0.0252 | 0.0278 | 0.0184 |

CrossAttn best val AUC per fold: Fold1=0.8524, Fold2=0.8619, Fold3=0.8167, Fold4=0.8719, Fold5=0.8378

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7341 | 0.2692 | 0.1772 | 0.7529 | 0.2921 |
| CrossAttn | 0.7212 | 0.2838 | 0.1884 | 0.7333 | 0.3200 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7582 | 0.3670 | 0.2284 | 0.6804 | 0.3673 |
| F | 158 | 0.6801 | 0.1679 | 0.1457 | 0.7975 | 0.2000 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7375 | 0.3589 | 0.2705 | 0.6186 | 0.3934 |
| F | 158 | 0.6504 | 0.2369 | 0.1379 | 0.8038 | 0.2051 |

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
| **True: Normal** | 171 | 57 |
| **True: Sarco**  | 11 | 16 |

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
