# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:00  |  5-Fold CV  |  Median best epoch: 11

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
| 1 | 0.7880 | 0.4168 | 0.1680 | 0.7304 | 0.3529 |
| 2 | 0.7903 | 0.3260 | 0.1649 | 0.7537 | 0.3750 |
| 3 | 0.8061 | 0.3038 | 0.1606 | 0.7931 | 0.4000 |
| 4 | 0.7878 | 0.4090 | 0.1829 | 0.7143 | 0.3556 |
| 5 | 0.8079 | 0.3847 | 0.1707 | 0.7734 | 0.4103 |
| **Mean** | **0.7960** | **0.3681** | **0.1694** | **0.7530** | **0.3788** |
| **±Std** | 0.0090 | 0.0452 | 0.0075 | 0.0284 | 0.0231 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8464 | 0.5373 | 0.1152 | 0.8186 | 0.4478 |
| 2 | 0.8446 | 0.3882 | 0.1544 | 0.7734 | 0.4250 |
| 3 | 0.8001 | 0.3514 | 0.1923 | 0.7143 | 0.3409 |
| 4 | 0.8757 | 0.6185 | 0.1696 | 0.7438 | 0.4348 |
| 5 | 0.8418 | 0.4018 | 0.1970 | 0.7537 | 0.4444 |
| **Mean** | **0.8417** | **0.4595** | **0.1657** | **0.7608** | **0.4186** |
| **±Std** | 0.0242 | 0.1014 | 0.0296 | 0.0347 | 0.0396 |

CrossAttn best val AUC per fold: Fold1=0.8464, Fold2=0.8446, Fold3=0.8001, Fold4=0.8757, Fold5=0.8418

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7292 | 0.2431 | 0.1783 | 0.7451 | 0.2857 |
| CrossAttn | 0.7211 | 0.2151 | 0.1706 | 0.7647 | 0.3182 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7547 | 0.3191 | 0.2310 | 0.6495 | 0.3462 |
| F | 158 | 0.6748 | 0.1694 | 0.1460 | 0.8038 | 0.2051 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7005 | 0.2516 | 0.2481 | 0.6495 | 0.3704 |
| F | 158 | 0.6971 | 0.2167 | 0.1230 | 0.8354 | 0.2353 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 177 | 51 |
| **True: Sarco**  | 14 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 181 | 47 |
| **True: Sarco**  | 13 | 14 |

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
