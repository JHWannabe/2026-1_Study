# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:40  |  5-Fold CV  |  Median best epoch: 20

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
| 1 | 0.7912 | 0.4233 | 0.1681 | 0.7402 | 0.3614 |
| 2 | 0.7896 | 0.3349 | 0.1642 | 0.7586 | 0.3636 |
| 3 | 0.8177 | 0.3083 | 0.1587 | 0.7783 | 0.3836 |
| 4 | 0.7953 | 0.4159 | 0.1867 | 0.7044 | 0.3478 |
| 5 | 0.8066 | 0.3819 | 0.1739 | 0.7734 | 0.4103 |
| **Mean** | **0.8001** | **0.3729** | **0.1704** | **0.7510** | **0.3733** |
| **±Std** | 0.0106 | 0.0449 | 0.0096 | 0.0268 | 0.0217 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8304 | 0.3684 | 0.1513 | 0.7843 | 0.4359 |
| 2 | 0.8749 | 0.4736 | 0.1775 | 0.7143 | 0.3958 |
| 3 | 0.8119 | 0.3361 | 0.1712 | 0.7291 | 0.3529 |
| 4 | 0.8611 | 0.5849 | 0.1570 | 0.7389 | 0.4176 |
| 5 | 0.8320 | 0.3586 | 0.2538 | 0.5862 | 0.3226 |
| **Mean** | **0.8421** | **0.4243** | **0.1822** | **0.7106** | **0.3850** |
| **±Std** | 0.0228 | 0.0932 | 0.0370 | 0.0664 | 0.0417 |

CrossAttn best val AUC per fold: Fold1=0.8304, Fold2=0.8749, Fold3=0.8119, Fold4=0.8611, Fold5=0.8320

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7303 | 0.2514 | 0.1772 | 0.7490 | 0.3043 |
| CrossAttn | 0.7695 | 0.2812 | 0.1961 | 0.7176 | 0.3208 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7573 | 0.3401 | 0.2290 | 0.6701 | 0.3846 |
| F | 158 | 0.6679 | 0.1435 | 0.1454 | 0.7975 | 0.2000 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7599 | 0.3296 | 0.2803 | 0.6186 | 0.3934 |
| F | 158 | 0.7459 | 0.2281 | 0.1444 | 0.7785 | 0.2222 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 177 | 51 |
| **True: Sarco**  | 13 | 14 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 166 | 62 |
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
