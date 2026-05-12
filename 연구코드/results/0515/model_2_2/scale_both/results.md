# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 17:04  |  5-Fold CV  |  Median best epoch: 9

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
| 1 | 0.7966 | 0.2953 | 0.1975 | 0.6864 | 0.3429 |
| 2 | 0.8123 | 0.3160 | 0.1477 | 0.7909 | 0.3611 |
| 3 | 0.7885 | 0.3577 | 0.1945 | 0.7215 | 0.3711 |
| 4 | 0.7060 | 0.2081 | 0.1756 | 0.7397 | 0.2785 |
| 5 | 0.8188 | 0.4361 | 0.1699 | 0.7534 | 0.4130 |
| **Mean** | **0.7844** | **0.3227** | **0.1770** | **0.7384** | **0.3533** |
| **±Std** | 0.0407 | 0.0748 | 0.0181 | 0.0346 | 0.0439 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8389 | 0.4155 | 0.1565 | 0.7682 | 0.4138 |
| 2 | 0.8131 | 0.3799 | 0.2015 | 0.6864 | 0.3429 |
| 3 | 0.7906 | 0.2694 | 0.2277 | 0.6849 | 0.3551 |
| 4 | 0.7600 | 0.2566 | 0.1667 | 0.7626 | 0.3333 |
| 5 | 0.8549 | 0.5391 | 0.1679 | 0.7306 | 0.3918 |
| **Mean** | **0.8115** | **0.3721** | **0.1841** | **0.7265** | **0.3674** |
| **±Std** | 0.0338 | 0.1037 | 0.0266 | 0.0358 | 0.0305 |

CrossAttn best val AUC per fold: Fold1=0.8389, Fold2=0.8131, Fold3=0.7906, Fold4=0.7600, Fold5=0.8549

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7259 | 0.2974 | 0.1920 | 0.7236 | 0.3559 |
| CrossAttn | 0.7735 | 0.3565 | 0.1989 | 0.6909 | 0.3411 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.6947 | 0.4251 | 0.2649 | 0.5982 | 0.3662 |
| F | 163 | 0.7308 | 0.2274 | 0.1420 | 0.8098 | 0.3404 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7579 | 0.4332 | 0.2637 | 0.5982 | 0.3836 |
| F | 163 | 0.7774 | 0.2936 | 0.1543 | 0.7546 | 0.2857 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 178 | 67 |
| **True: Sarco**  | 9 | 21 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 77 |
| **True: Sarco**  | 8 | 22 |

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
