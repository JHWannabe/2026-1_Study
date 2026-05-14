# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:47  |  5-Fold CV  |  Median best epoch: 21

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
| 1 | 0.7712 | 0.4050 | 0.1681 | 0.7402 | 0.3457 |
| 2 | 0.8104 | 0.3230 | 0.1574 | 0.7833 | 0.4054 |
| 3 | 0.7692 | 0.2790 | 0.1620 | 0.7734 | 0.3611 |
| 4 | 0.7737 | 0.3087 | 0.1756 | 0.7537 | 0.3750 |
| 5 | 0.7760 | 0.3586 | 0.1747 | 0.7635 | 0.3684 |
| **Mean** | **0.7801** | **0.3349** | **0.1676** | **0.7628** | **0.3711** |
| **±Std** | 0.0153 | 0.0434 | 0.0071 | 0.0150 | 0.0197 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8444 | 0.5324 | 0.1460 | 0.7843 | 0.4359 |
| 2 | 0.8423 | 0.5082 | 0.1757 | 0.7340 | 0.4130 |
| 3 | 0.8317 | 0.3534 | 0.2509 | 0.6355 | 0.3509 |
| 4 | 0.8900 | 0.6317 | 0.2424 | 0.6010 | 0.3306 |
| 5 | 0.8408 | 0.3496 | 0.1367 | 0.8424 | 0.5152 |
| **Mean** | **0.8498** | **0.4751** | **0.1903** | **0.7194** | **0.4091** |
| **±Std** | 0.0205 | 0.1090 | 0.0478 | 0.0901 | 0.0656 |

CrossAttn best val AUC per fold: Fold1=0.8444, Fold2=0.8423, Fold3=0.8317, Fold4=0.8900, Fold5=0.8408

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7071 | 0.2100 | 0.1753 | 0.7608 | 0.2824 |
| CrossAttn | 0.6652 | 0.2463 | 0.1783 | 0.7608 | 0.2824 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7048 | 0.2524 | 0.2356 | 0.6495 | 0.3462 |
| F | 158 | 0.6780 | 0.2126 | 0.1383 | 0.8291 | 0.1818 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7556 | 0.3611 | 0.2316 | 0.6289 | 0.3571 |
| F | 158 | 0.5034 | 0.1763 | 0.1457 | 0.8418 | 0.1379 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 182 | 46 |
| **True: Sarco**  | 15 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 182 | 46 |
| **True: Sarco**  | 15 | 12 |

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
