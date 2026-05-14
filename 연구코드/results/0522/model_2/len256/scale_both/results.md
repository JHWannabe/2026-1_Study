# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:59  |  5-Fold CV  |  Median best epoch: 9

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
| 1 | 0.7870 | 0.4100 | 0.1702 | 0.7353 | 0.3571 |
| 2 | 0.7833 | 0.3167 | 0.1654 | 0.7537 | 0.3590 |
| 3 | 0.8189 | 0.3103 | 0.1578 | 0.7783 | 0.3836 |
| 4 | 0.7880 | 0.4055 | 0.1845 | 0.7143 | 0.3556 |
| 5 | 0.7993 | 0.3862 | 0.1728 | 0.7783 | 0.4156 |
| **Mean** | **0.7953** | **0.3657** | **0.1701** | **0.7520** | **0.3742** |
| **±Std** | 0.0130 | 0.0434 | 0.0088 | 0.0249 | 0.0231 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8319 | 0.4764 | 0.1459 | 0.7794 | 0.4000 |
| 2 | 0.8576 | 0.5384 | 0.1764 | 0.7438 | 0.4222 |
| 3 | 0.8235 | 0.2842 | 0.2018 | 0.7044 | 0.3878 |
| 4 | 0.8576 | 0.5463 | 0.1800 | 0.7094 | 0.3918 |
| 5 | 0.8385 | 0.3633 | 0.1891 | 0.7488 | 0.4270 |
| **Mean** | **0.8418** | **0.4417** | **0.1786** | **0.7372** | **0.4057** |
| **±Std** | 0.0137 | 0.1024 | 0.0186 | 0.0276 | 0.0160 |

CrossAttn best val AUC per fold: Fold1=0.8319, Fold2=0.8576, Fold3=0.8235, Fold4=0.8576, Fold5=0.8385

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7261 | 0.2230 | 0.1776 | 0.7490 | 0.2889 |
| CrossAttn | 0.7614 | 0.3059 | 0.2090 | 0.6902 | 0.3130 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7478 | 0.2867 | 0.2298 | 0.6701 | 0.3600 |
| F | 158 | 0.6695 | 0.1647 | 0.1455 | 0.7975 | 0.2000 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7642 | 0.3999 | 0.2696 | 0.6082 | 0.3667 |
| F | 158 | 0.7401 | 0.2347 | 0.1718 | 0.7405 | 0.2545 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 178 | 50 |
| **True: Sarco**  | 14 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 158 | 70 |
| **True: Sarco**  | 9 | 18 |

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
