# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:38  |  5-Fold CV  |  Median best epoch: 20

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
| 1 | 0.7927 | 0.3882 | 0.1650 | 0.7549 | 0.3750 |
| 2 | 0.8089 | 0.3463 | 0.1592 | 0.7931 | 0.4167 |
| 3 | 0.8290 | 0.3099 | 0.1600 | 0.7882 | 0.4110 |
| 4 | 0.8147 | 0.4328 | 0.1865 | 0.7143 | 0.3696 |
| 5 | 0.8041 | 0.3927 | 0.1769 | 0.7685 | 0.4051 |
| **Mean** | **0.8099** | **0.3740** | **0.1695** | **0.7638** | **0.3955** |
| **±Std** | 0.0120 | 0.0421 | 0.0106 | 0.0283 | 0.0193 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8392 | 0.4448 | 0.1282 | 0.8088 | 0.4348 |
| 2 | 0.8493 | 0.4601 | 0.1531 | 0.7882 | 0.4691 |
| 3 | 0.8137 | 0.3755 | 0.1158 | 0.8325 | 0.4138 |
| 4 | 0.8621 | 0.6325 | 0.1763 | 0.7389 | 0.4176 |
| 5 | 0.8458 | 0.3705 | 0.2040 | 0.7192 | 0.4124 |
| **Mean** | **0.8420** | **0.4567** | **0.1555** | **0.7775** | **0.4295** |
| **±Std** | 0.0160 | 0.0949 | 0.0320 | 0.0424 | 0.0214 |

CrossAttn best val AUC per fold: Fold1=0.8392, Fold2=0.8493, Fold3=0.8137, Fold4=0.8621, Fold5=0.8458

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7484 | 0.2290 | 0.1775 | 0.7529 | 0.3077 |
| CrossAttn | 0.6720 | 0.2352 | 0.1809 | 0.7608 | 0.2824 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7608 | 0.3078 | 0.2307 | 0.6598 | 0.3774 |
| F | 158 | 0.7088 | 0.1554 | 0.1449 | 0.8101 | 0.2105 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6885 | 0.3130 | 0.2775 | 0.6082 | 0.3214 |
| F | 158 | 0.6281 | 0.2001 | 0.1216 | 0.8544 | 0.2069 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 178 | 50 |
| **True: Sarco**  | 13 | 14 |

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
