# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:09  |  5-Fold CV  |  Median best epoch: 15

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
| 1 | 0.7185 | 0.2185 | 0.1869 | 0.7304 | 0.3038 |
| 2 | 0.6261 | 0.1562 | 0.2077 | 0.7340 | 0.2500 |
| 3 | 0.7165 | 0.2278 | 0.1933 | 0.7438 | 0.2973 |
| 4 | 0.6570 | 0.2862 | 0.1746 | 0.7586 | 0.3288 |
| 5 | 0.6105 | 0.2146 | 0.2186 | 0.7044 | 0.2683 |
| **Mean** | **0.6657** | **0.2207** | **0.1962** | **0.7343** | **0.2896** |
| **±Std** | 0.0449 | 0.0414 | 0.0154 | 0.0178 | 0.0276 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8544 | 0.4559 | 0.1459 | 0.7745 | 0.4250 |
| 2 | 0.8551 | 0.3694 | 0.1647 | 0.7635 | 0.4419 |
| 3 | 0.8312 | 0.3681 | 0.1698 | 0.7734 | 0.3784 |
| 4 | 0.8594 | 0.5725 | 0.1673 | 0.7291 | 0.4086 |
| 5 | 0.8473 | 0.3586 | 0.2016 | 0.7094 | 0.4040 |
| **Mean** | **0.8495** | **0.4249** | **0.1699** | **0.7500** | **0.4116** |
| **±Std** | 0.0099 | 0.0818 | 0.0179 | 0.0262 | 0.0213 |

CrossAttn best val AUC per fold: Fold1=0.8544, Fold2=0.8551, Fold3=0.8312, Fold4=0.8594, Fold5=0.8473

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5746 | 0.2031 | 0.2215 | 0.6902 | 0.2020 |
| CrossAttn | 0.7217 | 0.2657 | 0.1835 | 0.7333 | 0.2609 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.5568 | 0.2491 | 0.2477 | 0.6907 | 0.3182 |
| F | 158 | 0.5889 | 0.1753 | 0.2054 | 0.6899 | 0.1091 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7263 | 0.3702 | 0.2570 | 0.6289 | 0.3077 |
| F | 158 | 0.6912 | 0.1686 | 0.1383 | 0.7975 | 0.2000 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 166 | 62 |
| **True: Sarco**  | 17 | 10 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 175 | 53 |
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
