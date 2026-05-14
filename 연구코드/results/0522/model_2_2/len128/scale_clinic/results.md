# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:31  |  5-Fold CV  |  Median best epoch: 7

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
| 1 | 0.4928 | 0.1193 | 0.2256 | 0.6618 | 0.1039 |
| 2 | 0.6695 | 0.1918 | 0.1860 | 0.7438 | 0.2778 |
| 3 | 0.6801 | 0.2492 | 0.2019 | 0.7143 | 0.2750 |
| 4 | 0.6740 | 0.2815 | 0.2140 | 0.7143 | 0.3095 |
| 5 | 0.6843 | 0.1877 | 0.2038 | 0.6946 | 0.2619 |
| **Mean** | **0.6401** | **0.2059** | **0.2063** | **0.7058** | **0.2456** |
| **±Std** | 0.0739 | 0.0559 | 0.0132 | 0.0270 | 0.0726 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8249 | 0.4465 | 0.2091 | 0.6765 | 0.3529 |
| 2 | 0.8506 | 0.4519 | 0.1725 | 0.7094 | 0.3789 |
| 3 | 0.8144 | 0.3425 | 0.1957 | 0.7192 | 0.3736 |
| 4 | 0.8692 | 0.5939 | 0.1656 | 0.7635 | 0.4146 |
| 5 | 0.8152 | 0.3334 | 0.1397 | 0.8128 | 0.4571 |
| **Mean** | **0.8348** | **0.4337** | **0.1765** | **0.7363** | **0.3955** |
| **±Std** | 0.0216 | 0.0944 | 0.0242 | 0.0473 | 0.0367 |

CrossAttn best val AUC per fold: Fold1=0.8249, Fold2=0.8506, Fold3=0.8144, Fold4=0.8692, Fold5=0.8152

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5416 | 0.1663 | 0.2096 | 0.7137 | 0.1978 |
| CrossAttn | 0.7736 | 0.2453 | 0.1774 | 0.7569 | 0.3542 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6377 | 0.2737 | 0.2141 | 0.6907 | 0.2500 |
| F | 158 | 0.4541 | 0.1214 | 0.2069 | 0.7278 | 0.1569 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7745 | 0.2922 | 0.2350 | 0.7010 | 0.4528 |
| F | 158 | 0.7422 | 0.1856 | 0.1420 | 0.7911 | 0.2326 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 173 | 55 |
| **True: Sarco**  | 18 | 9 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 176 | 52 |
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
