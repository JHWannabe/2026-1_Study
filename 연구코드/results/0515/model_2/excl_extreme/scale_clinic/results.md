# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:47  |  5-Fold CV  |  Median best epoch: 19

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 311 | 262 | 84.2% | 49 | 15.8% |
| Train | F | 604 | 556 | 92.1% | 48 | 7.9% |
| Train | **All** | **915** | **818** | **89.4%** | **97** | **10.6%** |
| Test | M | 86 | 73 | 84.9% | 13 | 15.1% |
| Test | F | 143 | 132 | 92.3% | 11 | 7.7% |
| Test | **All** | **229** | **205** | **89.5%** | **24** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 311 | 59.30 ± 12.73 | 18.00 | 60.00 | 88.00 |
| Train | F | 604 | 55.54 ± 11.62 | 14.00 | 55.50 | 91.00 |
| Train | **All** | **915** | **56.82 ± 12.14** | **14.00** | **57.00** | **91.00** |
| Test | M | 86 | 61.66 ± 10.80 | 34.00 | 61.00 | 89.00 |
| Test | F | 143 | 55.24 ± 13.24 | 11.00 | 54.00 | 86.00 |
| Test | **All** | **229** | **57.65 ± 12.77** | **11.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 311 | 23.91 ± 3.12 | 14.48 | 24.03 | 35.20 |
| Train | F | 604 | 22.91 ± 3.10 | 15.62 | 22.64 | 34.29 |
| Train | **All** | **915** | **23.25 ± 3.14** | **14.48** | **23.09** | **35.20** |
| Test | M | 86 | 24.09 ± 3.18 | 16.80 | 24.42 | 33.87 |
| Test | F | 143 | 23.04 ± 3.49 | 14.40 | 22.53 | 36.24 |
| Test | **All** | **229** | **23.44 ± 3.42** | **14.40** | **23.38** | **36.24** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.5424 | 0.1740 | 0.1944 | 0.7322 | 0.1967 |
| 2 | 0.6297 | 0.1811 | 0.2210 | 0.7213 | 0.2609 |
| 3 | 0.5501 | 0.1538 | 0.2079 | 0.6776 | 0.1449 |
| 4 | 0.5393 | 0.1294 | 0.2070 | 0.7049 | 0.2059 |
| 5 | 0.5684 | 0.1440 | 0.2314 | 0.6667 | 0.1159 |
| **Mean** | **0.5659** | **0.1565** | **0.2123** | **0.7005** | **0.1849** |
| **±Std** | 0.0334 | 0.0190 | 0.0127 | 0.0250 | 0.0504 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7660 | 0.2945 | 0.1653 | 0.7650 | 0.3385 |
| 2 | 0.7863 | 0.2947 | 0.2672 | 0.5355 | 0.2975 |
| 3 | 0.7847 | 0.3856 | 0.1159 | 0.8579 | 0.4348 |
| 4 | 0.8883 | 0.5850 | 0.2115 | 0.6503 | 0.3469 |
| 5 | 0.8359 | 0.4458 | 0.1783 | 0.7322 | 0.4096 |
| **Mean** | **0.8122** | **0.4011** | **0.1876** | **0.7082** | **0.3655** |
| **±Std** | 0.0445 | 0.1084 | 0.0503 | 0.1090 | 0.0499 |

CrossAttn best val AUC per fold: Fold1=0.7660, Fold2=0.7863, Fold3=0.7847, Fold4=0.8883, Fold5=0.8359

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5610 | 0.1908 | 0.2110 | 0.7118 | 0.1951 |
| CrossAttn | 0.8325 | 0.4029 | 0.1696 | 0.7293 | 0.3673 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 86 | 0.5258 | 0.3184 | 0.2595 | 0.6512 | 0.2105 |
| F | 143 | 0.5854 | 0.1281 | 0.1818 | 0.7483 | 0.1818 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 86 | 0.8240 | 0.4383 | 0.1987 | 0.7093 | 0.4681 |
| F | 143 | 0.8306 | 0.3725 | 0.1522 | 0.7413 | 0.2745 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 155 | 50 |
| **True: Sarco**  | 16 | 8 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 149 | 56 |
| **True: Sarco**  | 6 | 18 |

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
