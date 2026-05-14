# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 18:22  |  5-Fold CV  |  Median best epoch: 20

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
| 1 | 0.7935 | 0.4207 | 0.1651 | 0.7647 | 0.3846 |
| 2 | 0.7976 | 0.3459 | 0.1625 | 0.7537 | 0.3750 |
| 3 | 0.8089 | 0.2998 | 0.1631 | 0.7783 | 0.4000 |
| 4 | 0.7983 | 0.4128 | 0.1871 | 0.7143 | 0.3556 |
| 5 | 0.8199 | 0.3763 | 0.1733 | 0.7586 | 0.3951 |
| **Mean** | **0.8036** | **0.3711** | **0.1702** | **0.7539** | **0.3820** |
| **±Std** | 0.0096 | 0.0446 | 0.0093 | 0.0215 | 0.0158 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8671 | 0.4721 | 0.1499 | 0.7990 | 0.4675 |
| 2 | 0.8543 | 0.3949 | 0.1460 | 0.7980 | 0.4384 |
| 3 | 0.8169 | 0.3257 | 0.1448 | 0.7931 | 0.3438 |
| 4 | 0.8682 | 0.5445 | 0.1513 | 0.8030 | 0.4872 |
| 5 | 0.8405 | 0.3711 | 0.1233 | 0.8128 | 0.4571 |
| **Mean** | **0.8494** | **0.4217** | **0.1430** | **0.8012** | **0.4388** |
| **±Std** | 0.0191 | 0.0776 | 0.0102 | 0.0066 | 0.0501 |

CrossAttn best val AUC per fold: Fold1=0.8671, Fold2=0.8543, Fold3=0.8169, Fold4=0.8682, Fold5=0.8405

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7346 | 0.2731 | 0.1777 | 0.7529 | 0.3077 |
| CrossAttn | 0.7136 | 0.2458 | 0.1772 | 0.7490 | 0.2889 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7582 | 0.3696 | 0.2300 | 0.6701 | 0.3846 |
| F | 158 | 0.6748 | 0.1666 | 0.1456 | 0.8038 | 0.2051 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6799 | 0.2956 | 0.2682 | 0.6082 | 0.3214 |
| F | 158 | 0.6902 | 0.2088 | 0.1213 | 0.8354 | 0.2353 |

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
| **True: Normal** | 178 | 50 |
| **True: Sarco**  | 14 | 13 |

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
