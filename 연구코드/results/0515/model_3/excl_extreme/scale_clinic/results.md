# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:49  |  5-Fold CV  |  Median best epoch: 11

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
| 1 | 0.5427 | 0.1922 | 0.1940 | 0.7377 | 0.2000 |
| 2 | 0.6136 | 0.1700 | 0.2279 | 0.6885 | 0.1972 |
| 3 | 0.5574 | 0.1468 | 0.2122 | 0.6995 | 0.1791 |
| 4 | 0.5567 | 0.1409 | 0.2140 | 0.7158 | 0.2121 |
| 5 | 0.5816 | 0.1394 | 0.2440 | 0.6557 | 0.1127 |
| **Mean** | **0.5704** | **0.1579** | **0.2184** | **0.6995** | **0.1802** |
| **±Std** | 0.0250 | 0.0204 | 0.0167 | 0.0274 | 0.0354 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7522 | 0.2377 | 0.2190 | 0.6940 | 0.3488 |
| 2 | 0.7741 | 0.3850 | 0.2340 | 0.6557 | 0.3077 |
| 3 | 0.7628 | 0.3237 | 0.1702 | 0.7650 | 0.3768 |
| 4 | 0.8899 | 0.5735 | 0.1557 | 0.7814 | 0.4737 |
| 5 | 0.8150 | 0.4284 | 0.1315 | 0.8087 | 0.4262 |
| **Mean** | **0.7988** | **0.3897** | **0.1821** | **0.7410** | **0.3867** |
| **±Std** | 0.0503 | 0.1120 | 0.0386 | 0.0571 | 0.0581 |

CrossAttn best val AUC per fold: Fold1=0.7522, Fold2=0.7741, Fold3=0.7628, Fold4=0.8899, Fold5=0.8150

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5799 | 0.1973 | 0.2049 | 0.7118 | 0.2143 |
| CrossAttn | 0.8089 | 0.3427 | 0.1719 | 0.7555 | 0.4043 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 86 | 0.5564 | 0.3014 | 0.2472 | 0.6744 | 0.2632 |
| F | 143 | 0.6033 | 0.1542 | 0.1795 | 0.7343 | 0.1739 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 86 | 0.7597 | 0.3533 | 0.2100 | 0.7209 | 0.4783 |
| F | 143 | 0.8216 | 0.3904 | 0.1490 | 0.7762 | 0.3333 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 154 | 51 |
| **True: Sarco**  | 15 | 9 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 154 | 51 |
| **True: Sarco**  | 5 | 19 |

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
