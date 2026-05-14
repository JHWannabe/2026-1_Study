# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 20:05  |  5-Fold CV  |  Median best epoch: 11

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 309 | 261 | 84.5% | 48 | 15.5% |
| Train | F | 605 | 559 | 92.4% | 46 | 7.6% |
| Train | **All** | **914** | **820** | **89.7%** | **94** | **10.3%** |
| Test | M | 83 | 70 | 84.3% | 13 | 15.7% |
| Test | F | 146 | 133 | 91.1% | 13 | 8.9% |
| Test | **All** | **229** | **203** | **88.6%** | **26** | **11.4%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 309 | 59.60 ± 12.50 | 18.00 | 60.00 | 89.00 |
| Train | F | 605 | 55.60 ± 11.93 | 18.00 | 55.00 | 91.00 |
| Train | **All** | **914** | **56.95 ± 12.27** | **18.00** | **57.00** | **91.00** |
| Test | M | 83 | 59.20 ± 12.71 | 28.00 | 60.00 | 88.00 |
| Test | F | 146 | 54.76 ± 11.42 | 23.00 | 55.00 | 86.00 |
| Test | **All** | **229** | **56.37 ± 12.09** | **23.00** | **57.00** | **88.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 309 | 24.01 ± 3.23 | 14.48 | 24.07 | 35.20 |
| Train | F | 605 | 23.00 ± 3.25 | 14.40 | 22.72 | 36.24 |
| Train | **All** | **914** | **23.34 ± 3.27** | **14.40** | **23.24** | **36.24** |
| Test | M | 83 | 24.36 ± 2.96 | 18.37 | 24.39 | 33.87 |
| Test | F | 146 | 22.97 ± 3.08 | 16.87 | 22.65 | 34.23 |
| Test | **All** | **229** | **23.47 ± 3.11** | **16.87** | **23.28** | **34.23** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.5350 | 0.2018 | 0.2124 | 0.7158 | 0.2121 |
| 2 | 0.6277 | 0.2538 | 0.1763 | 0.7541 | 0.2105 |
| 3 | 0.5331 | 0.1521 | 0.2158 | 0.6940 | 0.1515 |
| 4 | 0.5719 | 0.1514 | 0.2364 | 0.6776 | 0.1690 |
| 5 | 0.5010 | 0.1053 | 0.2013 | 0.6978 | 0.1270 |
| **Mean** | **0.5537** | **0.1729** | **0.2084** | **0.7079** | **0.1740** |
| **±Std** | 0.0433 | 0.0507 | 0.0197 | 0.0261 | 0.0333 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8007 | 0.3641 | 0.1468 | 0.7923 | 0.4062 |
| 2 | 0.8752 | 0.4948 | 0.1501 | 0.7923 | 0.4412 |
| 3 | 0.9175 | 0.5834 | 0.1288 | 0.8415 | 0.5397 |
| 4 | 0.8302 | 0.4184 | 0.1710 | 0.7650 | 0.3944 |
| 5 | 0.8638 | 0.3636 | 0.1910 | 0.7363 | 0.4000 |
| **Mean** | **0.8575** | **0.4449** | **0.1575** | **0.7855** | **0.4363** |
| **±Std** | 0.0398 | 0.0843 | 0.0214 | 0.0349 | 0.0542 |

CrossAttn best val AUC per fold: Fold1=0.8007, Fold2=0.8752, Fold3=0.9175, Fold4=0.8302, Fold5=0.8638

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5102 | 0.1834 | 0.2323 | 0.6681 | 0.1739 |
| CrossAttn | 0.7219 | 0.2499 | 0.1826 | 0.7205 | 0.2889 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 83 | 0.4429 | 0.1945 | 0.2627 | 0.6265 | 0.1143 |
| F | 146 | 0.5830 | 0.1882 | 0.2151 | 0.6918 | 0.2105 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 83 | 0.7165 | 0.3352 | 0.2188 | 0.6988 | 0.3902 |
| F | 146 | 0.7183 | 0.1874 | 0.1620 | 0.7329 | 0.2041 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 145 | 58 |
| **True: Sarco**  | 18 | 8 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 152 | 51 |
| **True: Sarco**  | 13 | 13 |

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
