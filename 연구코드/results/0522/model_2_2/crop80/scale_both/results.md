# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:16  |  5-Fold CV  |  Median best epoch: 6

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
| 1 | 0.7697 | 0.2970 | 0.1654 | 0.7500 | 0.3377 |
| 2 | 0.8302 | 0.4397 | 0.1454 | 0.8128 | 0.4722 |
| 3 | 0.8697 | 0.3708 | 0.1469 | 0.8030 | 0.4872 |
| 4 | 0.8433 | 0.4235 | 0.1753 | 0.7389 | 0.4045 |
| 5 | 0.8209 | 0.3627 | 0.1755 | 0.7537 | 0.4186 |
| **Mean** | **0.8268** | **0.3787** | **0.1617** | **0.7717** | **0.4240** |
| **±Std** | 0.0329 | 0.0504 | 0.0132 | 0.0301 | 0.0533 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7927 | 0.3915 | 0.1646 | 0.7304 | 0.3529 |
| 2 | 0.8521 | 0.4896 | 0.1523 | 0.7586 | 0.4235 |
| 3 | 0.8071 | 0.3375 | 0.1652 | 0.7734 | 0.3611 |
| 4 | 0.8538 | 0.5898 | 0.1477 | 0.7635 | 0.4146 |
| 5 | 0.8435 | 0.3910 | 0.1931 | 0.6995 | 0.3838 |
| **Mean** | **0.8299** | **0.4399** | **0.1646** | **0.7451** | **0.3872** |
| **±Std** | 0.0251 | 0.0896 | 0.0158 | 0.0269 | 0.0281 |

CrossAttn best val AUC per fold: Fold1=0.7927, Fold2=0.8521, Fold3=0.8071, Fold4=0.8538, Fold5=0.8435

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6948 | 0.2111 | 0.1790 | 0.7608 | 0.2989 |
| CrossAttn | 0.7498 | 0.2606 | 0.1855 | 0.7216 | 0.3238 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7238 | 0.2776 | 0.2398 | 0.6804 | 0.4151 |
| F | 158 | 0.6387 | 0.1257 | 0.1416 | 0.8101 | 0.1176 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7711 | 0.3240 | 0.2364 | 0.6495 | 0.3929 |
| F | 158 | 0.7082 | 0.1818 | 0.1543 | 0.7658 | 0.2449 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 181 | 47 |
| **True: Sarco**  | 14 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 61 |
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
