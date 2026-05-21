# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 20:46  |  5-Fold CV  |  Median best epoch: 8

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 338 | 285 | 84.3% | 53 | 15.7% |
| Train | F | 591 | 545 | 92.2% | 46 | 7.8% |
| Train | **All** | **929** | **830** | **89.3%** | **99** | **10.7%** |
| Test | M | 81 | 68 | 84.0% | 13 | 16.0% |
| Test | F | 152 | 140 | 92.1% | 12 | 7.9% |
| Test | **All** | **233** | **208** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 338 | 60.04 ± 11.99 | 20.00 | 60.00 | 89.00 |
| Train | F | 591 | 55.33 ± 11.80 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **929** | **57.05 ± 12.08** | **14.00** | **57.00** | **91.00** |
| Test | M | 81 | 57.89 ± 12.84 | 22.00 | 58.00 | 80.00 |
| Test | F | 152 | 55.18 ± 12.05 | 18.00 | 55.00 | 86.00 |
| Test | **All** | **233** | **56.12 ± 12.40** | **18.00** | **56.00** | **86.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 338 | 24.29 ± 3.03 | 14.34 | 24.20 | 32.59 |
| Train | F | 591 | 23.04 ± 3.26 | 12.02 | 22.91 | 34.20 |
| Train | **All** | **929** | **23.49 ± 3.24** | **12.02** | **23.41** | **34.20** |
| Test | M | 81 | 24.68 ± 2.98 | 18.44 | 24.26 | 32.67 |
| Test | F | 152 | 23.49 ± 3.16 | 16.44 | 22.99 | 34.61 |
| Test | **All** | **233** | **23.90 ± 3.15** | **16.44** | **23.67** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8608 | 0.3647 | 0.1606 | 0.7634 | 0.4359 |
| 2 | 0.8175 | 0.4142 | 0.2163 | 0.6505 | 0.3299 |
| 3 | 0.8925 | 0.4786 | 0.1723 | 0.7258 | 0.4396 |
| 4 | 0.8747 | 0.3600 | 0.1236 | 0.8226 | 0.4762 |
| 5 | 0.8497 | 0.4469 | 0.1309 | 0.8054 | 0.4375 |
| **Mean** | **0.8590** | **0.4129** | **0.1607** | **0.7536** | **0.4238** |
| **±Std** | 0.0252 | 0.0460 | 0.0332 | 0.0615 | 0.0493 |

CrossAttn best val AUC per fold: Fold1=0.8608, Fold2=0.8175, Fold3=0.8925, Fold4=0.8747, Fold5=0.8497

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.6971 | 0.2715 | 0.1784 | 0.7425 | 0.3182 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 81 | 0.6674 | 0.2894 | 0.2448 | 0.6543 | 0.3913 |
| F | 152 | 0.6851 | 0.2955 | 0.1430 | 0.7895 | 0.2381 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 159 | 49 |
| **True: Sarco**  | 11 | 14 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (CrossAttn) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curve |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
