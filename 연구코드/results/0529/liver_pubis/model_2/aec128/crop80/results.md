# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-25 18:07  |  5-Fold CV  |  Median best epoch: 11

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 335 | 283 | 84.5% | 52 | 15.5% |
| Train | F | 595 | 548 | 92.1% | 47 | 7.9% |
| Train | **All** | **930** | **831** | **89.4%** | **99** | **10.6%** |
| Test | M | 85 | 71 | 83.5% | 14 | 16.5% |
| Test | F | 148 | 137 | 92.6% | 11 | 7.4% |
| Test | **All** | **233** | **208** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 335 | 59.78 ± 12.13 | 20.00 | 60.00 | 89.00 |
| Train | F | 595 | 55.26 ± 11.90 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **930** | **56.89 ± 12.18** | **14.00** | **57.00** | **91.00** |
| Test | M | 85 | 59.08 ± 12.34 | 29.00 | 60.00 | 84.00 |
| Test | F | 148 | 55.47 ± 11.64 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **233** | **56.79 ± 12.03** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 335 | 24.39 ± 2.93 | 17.33 | 24.24 | 32.67 |
| Train | F | 595 | 23.15 ± 3.17 | 16.00 | 22.96 | 34.61 |
| Train | **All** | **930** | **23.60 ± 3.15** | **16.00** | **23.45** | **34.61** |
| Test | M | 85 | 24.34 ± 3.40 | 14.34 | 24.26 | 32.56 |
| Test | F | 148 | 23.06 ± 3.53 | 12.02 | 22.64 | 32.48 |
| Test | **All** | **233** | **23.52 ± 3.53** | **12.02** | **23.41** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8193 | 0.3407 | 0.1287 | 0.7849 | 0.4595 |
| 2 | 0.8069 | 0.4180 | 0.1973 | 0.7527 | 0.3947 |
| 3 | 0.8114 | 0.4372 | 0.1899 | 0.7957 | 0.4571 |
| 4 | 0.8925 | 0.4501 | 0.2411 | 0.7527 | 0.4524 |
| 5 | 0.7743 | 0.3353 | 0.1613 | 0.7742 | 0.4167 |
| **Mean** | **0.8209** | **0.3962** | **0.1837** | **0.7720** | **0.4361** |
| **±Std** | 0.0389 | 0.0487 | 0.0375 | 0.0172 | 0.0259 |

CrossAttn best val AUC per fold: Fold1=0.8193, Fold2=0.8069, Fold3=0.8114, Fold4=0.8925, Fold5=0.7743

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8085 | 0.3953 | 0.1675 | 0.7854 | 0.3421 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 85 | 0.7827 | 0.4617 | 0.2172 | 0.6941 | 0.3810 |
| F | 148 | 0.7996 | 0.3379 | 0.1391 | 0.8378 | 0.2941 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 170 | 38 |
| **True: Sarco**  | 12 | 13 |

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
