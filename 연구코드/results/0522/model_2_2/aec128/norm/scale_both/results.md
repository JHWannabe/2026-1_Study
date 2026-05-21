# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 20:44  |  5-Fold CV  |  Median best epoch: 33

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
| 1 | 0.9009 | 0.5002 | 0.1973 | 0.6989 | 0.4043 |
| 2 | 0.7849 | 0.3309 | 0.1841 | 0.7527 | 0.4103 |
| 3 | 0.9232 | 0.6275 | 0.1351 | 0.7849 | 0.4737 |
| 4 | 0.8792 | 0.3836 | 0.1130 | 0.8548 | 0.5263 |
| 5 | 0.8412 | 0.4614 | 0.1464 | 0.7784 | 0.4225 |
| **Mean** | **0.8659** | **0.4607** | **0.1552** | **0.7740** | **0.4474** |
| **±Std** | 0.0487 | 0.1022 | 0.0312 | 0.0505 | 0.0464 |

CrossAttn best val AUC per fold: Fold1=0.9009, Fold2=0.7849, Fold3=0.9232, Fold4=0.8792, Fold5=0.8412

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.6635 | 0.2158 | 0.1674 | 0.7811 | 0.3014 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 81 | 0.6527 | 0.3050 | 0.2383 | 0.6543 | 0.3333 |
| F | 152 | 0.6464 | 0.1683 | 0.1296 | 0.8487 | 0.2581 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 171 | 37 |
| **True: Sarco**  | 14 | 11 |

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
