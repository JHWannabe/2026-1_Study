# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-19 12:59  |  5-Fold CV  |  Median best epoch: 17

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 251 | 207 | 82.5% | 44 | 17.5% |
| Train | F | 574 | 535 | 93.2% | 39 | 6.8% |
| Train | **All** | **825** | **742** | **89.9%** | **83** | **10.1%** |
| Test | M | 65 | 53 | 81.5% | 12 | 18.5% |
| Test | F | 142 | 133 | 93.7% | 9 | 6.3% |
| Test | **All** | **207** | **186** | **89.9%** | **21** | **10.1%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 251 | 61.35 ± 11.39 | 28.00 | 62.00 | 89.00 |
| Train | F | 574 | 55.42 ± 11.49 | 24.00 | 55.00 | 87.00 |
| Train | **All** | **825** | **57.22 ± 11.78** | **24.00** | **58.00** | **89.00** |
| Test | M | 65 | 64.37 ± 11.48 | 31.00 | 66.00 | 84.00 |
| Test | F | 142 | 57.01 ± 11.61 | 27.00 | 57.00 | 91.00 |
| Test | **All** | **207** | **59.32 ± 12.06** | **27.00** | **59.00** | **91.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 251 | 24.15 ± 2.76 | 15.22 | 24.34 | 31.94 |
| Train | F | 574 | 23.22 ± 3.14 | 15.63 | 23.03 | 31.93 |
| Train | **All** | **825** | **23.50 ± 3.06** | **15.22** | **23.42** | **31.94** |
| Test | M | 65 | 23.83 ± 2.48 | 19.29 | 23.55 | 29.65 |
| Test | F | 142 | 23.10 ± 2.82 | 16.80 | 22.98 | 30.05 |
| Test | **All** | **207** | **23.33 ± 2.74** | **16.80** | **23.12** | **30.05** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.9027 | 0.5360 | 0.1693 | 0.7212 | 0.3784 |
| 2 | 0.8423 | 0.3030 | 0.1693 | 0.7455 | 0.3824 |
| 3 | 0.8164 | 0.3540 | 0.1224 | 0.8424 | 0.4583 |
| 4 | 0.7564 | 0.3036 | 0.1908 | 0.6848 | 0.2973 |
| 5 | 0.8136 | 0.3395 | 0.2236 | 0.6061 | 0.3158 |
| **Mean** | **0.8263** | **0.3672** | **0.1751** | **0.7200** | **0.3664** |
| **±Std** | 0.0474 | 0.0867 | 0.0330 | 0.0773 | 0.0569 |

CrossAttn best val AUC per fold: Fold1=0.9027, Fold2=0.8423, Fold3=0.8164, Fold4=0.7564, Fold5=0.8136

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7307 | 0.2853 | 0.2389 | 0.5894 | 0.2609 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 65 | 0.7044 | 0.3565 | 0.3366 | 0.4769 | 0.3704 |
| F | 142 | 0.6617 | 0.2143 | 0.1942 | 0.6408 | 0.1639 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 107 | 79 |
| **True: Sarco**  | 6 | 15 |

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
