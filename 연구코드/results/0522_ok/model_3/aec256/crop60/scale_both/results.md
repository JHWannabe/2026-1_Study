# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 18:35  |  5-Fold CV  |  Median best epoch: 63

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 284 | 240 | 84.5% | 44 | 15.5% |
| Train | F | 376 | 340 | 90.4% | 36 | 9.6% |
| Train | **All** | **660** | **580** | **87.9%** | **80** | **12.1%** |
| Test | M | 68 | 58 | 85.3% | 10 | 14.7% |
| Test | F | 98 | 88 | 89.8% | 10 | 10.2% |
| Test | **All** | **166** | **146** | **88.0%** | **20** | **12.0%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 284 | 59.65 ± 12.06 | 23.00 | 60.00 | 89.00 |
| Train | F | 376 | 56.61 ± 12.46 | 14.00 | 58.00 | 91.00 |
| Train | **All** | **660** | **57.92 ± 12.38** | **14.00** | **59.00** | **91.00** |
| Test | M | 68 | 58.24 ± 10.73 | 32.00 | 58.00 | 82.00 |
| Test | F | 98 | 55.86 ± 11.81 | 29.00 | 54.50 | 84.00 |
| Test | **All** | **166** | **56.83 ± 11.44** | **29.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 284 | 24.38 ± 3.03 | 14.34 | 24.18 | 32.67 |
| Train | F | 376 | 22.90 ± 3.23 | 12.02 | 22.69 | 34.20 |
| Train | **All** | **660** | **23.53 ± 3.23** | **12.02** | **23.37** | **34.20** |
| Test | M | 68 | 24.45 ± 2.99 | 17.51 | 24.56 | 32.56 |
| Test | F | 98 | 23.18 ± 3.68 | 16.00 | 23.06 | 34.61 |
| Test | **All** | **166** | **23.70 ± 3.47** | **16.00** | **23.59** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8055 | 0.3270 | 0.2131 | 0.6894 | 0.4225 |
| 2 | 0.8470 | 0.4908 | 0.1201 | 0.8258 | 0.4889 |
| 3 | 0.7823 | 0.4133 | 0.1689 | 0.7803 | 0.4082 |
| 4 | 0.8163 | 0.5365 | 0.1360 | 0.7955 | 0.4000 |
| 5 | 0.8982 | 0.5879 | 0.1123 | 0.8182 | 0.5200 |
| **Mean** | **0.8298** | **0.4711** | **0.1501** | **0.7818** | **0.4479** |
| **±Std** | 0.0400 | 0.0921 | 0.0370 | 0.0490 | 0.0477 |

CrossAttn best val AUC per fold: Fold1=0.8055, Fold2=0.8470, Fold3=0.7823, Fold4=0.8163, Fold5=0.8982

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7342 | 0.2731 | 0.2010 | 0.6988 | 0.3421 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 68 | 0.6793 | 0.2822 | 0.2358 | 0.6765 | 0.3529 |
| F | 98 | 0.7648 | 0.3186 | 0.1769 | 0.7143 | 0.3333 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 103 | 43 |
| **True: Sarco**  | 7 | 13 |

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
