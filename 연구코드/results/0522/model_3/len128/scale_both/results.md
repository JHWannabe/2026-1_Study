# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-19 12:50  |  5-Fold CV  |  Median best epoch: 14

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
| 1 | 0.9065 | 0.4849 | 0.1581 | 0.7697 | 0.4412 |
| 2 | 0.8372 | 0.3179 | 0.2012 | 0.6909 | 0.3544 |
| 3 | 0.8108 | 0.3143 | 0.1257 | 0.8364 | 0.4490 |
| 4 | 0.7361 | 0.2372 | 0.1994 | 0.6727 | 0.3077 |
| 5 | 0.8327 | 0.3413 | 0.2498 | 0.5697 | 0.3107 |
| **Mean** | **0.8247** | **0.3391** | **0.1868** | **0.7079** | **0.3726** |
| **±Std** | 0.0547 | 0.0809 | 0.0422 | 0.0905 | 0.0615 |

CrossAttn best val AUC per fold: Fold1=0.9065, Fold2=0.8372, Fold3=0.8108, Fold4=0.7361, Fold5=0.8327

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7627 | 0.3332 | 0.1793 | 0.7053 | 0.3441 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 65 | 0.7123 | 0.4040 | 0.2650 | 0.5846 | 0.4490 |
| F | 142 | 0.7135 | 0.2774 | 0.1401 | 0.7606 | 0.2273 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 130 | 56 |
| **True: Sarco**  | 5 | 16 |

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
