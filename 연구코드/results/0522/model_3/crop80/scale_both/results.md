# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-19 12:55  |  5-Fold CV  |  Median best epoch: 4

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
| 1 | 0.8997 | 0.4762 | 0.1241 | 0.8000 | 0.4590 |
| 2 | 0.8356 | 0.2992 | 0.1613 | 0.8121 | 0.4151 |
| 3 | 0.8029 | 0.3163 | 0.1619 | 0.7333 | 0.3889 |
| 4 | 0.7047 | 0.1999 | 0.2334 | 0.6485 | 0.3095 |
| 5 | 0.8446 | 0.4006 | 0.1857 | 0.7091 | 0.3846 |
| **Mean** | **0.8175** | **0.3385** | **0.1733** | **0.7406** | **0.3914** |
| **±Std** | 0.0644 | 0.0939 | 0.0359 | 0.0603 | 0.0488 |

CrossAttn best val AUC per fold: Fold1=0.8997, Fold2=0.8356, Fold3=0.8029, Fold4=0.7047, Fold5=0.8446

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7645 | 0.3198 | 0.2162 | 0.6473 | 0.3178 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 65 | 0.7311 | 0.4099 | 0.2825 | 0.5538 | 0.4314 |
| F | 142 | 0.7101 | 0.1872 | 0.1859 | 0.6901 | 0.2143 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 117 | 69 |
| **True: Sarco**  | 4 | 17 |

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
