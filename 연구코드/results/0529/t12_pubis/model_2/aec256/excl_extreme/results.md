# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-27 16:54  |  5-Fold CV  |  Median best epoch: 6

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 246 | 212 | 86.2% | 34 | 13.8% |
| Train | F | 371 | 336 | 90.6% | 35 | 9.4% |
| Train | **All** | **617** | **548** | **88.8%** | **69** | **11.2%** |
| Test | M | 64 | 53 | 82.8% | 11 | 17.2% |
| Test | F | 90 | 80 | 88.9% | 10 | 11.1% |
| Test | **All** | **154** | **133** | **86.4%** | **21** | **13.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 246 | 59.52 ± 11.62 | 23.00 | 59.00 | 85.00 |
| Train | F | 371 | 56.44 ± 11.98 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.67 ± 11.93** | **14.00** | **58.00** | **91.00** |
| Test | M | 64 | 60.75 ± 12.72 | 28.00 | 62.00 | 89.00 |
| Test | F | 90 | 54.97 ± 12.65 | 24.00 | 55.00 | 86.00 |
| Test | **All** | **154** | **57.37 ± 12.99** | **24.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 246 | 24.23 ± 2.78 | 16.39 | 24.20 | 32.56 |
| Train | F | 371 | 22.88 ± 3.11 | 12.02 | 22.78 | 31.50 |
| Train | **All** | **617** | **23.42 ± 3.05** | **12.02** | **23.31** | **32.56** |
| Test | M | 64 | 24.23 ± 3.36 | 17.33 | 24.12 | 32.33 |
| Test | F | 90 | 22.69 ± 2.83 | 16.51 | 22.61 | 30.63 |
| Test | **All** | **154** | **23.33 ± 3.15** | **16.51** | **23.25** | **32.33** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8253 | 0.6014 | 0.1977 | 0.9113 | 0.6667 |
| 2 | 0.8006 | 0.3517 | 0.1609 | 0.7742 | 0.4400 |
| 3 | 0.8617 | 0.3431 | 0.1190 | 0.7642 | 0.4727 |
| 4 | 0.7602 | 0.3037 | 0.2357 | 0.5691 | 0.3291 |
| 5 | 0.8014 | 0.2841 | 0.2923 | 0.7724 | 0.4400 |
| **Mean** | **0.8099** | **0.3768** | **0.2011** | **0.7582** | **0.4697** |
| **±Std** | 0.0333 | 0.1150 | 0.0598 | 0.1093 | 0.1098 |

CrossAttn best val AUC per fold: Fold1=0.8253, Fold2=0.8006, Fold3=0.8617, Fold4=0.7602, Fold5=0.8014

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8790 | 0.5129 | 0.2094 | 0.6753 | 0.4444 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 64 | 0.8593 | 0.5928 | 0.2269 | 0.6719 | 0.5116 |
| F | 90 | 0.8912 | 0.5109 | 0.1971 | 0.6778 | 0.3830 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 84 | 49 |
| **True: Sarco**  | 1 | 20 |

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
