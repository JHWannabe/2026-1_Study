# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 12:35  |  5-Fold CV  |  Median best epoch: 3

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 245 | 211 | 86.1% | 34 | 13.9% |
| Train | F | 372 | 337 | 90.6% | 35 | 9.4% |
| Train | **All** | **617** | **548** | **88.8%** | **69** | **11.2%** |
| Test | M | 64 | 53 | 82.8% | 11 | 17.2% |
| Test | F | 90 | 80 | 88.9% | 10 | 11.1% |
| Test | **All** | **154** | **133** | **86.4%** | **21** | **13.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 245 | 59.47 ± 11.67 | 23.00 | 59.00 | 85.00 |
| Train | F | 372 | 56.48 ± 11.97 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.67 ± 11.95** | **14.00** | **58.00** | **91.00** |
| Test | M | 64 | 60.75 ± 12.72 | 28.00 | 62.00 | 89.00 |
| Test | F | 90 | 54.97 ± 12.65 | 24.00 | 55.00 | 86.00 |
| Test | **All** | **154** | **57.37 ± 12.99** | **24.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 245 | 24.20 ± 2.76 | 16.39 | 24.17 | 32.56 |
| Train | F | 372 | 22.89 ± 3.12 | 12.02 | 22.81 | 31.50 |
| Train | **All** | **617** | **23.41 ± 3.05** | **12.02** | **23.31** | **32.56** |
| Test | M | 64 | 24.23 ± 3.36 | 17.33 | 24.12 | 32.33 |
| Test | F | 90 | 22.69 ± 2.83 | 16.51 | 22.61 | 30.63 |
| Test | **All** | **154** | **23.33 ± 3.15** | **16.51** | **23.25** | **32.33** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8058 | 0.5616 | 0.1752 | 0.7339 | 0.4000 |
| 2 | 0.7870 | 0.2688 | 0.1810 | 0.7258 | 0.3704 |
| 3 | 0.8440 | 0.3640 | 0.1893 | 0.6911 | 0.4062 |
| 4 | 0.7765 | 0.3079 | 0.2124 | 0.6748 | 0.3548 |
| 5 | 0.8105 | 0.2788 | 0.1728 | 0.7561 | 0.4231 |
| **Mean** | **0.8048** | **0.3562** | **0.1861** | **0.7163** | **0.3909** |
| **±Std** | 0.0232 | 0.1079 | 0.0143 | 0.0295 | 0.0248 |

CrossAttn best val AUC per fold: Fold1=0.8058, Fold2=0.7870, Fold3=0.8440, Fold4=0.7765, Fold5=0.8105

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8453 | 0.4630 | 0.1545 | 0.7597 | 0.4932 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 64 | 0.8508 | 0.5606 | 0.1512 | 0.7812 | 0.5625 |
| F | 90 | 0.8350 | 0.3716 | 0.1568 | 0.7444 | 0.4390 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 99 | 34 |
| **True: Sarco**  | 3 | 18 |

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
