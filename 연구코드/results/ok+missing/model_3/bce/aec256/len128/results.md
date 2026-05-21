# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-22 01:27  |  5-Fold CV  |  Median best epoch: 15

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 334 | 282 | 84.4% | 52 | 15.6% |
| Train | F | 595 | 548 | 92.1% | 47 | 7.9% |
| Train | **All** | **929** | **830** | **89.3%** | **99** | **10.7%** |
| Test | M | 85 | 71 | 83.5% | 14 | 16.5% |
| Test | F | 148 | 137 | 92.6% | 11 | 7.4% |
| Test | **All** | **233** | **208** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 59.72 ± 12.11 | 20.00 | 60.00 | 89.00 |
| Train | F | 595 | 55.26 ± 11.90 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **929** | **56.86 ± 12.16** | **14.00** | **57.00** | **91.00** |
| Test | M | 85 | 59.26 ± 12.48 | 29.00 | 60.00 | 84.00 |
| Test | F | 148 | 55.47 ± 11.64 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **233** | **56.85 ± 12.09** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 24.38 ± 2.93 | 17.33 | 24.24 | 32.67 |
| Train | F | 595 | 23.15 ± 3.17 | 16.00 | 22.96 | 34.61 |
| Train | **All** | **929** | **23.59 ± 3.14** | **16.00** | **23.44** | **34.61** |
| Test | M | 85 | 24.33 ± 3.39 | 14.34 | 24.24 | 32.56 |
| Test | F | 148 | 23.06 ± 3.53 | 12.02 | 22.64 | 32.48 |
| Test | **All** | **233** | **23.52 ± 3.53** | **12.02** | **23.41** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8542 | 0.5526 | 0.1990 | 0.7903 | 0.4507 |
| 2 | 0.8367 | 0.3791 | 0.1773 | 0.7688 | 0.4691 |
| 3 | 0.7970 | 0.3533 | 0.1881 | 0.6667 | 0.3673 |
| 4 | 0.7633 | 0.2738 | 0.2045 | 0.7204 | 0.3810 |
| 5 | 0.7777 | 0.3062 | 0.1900 | 0.7189 | 0.3659 |
| **Mean** | **0.8058** | **0.3730** | **0.1918** | **0.7330** | **0.4068** |
| **±Std** | 0.0346 | 0.0969 | 0.0094 | 0.0432 | 0.0441 |

CrossAttn best val AUC per fold: Fold1=0.8542, Fold2=0.8367, Fold3=0.7970, Fold4=0.7633, Fold5=0.7777

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8077 | 0.4203 | 0.1613 | 0.7639 | 0.4086 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 85 | 0.8018 | 0.5033 | 0.2222 | 0.6824 | 0.4706 |
| F | 148 | 0.7830 | 0.3816 | 0.1264 | 0.8108 | 0.3333 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 159 | 49 |
| **True: Sarco**  | 6 | 19 |

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
