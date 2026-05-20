# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:19  |  5-Fold CV  |  Median best epoch: 7

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 332 | 276 | 83.1% | 56 | 16.9% |
| Train | F | 597 | 554 | 92.8% | 43 | 7.2% |
| Train | **All** | **929** | **830** | **89.3%** | **99** | **10.7%** |
| Test | M | 87 | 77 | 88.5% | 10 | 11.5% |
| Test | F | 146 | 131 | 89.7% | 15 | 10.3% |
| Test | **All** | **233** | **208** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 332 | 60.33 ± 11.99 | 20.00 | 60.00 | 89.00 |
| Train | F | 597 | 55.52 ± 11.67 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **929** | **57.24 ± 12.01** | **14.00** | **57.00** | **91.00** |
| Test | M | 87 | 56.93 ± 12.53 | 22.00 | 58.00 | 84.00 |
| Test | F | 146 | 54.41 ± 12.53 | 23.00 | 54.00 | 87.00 |
| Test | **All** | **233** | **55.35 ± 12.59** | **22.00** | **57.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 332 | 24.41 ± 3.03 | 16.39 | 24.39 | 32.67 |
| Train | F | 597 | 23.19 ± 3.21 | 12.02 | 22.96 | 34.61 |
| Train | **All** | **929** | **23.63 ± 3.20** | **12.02** | **23.56** | **34.61** |
| Test | M | 87 | 24.22 ± 3.01 | 14.34 | 23.94 | 32.56 |
| Test | F | 146 | 22.87 ± 3.38 | 15.84 | 22.44 | 34.20 |
| Test | **All** | **233** | **23.38 ± 3.31** | **14.34** | **23.31** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8741 | 0.4062 | 0.1940 | 0.6989 | 0.3913 |
| 2 | 0.8708 | 0.4405 | 0.1267 | 0.8118 | 0.4928 |
| 3 | 0.8428 | 0.3692 | 0.1647 | 0.7688 | 0.4110 |
| 4 | 0.8340 | 0.4576 | 0.2024 | 0.6989 | 0.3913 |
| 5 | 0.8193 | 0.3842 | 0.1412 | 0.8000 | 0.4308 |
| **Mean** | **0.8482** | **0.4115** | **0.1658** | **0.7557** | **0.4234** |
| **±Std** | 0.0212 | 0.0332 | 0.0292 | 0.0484 | 0.0376 |

CrossAttn best val AUC per fold: Fold1=0.8741, Fold2=0.8708, Fold3=0.8428, Fold4=0.8340, Fold5=0.8193

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.6954 | 0.2613 | 0.1687 | 0.7425 | 0.3182 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 87 | 0.5909 | 0.2493 | 0.2162 | 0.6782 | 0.2632 |
| F | 146 | 0.7715 | 0.3607 | 0.1404 | 0.7808 | 0.3600 |

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
