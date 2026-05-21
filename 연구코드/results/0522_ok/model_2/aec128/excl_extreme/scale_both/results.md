# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 20:40  |  5-Fold CV  |  Median best epoch: 6

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 241 | 203 | 84.2% | 38 | 15.8% |
| Train | F | 376 | 343 | 91.2% | 33 | 8.8% |
| Train | **All** | **617** | **546** | **88.5%** | **71** | **11.5%** |
| Test | M | 67 | 60 | 89.6% | 7 | 10.4% |
| Test | F | 87 | 75 | 86.2% | 12 | 13.8% |
| Test | **All** | **154** | **135** | **87.7%** | **19** | **12.3%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 241 | 59.32 ± 11.82 | 23.00 | 59.00 | 89.00 |
| Train | F | 376 | 55.72 ± 11.61 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **617** | **57.13 ± 11.83** | **14.00** | **58.00** | **91.00** |
| Test | M | 67 | 60.81 ± 12.01 | 29.00 | 61.00 | 83.00 |
| Test | F | 87 | 57.93 ± 14.05 | 18.00 | 59.00 | 87.00 |
| Test | **All** | **154** | **59.18 ± 13.28** | **18.00** | **59.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 241 | 24.11 ± 2.89 | 16.39 | 24.11 | 32.33 |
| Train | F | 376 | 22.72 ± 3.03 | 12.02 | 22.66 | 31.50 |
| Train | **All** | **617** | **23.26 ± 3.05** | **12.02** | **23.24** | **32.33** |
| Test | M | 67 | 24.48 ± 2.86 | 17.51 | 24.39 | 32.56 |
| Test | F | 87 | 23.36 ± 3.16 | 16.92 | 23.21 | 31.14 |
| Test | **All** | **154** | **23.85 ± 3.08** | **16.92** | **23.45** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8162 | 0.5751 | 0.1820 | 0.7419 | 0.3600 |
| 2 | 0.7963 | 0.3050 | 0.3707 | 0.4839 | 0.3191 |
| 3 | 0.7746 | 0.4185 | 0.1645 | 0.7561 | 0.3750 |
| 4 | 0.8519 | 0.5479 | 0.1749 | 0.7073 | 0.4000 |
| 5 | 0.8578 | 0.4449 | 0.1519 | 0.7886 | 0.4583 |
| **Mean** | **0.8194** | **0.4583** | **0.2088** | **0.6956** | **0.3825** |
| **±Std** | 0.0319 | 0.0969 | 0.0816 | 0.1090 | 0.0461 |

CrossAttn best val AUC per fold: Fold1=0.8162, Fold2=0.7963, Fold3=0.7746, Fold4=0.8519, Fold5=0.8578

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8226 | 0.3189 | 0.1954 | 0.7273 | 0.4474 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 67 | 0.8143 | 0.2540 | 0.2259 | 0.6866 | 0.4000 |
| F | 87 | 0.8367 | 0.4986 | 0.1719 | 0.7586 | 0.4878 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 95 | 40 |
| **True: Sarco**  | 2 | 17 |

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
