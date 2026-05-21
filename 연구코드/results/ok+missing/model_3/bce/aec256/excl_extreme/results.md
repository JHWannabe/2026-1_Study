# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-22 01:21  |  5-Fold CV  |  Median best epoch: 4

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 289 | 246 | 85.1% | 43 | 14.9% |
| Train | F | 546 | 504 | 92.3% | 42 | 7.7% |
| Train | **All** | **835** | **750** | **89.8%** | **85** | **10.2%** |
| Test | M | 71 | 60 | 84.5% | 11 | 15.5% |
| Test | F | 138 | 127 | 92.0% | 11 | 8.0% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 289 | 59.77 ± 12.01 | 20.00 | 60.00 | 89.00 |
| Train | F | 546 | 54.95 ± 11.82 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **835** | **56.62 ± 12.10** | **14.00** | **57.00** | **91.00** |
| Test | M | 71 | 59.92 ± 12.32 | 29.00 | 61.00 | 84.00 |
| Test | F | 138 | 55.92 ± 11.36 | 23.00 | 55.50 | 83.00 |
| Test | **All** | **209** | **57.28 ± 11.85** | **23.00** | **58.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 289 | 24.10 ± 2.75 | 17.33 | 24.12 | 32.33 |
| Train | F | 546 | 23.05 ± 2.97 | 16.00 | 22.95 | 32.24 |
| Train | **All** | **835** | **23.41 ± 2.94** | **16.00** | **23.33** | **32.33** |
| Test | M | 71 | 24.15 ± 3.49 | 14.34 | 23.88 | 32.56 |
| Test | F | 138 | 22.96 ± 3.29 | 12.02 | 22.64 | 30.84 |
| Test | **All** | **209** | **23.36 ± 3.41** | **12.02** | **23.17** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8494 | 0.4081 | 0.1787 | 0.7425 | 0.4267 |
| 2 | 0.8773 | 0.3677 | 0.1475 | 0.7725 | 0.4722 |
| 3 | 0.8071 | 0.3730 | 0.1759 | 0.7186 | 0.3733 |
| 4 | 0.6831 | 0.2414 | 0.2410 | 0.8204 | 0.3182 |
| 5 | 0.8510 | 0.4101 | 0.1697 | 0.7365 | 0.4054 |
| **Mean** | **0.8136** | **0.3601** | **0.1826** | **0.7581** | **0.3992** |
| **±Std** | 0.0690 | 0.0619 | 0.0312 | 0.0356 | 0.0517 |

CrossAttn best val AUC per fold: Fold1=0.8494, Fold2=0.8773, Fold3=0.8071, Fold4=0.6831, Fold5=0.8510

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8230 | 0.4074 | 0.1681 | 0.6842 | 0.3400 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 71 | 0.8348 | 0.5176 | 0.1774 | 0.7183 | 0.4737 |
| F | 138 | 0.8125 | 0.3844 | 0.1633 | 0.6667 | 0.2581 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 126 | 61 |
| **True: Sarco**  | 5 | 17 |

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
