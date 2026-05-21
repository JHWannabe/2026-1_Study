# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 20:08  |  5-Fold CV  |  Median best epoch: 12

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
| 1 | 0.8217 | 0.4070 | 0.1907 | 0.7197 | 0.4127 |
| 2 | 0.8508 | 0.4846 | 0.1034 | 0.8636 | 0.4706 |
| 3 | 0.7516 | 0.3336 | 0.2216 | 0.6136 | 0.3200 |
| 4 | 0.8082 | 0.4723 | 0.1838 | 0.6970 | 0.3333 |
| 5 | 0.8766 | 0.4911 | 0.0966 | 0.8712 | 0.5405 |
| **Mean** | **0.8218** | **0.4377** | **0.1592** | **0.7530** | **0.4154** |
| **±Std** | 0.0423 | 0.0601 | 0.0500 | 0.0999 | 0.0831 |

CrossAttn best val AUC per fold: Fold1=0.8217, Fold2=0.8508, Fold3=0.7516, Fold4=0.8082, Fold5=0.8766

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7952 | 0.3892 | 0.1686 | 0.7048 | 0.3951 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 68 | 0.7672 | 0.4309 | 0.1789 | 0.6912 | 0.4324 |
| F | 98 | 0.8170 | 0.4399 | 0.1614 | 0.7143 | 0.3636 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 101 | 45 |
| **True: Sarco**  | 4 | 16 |

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
