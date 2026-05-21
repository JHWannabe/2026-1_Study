# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 20:40  |  5-Fold CV  |  Median best epoch: 24

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 253 | 212 | 83.8% | 41 | 16.2% |
| Train | F | 364 | 332 | 91.2% | 32 | 8.8% |
| Train | **All** | **617** | **544** | **88.2%** | **73** | **11.8%** |
| Test | M | 63 | 57 | 90.5% | 6 | 9.5% |
| Test | F | 91 | 78 | 85.7% | 13 | 14.3% |
| Test | **All** | **154** | **135** | **87.7%** | **19** | **12.3%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 253 | 59.26 ± 11.71 | 23.00 | 59.00 | 89.00 |
| Train | F | 364 | 56.02 ± 11.83 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.35 ± 11.89** | **14.00** | **58.00** | **91.00** |
| Test | M | 63 | 60.44 ± 12.65 | 29.00 | 61.00 | 83.00 |
| Test | F | 91 | 57.99 ± 14.06 | 18.00 | 59.00 | 87.00 |
| Test | **All** | **154** | **58.99 ± 13.56** | **18.00** | **59.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 253 | 24.44 ± 3.07 | 14.34 | 24.28 | 32.67 |
| Train | F | 364 | 22.84 ± 3.34 | 12.02 | 22.67 | 34.20 |
| Train | **All** | **617** | **23.49 ± 3.33** | **12.02** | **23.36** | **34.20** |
| Test | M | 63 | 24.71 ± 3.00 | 17.51 | 24.51 | 32.56 |
| Test | F | 91 | 23.49 ± 3.30 | 16.92 | 23.12 | 34.61 |
| Test | **All** | **154** | **23.98 ± 3.23** | **16.92** | **23.62** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8428 | 0.4067 | 0.1702 | 0.7661 | 0.4912 |
| 2 | 0.8801 | 0.4567 | 0.1099 | 0.8548 | 0.5000 |
| 3 | 0.7759 | 0.3115 | 0.2020 | 0.6911 | 0.3448 |
| 4 | 0.7818 | 0.4283 | 0.1353 | 0.8374 | 0.4444 |
| 5 | 0.8327 | 0.5746 | 0.1767 | 0.7236 | 0.4138 |
| **Mean** | **0.8227** | **0.4356** | **0.1588** | **0.7746** | **0.4389** |
| **±Std** | 0.0392 | 0.0849 | 0.0324 | 0.0633 | 0.0565 |

CrossAttn best val AUC per fold: Fold1=0.8428, Fold2=0.8801, Fold3=0.7759, Fold4=0.7818, Fold5=0.8327

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7485 | 0.2616 | 0.1825 | 0.7468 | 0.4000 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 63 | 0.7690 | 0.2503 | 0.2294 | 0.6825 | 0.3333 |
| F | 91 | 0.7702 | 0.3406 | 0.1501 | 0.7912 | 0.4571 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 102 | 33 |
| **True: Sarco**  | 6 | 13 |

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
