# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:18  |  5-Fold CV  |  Median best epoch: 4

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 334 | 280 | 83.8% | 54 | 16.2% |
| Train | F | 598 | 553 | 92.5% | 45 | 7.5% |
| Train | **All** | **932** | **833** | **89.4%** | **99** | **10.6%** |
| Test | M | 86 | 74 | 86.0% | 12 | 14.0% |
| Test | F | 148 | 135 | 91.2% | 13 | 8.8% |
| Test | **All** | **234** | **209** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 59.81 ± 12.21 | 20.00 | 60.00 | 89.00 |
| Train | F | 598 | 55.43 ± 11.87 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **932** | **57.00 ± 12.17** | **14.00** | **57.00** | **91.00** |
| Test | M | 86 | 58.88 ± 11.98 | 28.00 | 60.50 | 84.00 |
| Test | F | 148 | 54.66 ± 11.70 | 23.00 | 54.00 | 87.00 |
| Test | **All** | **234** | **56.21 ± 11.98** | **23.00** | **56.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 24.31 ± 3.02 | 14.34 | 24.27 | 32.67 |
| Train | F | 598 | 23.26 ± 3.27 | 12.02 | 23.06 | 34.61 |
| Train | **All** | **932** | **23.64 ± 3.22** | **12.02** | **23.56** | **34.61** |
| Test | M | 86 | 24.67 ± 3.13 | 17.43 | 24.15 | 32.56 |
| Test | F | 148 | 22.65 ± 3.12 | 16.44 | 22.14 | 34.20 |
| Test | **All** | **234** | **23.39 ± 3.27** | **16.44** | **23.30** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8383 | 0.4238 | 0.2070 | 0.6524 | 0.3434 |
| 2 | 0.8138 | 0.4166 | 0.1201 | 0.8449 | 0.4528 |
| 3 | 0.8308 | 0.3533 | 0.1972 | 0.7258 | 0.3855 |
| 4 | 0.7970 | 0.3150 | 0.1524 | 0.7419 | 0.3846 |
| 5 | 0.8386 | 0.3549 | 0.1382 | 0.7849 | 0.4444 |
| **Mean** | **0.8237** | **0.3727** | **0.1630** | **0.7500** | **0.4022** |
| **±Std** | 0.0161 | 0.0414 | 0.0337 | 0.0639 | 0.0410 |

CrossAttn best val AUC per fold: Fold1=0.8383, Fold2=0.8138, Fold3=0.8308, Fold4=0.7970, Fold5=0.8386

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7971 | 0.3305 | 0.2983 | 0.5556 | 0.3067 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 86 | 0.7286 | 0.3249 | 0.3460 | 0.4767 | 0.3077 |
| F | 148 | 0.8387 | 0.4382 | 0.2706 | 0.6014 | 0.3059 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 107 | 102 |
| **True: Sarco**  | 2 | 23 |

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
