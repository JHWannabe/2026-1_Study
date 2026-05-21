# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 15:58  |  5-Fold CV  |  Median best epoch: 4

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 281 | 238 | 84.7% | 43 | 15.3% |
| Train | F | 379 | 343 | 90.5% | 36 | 9.5% |
| Train | **All** | **660** | **581** | **88.0%** | **79** | **12.0%** |
| Test | M | 71 | 60 | 84.5% | 11 | 15.5% |
| Test | F | 95 | 85 | 89.5% | 10 | 10.5% |
| Test | **All** | **166** | **145** | **87.3%** | **21** | **12.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 281 | 59.44 ± 11.39 | 28.00 | 59.00 | 85.00 |
| Train | F | 379 | 56.28 ± 12.55 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **660** | **57.62 ± 12.17** | **14.00** | **58.00** | **91.00** |
| Test | M | 71 | 59.15 ± 13.44 | 23.00 | 60.00 | 89.00 |
| Test | F | 95 | 57.14 ± 11.39 | 29.00 | 57.00 | 87.00 |
| Test | **All** | **166** | **58.00 ± 12.35** | **23.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 281 | 24.32 ± 2.93 | 14.34 | 24.28 | 32.59 |
| Train | F | 379 | 22.96 ± 3.27 | 12.02 | 22.77 | 34.61 |
| Train | **All** | **660** | **23.54 ± 3.20** | **12.02** | **23.44** | **34.61** |
| Test | M | 71 | 24.66 ± 3.33 | 17.51 | 24.42 | 32.67 |
| Test | F | 95 | 22.96 ± 3.57 | 16.00 | 22.64 | 34.20 |
| Test | **All** | **166** | **23.69 ± 3.57** | **16.00** | **23.42** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8000 | 0.5457 | 0.2462 | 0.5985 | 0.3117 |
| 2 | 0.8319 | 0.3644 | 0.1671 | 0.7652 | 0.3922 |
| 3 | 0.7958 | 0.3607 | 0.2292 | 0.6439 | 0.3733 |
| 4 | 0.8389 | 0.4189 | 0.1400 | 0.7727 | 0.4828 |
| 5 | 0.8551 | 0.4067 | 0.1222 | 0.8030 | 0.4583 |
| **Mean** | **0.8243** | **0.4193** | **0.1809** | **0.7167** | **0.4037** |
| **±Std** | 0.0229 | 0.0672 | 0.0488 | 0.0803 | 0.0612 |

CrossAttn best val AUC per fold: Fold1=0.8000, Fold2=0.8319, Fold3=0.7958, Fold4=0.8389, Fold5=0.8551

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7862 | 0.3697 | 0.2391 | 0.6205 | 0.3226 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 71 | 0.7697 | 0.3920 | 0.2527 | 0.5915 | 0.3556 |
| F | 95 | 0.8024 | 0.3827 | 0.2290 | 0.6421 | 0.2917 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 88 | 57 |
| **True: Sarco**  | 6 | 15 |

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
