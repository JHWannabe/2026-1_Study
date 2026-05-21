# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 12:12  |  5-Fold CV  |  Median best epoch: 57

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
| 1 | 0.8245 | 0.5573 | 0.1750 | 0.7273 | 0.3571 |
| 2 | 0.8529 | 0.3541 | 0.1867 | 0.7045 | 0.4348 |
| 3 | 0.7899 | 0.3355 | 0.1518 | 0.7727 | 0.4231 |
| 4 | 0.8675 | 0.4610 | 0.1342 | 0.8409 | 0.3636 |
| 5 | 0.8572 | 0.3977 | 0.1599 | 0.7727 | 0.4643 |
| **Mean** | **0.8384** | **0.4211** | **0.1615** | **0.7636** | **0.4086** |
| **±Std** | 0.0281 | 0.0806 | 0.0182 | 0.0468 | 0.0416 |

CrossAttn best val AUC per fold: Fold1=0.8245, Fold2=0.8529, Fold3=0.7899, Fold4=0.8675, Fold5=0.8572

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.6949 | 0.2852 | 0.1955 | 0.7410 | 0.3385 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 71 | 0.6606 | 0.4032 | 0.2673 | 0.6479 | 0.3243 |
| F | 95 | 0.7082 | 0.2366 | 0.1418 | 0.8105 | 0.3571 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 112 | 33 |
| **True: Sarco**  | 10 | 11 |

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
