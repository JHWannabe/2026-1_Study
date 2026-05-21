# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 18:51  |  5-Fold CV  |  Median best epoch: 6

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 247 | 211 | 85.4% | 36 | 14.6% |
| Train | F | 347 | 312 | 89.9% | 35 | 10.1% |
| Train | **All** | **594** | **523** | **88.0%** | **71** | **12.0%** |
| Test | M | 56 | 48 | 85.7% | 8 | 14.3% |
| Test | F | 92 | 84 | 91.3% | 8 | 8.7% |
| Test | **All** | **148** | **132** | **89.2%** | **16** | **10.8%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 247 | 59.91 ± 12.18 | 23.00 | 60.00 | 89.00 |
| Train | F | 347 | 56.58 ± 12.34 | 14.00 | 58.00 | 91.00 |
| Train | **All** | **594** | **57.96 ± 12.39** | **14.00** | **59.00** | **91.00** |
| Test | M | 56 | 58.50 ± 10.67 | 32.00 | 58.00 | 82.00 |
| Test | F | 92 | 55.45 ± 11.71 | 29.00 | 54.00 | 84.00 |
| Test | **All** | **148** | **56.60 ± 11.42** | **29.00** | **56.50** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 247 | 24.21 ± 2.89 | 16.39 | 24.07 | 32.33 |
| Train | F | 347 | 22.69 ± 2.92 | 12.02 | 22.64 | 31.50 |
| Train | **All** | **594** | **23.32 ± 3.00** | **12.02** | **23.24** | **32.33** |
| Test | M | 56 | 24.28 ± 3.04 | 17.51 | 24.45 | 32.56 |
| Test | F | 92 | 23.11 ± 3.35 | 16.23 | 23.06 | 31.14 |
| Test | **All** | **148** | **23.55 ± 3.28** | **16.23** | **23.49** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8694 | 0.4238 | 0.2111 | 0.6723 | 0.4000 |
| 2 | 0.8027 | 0.4695 | 0.1073 | 0.8571 | 0.5143 |
| 3 | 0.8061 | 0.3897 | 0.1397 | 0.8319 | 0.5000 |
| 4 | 0.7635 | 0.3298 | 0.2316 | 0.6555 | 0.3692 |
| 5 | 0.8668 | 0.4911 | 0.1209 | 0.8220 | 0.5116 |
| **Mean** | **0.8217** | **0.4208** | **0.1621** | **0.7678** | **0.4590** |
| **±Std** | 0.0407 | 0.0576 | 0.0499 | 0.0858 | 0.0617 |

CrossAttn best val AUC per fold: Fold1=0.8694, Fold2=0.8027, Fold3=0.8061, Fold4=0.7635, Fold5=0.8668

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7912 | 0.3455 | 0.1810 | 0.7162 | 0.3438 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 56 | 0.8516 | 0.4749 | 0.1432 | 0.7857 | 0.4545 |
| F | 92 | 0.7812 | 0.3136 | 0.2039 | 0.6739 | 0.2857 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 95 | 37 |
| **True: Sarco**  | 5 | 11 |

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
