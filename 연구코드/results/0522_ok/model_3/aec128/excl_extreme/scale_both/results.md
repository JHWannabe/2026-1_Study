# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 20:40  |  5-Fold CV  |  Median best epoch: 13

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
| Train | M | 247 | 59.92 ± 12.19 | 23.00 | 60.00 | 89.00 |
| Train | F | 347 | 56.57 ± 12.33 | 14.00 | 58.00 | 91.00 |
| Train | **All** | **594** | **57.96 ± 12.38** | **14.00** | **59.00** | **91.00** |
| Test | M | 56 | 58.50 ± 10.67 | 32.00 | 58.00 | 82.00 |
| Test | F | 92 | 55.45 ± 11.71 | 29.00 | 54.00 | 84.00 |
| Test | **All** | **148** | **56.60 ± 11.42** | **29.00** | **56.50** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 247 | 24.21 ± 2.89 | 16.39 | 24.07 | 32.33 |
| Train | F | 347 | 22.71 ± 2.95 | 12.02 | 22.64 | 31.50 |
| Train | **All** | **594** | **23.33 ± 3.02** | **12.02** | **23.24** | **32.33** |
| Test | M | 56 | 24.28 ± 3.04 | 17.51 | 24.45 | 32.56 |
| Test | F | 92 | 23.11 ± 3.35 | 16.23 | 23.06 | 31.14 |
| Test | **All** | **148** | **23.55 ± 3.28** | **16.23** | **23.49** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8327 | 0.3751 | 0.1996 | 0.7311 | 0.4286 |
| 2 | 0.8367 | 0.5381 | 0.0974 | 0.8655 | 0.5294 |
| 3 | 0.8156 | 0.4169 | 0.1708 | 0.7227 | 0.4000 |
| 4 | 0.7635 | 0.2814 | 0.1911 | 0.7059 | 0.3396 |
| 5 | 0.8201 | 0.5152 | 0.1500 | 0.7542 | 0.4528 |
| **Mean** | **0.8137** | **0.4254** | **0.1618** | **0.7559** | **0.4301** |
| **±Std** | 0.0263 | 0.0939 | 0.0365 | 0.0570 | 0.0624 |

CrossAttn best val AUC per fold: Fold1=0.8327, Fold2=0.8367, Fold3=0.8156, Fold4=0.7635, Fold5=0.8201

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7656 | 0.3427 | 0.1675 | 0.7635 | 0.3860 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 56 | 0.8333 | 0.4940 | 0.1606 | 0.7857 | 0.4545 |
| F | 92 | 0.7247 | 0.2054 | 0.1717 | 0.7500 | 0.3429 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 102 | 30 |
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
