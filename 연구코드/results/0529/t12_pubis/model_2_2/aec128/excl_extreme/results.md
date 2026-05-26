# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-22 17:39  |  5-Fold CV  |  Median best epoch: 39

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 256 | 217 | 84.8% | 39 | 15.2% |
| Train | F | 361 | 325 | 90.0% | 36 | 10.0% |
| Train | **All** | **617** | **542** | **87.8%** | **75** | **12.2%** |
| Test | M | 63 | 55 | 87.3% | 8 | 12.7% |
| Test | F | 91 | 82 | 90.1% | 9 | 9.9% |
| Test | **All** | **154** | **137** | **89.0%** | **17** | **11.0%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 256 | 59.57 ± 11.76 | 23.00 | 59.00 | 85.00 |
| Train | F | 361 | 56.66 ± 12.28 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.87 ± 12.15** | **14.00** | **58.00** | **91.00** |
| Test | M | 63 | 59.02 ± 12.66 | 28.00 | 61.00 | 89.00 |
| Test | F | 91 | 55.78 ± 12.86 | 24.00 | 56.00 | 86.00 |
| Test | **All** | **154** | **57.10 ± 12.87** | **24.00** | **57.50** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 256 | 24.53 ± 2.97 | 14.34 | 24.45 | 32.67 |
| Train | F | 361 | 22.94 ± 3.38 | 12.02 | 22.77 | 34.61 |
| Train | **All** | **617** | **23.60 ± 3.31** | **12.02** | **23.51** | **34.61** |
| Test | M | 63 | 24.50 ± 3.21 | 17.33 | 24.12 | 32.33 |
| Test | F | 91 | 22.99 ± 3.31 | 16.00 | 22.76 | 34.20 |
| Test | **All** | **154** | **23.61 ± 3.35** | **16.00** | **23.35** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8135 | 0.5423 | 0.1425 | 0.7258 | 0.4333 |
| 2 | 0.8624 | 0.5153 | 0.1101 | 0.8710 | 0.5789 |
| 3 | 0.8790 | 0.5980 | 0.1102 | 0.6585 | 0.4167 |
| 4 | 0.8191 | 0.4439 | 0.1605 | 0.7073 | 0.4194 |
| 5 | 0.7062 | 0.3165 | 0.2324 | 0.6016 | 0.3288 |
| **Mean** | **0.8160** | **0.4832** | **0.1511** | **0.7129** | **0.4354** |
| **±Std** | 0.0603 | 0.0970 | 0.0450 | 0.0900 | 0.0807 |

CrossAttn best val AUC per fold: Fold1=0.8135, Fold2=0.8624, Fold3=0.8790, Fold4=0.8191, Fold5=0.7062

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7978 | 0.3298 | 0.1763 | 0.7208 | 0.3768 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 63 | 0.8045 | 0.4296 | 0.2310 | 0.6190 | 0.4000 |
| F | 91 | 0.8008 | 0.2754 | 0.1384 | 0.7912 | 0.3448 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 98 | 39 |
| **True: Sarco**  | 4 | 13 |

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
