# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 13:01  |  5-Fold CV  |  Median best epoch: 21

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 257 | 214 | 83.3% | 43 | 16.7% |
| Train | F | 363 | 330 | 90.9% | 33 | 9.1% |
| Train | **All** | **620** | **544** | **87.7%** | **76** | **12.3%** |
| Test | M | 57 | 51 | 89.5% | 6 | 10.5% |
| Test | F | 98 | 86 | 87.8% | 12 | 12.2% |
| Test | **All** | **155** | **137** | **88.4%** | **18** | **11.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 257 | 59.57 ± 11.72 | 28.00 | 60.00 | 89.00 |
| Train | F | 363 | 56.09 ± 12.40 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **620** | **57.53 ± 12.24** | **14.00** | **58.00** | **91.00** |
| Test | M | 57 | 58.04 ± 12.44 | 29.00 | 58.00 | 78.00 |
| Test | F | 98 | 56.66 ± 12.49 | 23.00 | 56.50 | 86.00 |
| Test | **All** | **155** | **57.17 ± 12.49** | **23.00** | **57.00** | **86.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 257 | 24.36 ± 3.09 | 14.34 | 24.28 | 32.67 |
| Train | F | 363 | 22.94 ± 3.14 | 16.00 | 22.77 | 34.61 |
| Train | **All** | **620** | **23.53 ± 3.19** | **14.34** | **23.37** | **34.61** |
| Test | M | 57 | 24.72 ± 2.87 | 19.22 | 24.11 | 32.56 |
| Test | F | 98 | 22.91 ± 3.63 | 12.02 | 22.58 | 34.20 |
| Test | **All** | **155** | **23.57 ± 3.48** | **12.02** | **23.24** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8275 | 0.3372 | 0.1628 | 0.7500 | 0.3111 |
| 2 | 0.8917 | 0.4888 | 0.1928 | 0.7339 | 0.4590 |
| 3 | 0.8709 | 0.5937 | 0.1266 | 0.8145 | 0.4889 |
| 4 | 0.7878 | 0.3574 | 0.2322 | 0.5887 | 0.3377 |
| 5 | 0.7616 | 0.3787 | 0.2200 | 0.6210 | 0.3380 |
| **Mean** | **0.8279** | **0.4312** | **0.1869** | **0.7016** | **0.3869** |
| **±Std** | 0.0489 | 0.0967 | 0.0385 | 0.0841 | 0.0723 |

CrossAttn best val AUC per fold: Fold1=0.8275, Fold2=0.8917, Fold3=0.8709, Fold4=0.7878, Fold5=0.7616

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8058 | 0.5116 | 0.1840 | 0.7161 | 0.3714 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 57 | 0.7026 | 0.2371 | 0.2206 | 0.7368 | 0.3478 |
| F | 98 | 0.8527 | 0.6803 | 0.1627 | 0.7041 | 0.3830 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 98 | 39 |
| **True: Sarco**  | 5 | 13 |

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
