# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-19 13:08  |  5-Fold CV  |  Median best epoch: 19

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 231 | 189 | 81.8% | 42 | 18.2% |
| Train | F | 510 | 476 | 93.3% | 34 | 6.7% |
| Train | **All** | **741** | **665** | **89.7%** | **76** | **10.3%** |
| Test | M | 60 | 49 | 81.7% | 11 | 18.3% |
| Test | F | 125 | 118 | 94.4% | 7 | 5.6% |
| Test | **All** | **185** | **167** | **90.3%** | **18** | **9.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 231 | 61.74 ± 11.21 | 28.00 | 62.00 | 89.00 |
| Train | F | 510 | 55.09 ± 11.32 | 24.00 | 55.00 | 87.00 |
| Train | **All** | **741** | **57.16 ± 11.70** | **24.00** | **57.00** | **89.00** |
| Test | M | 60 | 64.63 ± 11.78 | 31.00 | 67.50 | 84.00 |
| Test | F | 125 | 56.86 ± 11.67 | 27.00 | 57.00 | 91.00 |
| Test | **All** | **185** | **59.38 ± 12.26** | **27.00** | **59.00** | **91.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 231 | 24.22 ± 2.77 | 15.22 | 24.44 | 31.94 |
| Train | F | 510 | 23.21 ± 3.12 | 15.63 | 23.00 | 31.78 |
| Train | **All** | **741** | **23.52 ± 3.05** | **15.22** | **23.42** | **31.94** |
| Test | M | 60 | 23.73 ± 2.50 | 19.29 | 23.34 | 29.65 |
| Test | F | 125 | 23.25 ± 2.76 | 16.80 | 23.12 | 30.05 |
| Test | **All** | **185** | **23.40 ± 2.69** | **16.80** | **23.28** | **30.05** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8430 | 0.4661 | 0.2135 | 0.6443 | 0.3457 |
| 2 | 0.8351 | 0.4773 | 0.1953 | 0.6959 | 0.3284 |
| 3 | 0.9198 | 0.6362 | 0.1169 | 0.8311 | 0.5098 |
| 4 | 0.8792 | 0.3750 | 0.1498 | 0.7973 | 0.4444 |
| 5 | 0.8130 | 0.2816 | 0.1633 | 0.7635 | 0.4262 |
| **Mean** | **0.8580** | **0.4473** | **0.1678** | **0.7464** | **0.4109** |
| **±Std** | 0.0375 | 0.1180 | 0.0340 | 0.0679 | 0.0666 |

CrossAttn best val AUC per fold: Fold1=0.8430, Fold2=0.8351, Fold3=0.9198, Fold4=0.8792, Fold5=0.8130

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.6933 | 0.1908 | 0.2366 | 0.6865 | 0.3256 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 60 | 0.6364 | 0.2465 | 0.4021 | 0.4333 | 0.3704 |
| F | 125 | 0.5654 | 0.1092 | 0.1572 | 0.8080 | 0.2500 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 113 | 54 |
| **True: Sarco**  | 4 | 14 |

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
