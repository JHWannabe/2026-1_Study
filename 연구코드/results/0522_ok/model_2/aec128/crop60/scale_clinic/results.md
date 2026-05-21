# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 19:37  |  5-Fold CV  |  Median best epoch: 37

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 285 | 241 | 84.6% | 44 | 15.4% |
| Train | F | 402 | 364 | 90.5% | 38 | 9.5% |
| Train | **All** | **687** | **605** | **88.1%** | **82** | **11.9%** |
| Test | M | 71 | 60 | 84.5% | 11 | 15.5% |
| Test | F | 101 | 91 | 90.1% | 10 | 9.9% |
| Test | **All** | **172** | **151** | **87.8%** | **21** | **12.2%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 285 | 59.36 ± 11.63 | 23.00 | 59.00 | 85.00 |
| Train | F | 402 | 56.39 ± 12.21 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **687** | **57.62 ± 12.06** | **14.00** | **58.00** | **91.00** |
| Test | M | 71 | 59.82 ± 12.49 | 28.00 | 61.00 | 89.00 |
| Test | F | 101 | 55.65 ± 12.38 | 24.00 | 56.00 | 86.00 |
| Test | **All** | **172** | **57.37 ± 12.59** | **24.00** | **57.50** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 285 | 24.43 ± 2.95 | 14.34 | 24.28 | 32.67 |
| Train | F | 402 | 22.98 ± 3.35 | 12.02 | 22.81 | 34.61 |
| Train | **All** | **687** | **23.58 ± 3.27** | **12.02** | **23.44** | **34.61** |
| Test | M | 71 | 24.27 ± 3.24 | 17.33 | 24.12 | 32.33 |
| Test | F | 101 | 22.90 ± 3.20 | 16.00 | 22.64 | 34.20 |
| Test | **All** | **172** | **23.47 ± 3.29** | **16.00** | **23.27** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7720 | 0.2924 | 0.1591 | 0.7609 | 0.4407 |
| 2 | 0.8177 | 0.4347 | 0.1724 | 0.7464 | 0.4444 |
| 3 | 0.8332 | 0.3413 | 0.1998 | 0.6715 | 0.3836 |
| 4 | 0.8228 | 0.4522 | 0.1987 | 0.6642 | 0.3611 |
| 5 | 0.8120 | 0.2900 | 0.1874 | 0.7591 | 0.4407 |
| **Mean** | **0.8115** | **0.3621** | **0.1835** | **0.7204** | **0.4141** |
| **±Std** | 0.0210 | 0.0691 | 0.0157 | 0.0433 | 0.0349 |

CrossAttn best val AUC per fold: Fold1=0.7720, Fold2=0.8177, Fold3=0.8332, Fold4=0.8228, Fold5=0.8120

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8310 | 0.4444 | 0.2049 | 0.7093 | 0.4318 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 71 | 0.8364 | 0.5467 | 0.2466 | 0.6479 | 0.4444 |
| F | 101 | 0.8286 | 0.3730 | 0.1755 | 0.7525 | 0.4186 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 103 | 48 |
| **True: Sarco**  | 2 | 19 |

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
