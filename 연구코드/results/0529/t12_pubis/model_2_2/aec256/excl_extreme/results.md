# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-27 16:54  |  5-Fold CV  |  Median best epoch: 10

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 252 | 214 | 84.9% | 38 | 15.1% |
| Train | F | 365 | 329 | 90.1% | 36 | 9.9% |
| Train | **All** | **617** | **543** | **88.0%** | **74** | **12.0%** |
| Test | M | 63 | 55 | 87.3% | 8 | 12.7% |
| Test | F | 91 | 83 | 91.2% | 8 | 8.8% |
| Test | **All** | **154** | **138** | **89.6%** | **16** | **10.4%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 252 | 59.62 ± 11.73 | 23.00 | 59.00 | 85.00 |
| Train | F | 365 | 56.58 ± 12.24 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.82 ± 12.13** | **14.00** | **58.00** | **91.00** |
| Test | M | 63 | 59.02 ± 12.66 | 28.00 | 61.00 | 89.00 |
| Test | F | 91 | 55.74 ± 12.87 | 24.00 | 56.00 | 86.00 |
| Test | **All** | **154** | **57.08 ± 12.89** | **24.00** | **57.50** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 252 | 24.55 ± 2.97 | 14.34 | 24.47 | 32.67 |
| Train | F | 365 | 22.93 ± 3.37 | 12.02 | 22.77 | 34.61 |
| Train | **All** | **617** | **23.59 ± 3.31** | **12.02** | **23.46** | **34.61** |
| Test | M | 63 | 24.50 ± 3.21 | 17.33 | 24.12 | 32.33 |
| Test | F | 91 | 23.00 ± 3.30 | 16.00 | 22.76 | 34.20 |
| Test | **All** | **154** | **23.62 ± 3.34** | **16.00** | **23.35** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8122 | 0.3714 | 0.3219 | 0.7823 | 0.4706 |
| 2 | 0.8758 | 0.5430 | 0.1558 | 0.7661 | 0.4727 |
| 3 | 0.8019 | 0.3871 | 0.2033 | 0.6585 | 0.3824 |
| 4 | 0.8691 | 0.4949 | 0.1884 | 0.8780 | 0.5946 |
| 5 | 0.7693 | 0.3198 | 0.2044 | 0.7642 | 0.4082 |
| **Mean** | **0.8257** | **0.4232** | **0.2148** | **0.7698** | **0.4657** |
| **±Std** | 0.0408 | 0.0826 | 0.0563 | 0.0697 | 0.0734 |

CrossAttn best val AUC per fold: Fold1=0.8122, Fold2=0.8758, Fold3=0.8019, Fold4=0.8691, Fold5=0.7693

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8764 | 0.4707 | 0.1693 | 0.8052 | 0.4444 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 63 | 0.8955 | 0.5936 | 0.2000 | 0.7778 | 0.5000 |
| F | 91 | 0.8630 | 0.3980 | 0.1480 | 0.8242 | 0.3846 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 112 | 26 |
| **True: Sarco**  | 4 | 12 |

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
