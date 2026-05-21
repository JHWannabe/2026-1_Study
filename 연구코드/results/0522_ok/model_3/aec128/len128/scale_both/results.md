# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 12:04  |  5-Fold CV  |  Median best epoch: 15

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
| 1 | 0.7989 | 0.5251 | 0.2467 | 0.5985 | 0.3117 |
| 2 | 0.8475 | 0.4432 | 0.1823 | 0.7348 | 0.4444 |
| 3 | 0.8017 | 0.3712 | 0.2452 | 0.6136 | 0.3704 |
| 4 | 0.8475 | 0.4608 | 0.1479 | 0.7803 | 0.4912 |
| 5 | 0.8561 | 0.4308 | 0.1232 | 0.7879 | 0.4815 |
| **Mean** | **0.8304** | **0.4462** | **0.1891** | **0.7030** | **0.4198** |
| **±Std** | 0.0248 | 0.0496 | 0.0501 | 0.0814 | 0.0688 |

CrossAttn best val AUC per fold: Fold1=0.7989, Fold2=0.8475, Fold3=0.8017, Fold4=0.8475, Fold5=0.8561

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7550 | 0.4024 | 0.1833 | 0.7048 | 0.3797 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 71 | 0.7242 | 0.3766 | 0.2261 | 0.6338 | 0.4091 |
| F | 95 | 0.7682 | 0.4506 | 0.1513 | 0.7579 | 0.3429 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 102 | 43 |
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
