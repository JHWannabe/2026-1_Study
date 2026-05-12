# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 16:58  |  5-Fold CV  |  Median best epoch: 109

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 402 | 332 | 82.6% | 70 | 17.4% |
| Train | F | 695 | 645 | 92.8% | 50 | 7.2% |
| Train | **All** | **1097** | **977** | **89.1%** | **120** | **10.9%** |
| Test | M | 112 | 95 | 84.8% | 17 | 15.2% |
| Test | F | 163 | 150 | 92.0% | 13 | 8.0% |
| Test | **All** | **275** | **245** | **89.1%** | **30** | **10.9%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 402 | 59.81 ± 12.51 | 18.00 | 60.00 | 89.00 |
| Train | F | 695 | 55.36 ± 12.15 | 11.00 | 55.00 | 91.00 |
| Train | **All** | **1097** | **56.99 ± 12.47** | **11.00** | **58.00** | **91.00** |
| Test | M | 112 | 59.05 ± 12.52 | 23.00 | 59.50 | 84.00 |
| Test | F | 163 | 56.52 ± 12.29 | 22.00 | 56.00 | 87.00 |
| Test | **All** | **275** | **57.55 ± 12.45** | **22.00** | **58.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 402 | 24.22 ± 3.26 | 14.48 | 24.19 | 36.76 |
| Train | F | 695 | 23.09 ± 3.43 | 14.40 | 22.70 | 39.49 |
| Train | **All** | **1097** | **23.51 ± 3.41** | **14.40** | **23.30** | **39.49** |
| Test | M | 112 | 24.07 ± 3.30 | 16.44 | 24.16 | 35.20 |
| Test | F | 163 | 22.99 ± 3.19 | 16.06 | 22.83 | 34.23 |
| Test | **All** | **275** | **23.43 ± 3.28** | **16.06** | **23.44** | **35.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7338 | 0.2588 | 0.2129 | 0.6864 | 0.3301 |
| 2 | 0.7768 | 0.3119 | 0.1645 | 0.7636 | 0.3333 |
| 3 | 0.7709 | 0.3381 | 0.1963 | 0.7169 | 0.3542 |
| 4 | 0.7365 | 0.2290 | 0.1727 | 0.7671 | 0.3544 |
| 5 | 0.8447 | 0.4545 | 0.1604 | 0.7763 | 0.4235 |
| **Mean** | **0.7726** | **0.3185** | **0.1813** | **0.7421** | **0.3591** |
| **±Std** | 0.0400 | 0.0781 | 0.0201 | 0.0346 | 0.0338 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8304 | 0.3805 | 0.2044 | 0.7318 | 0.4040 |
| 2 | 0.7936 | 0.3975 | 0.1760 | 0.8000 | 0.3889 |
| 3 | 0.7361 | 0.2344 | 0.2538 | 0.6393 | 0.3130 |
| 4 | 0.7904 | 0.3541 | 0.1481 | 0.8265 | 0.4242 |
| 5 | 0.8692 | 0.5114 | 0.2025 | 0.6712 | 0.3898 |
| **Mean** | **0.8039** | **0.3756** | **0.1970** | **0.7338** | **0.3840** |
| **±Std** | 0.0444 | 0.0887 | 0.0350 | 0.0719 | 0.0377 |

CrossAttn best val AUC per fold: Fold1=0.8304, Fold2=0.7936, Fold3=0.7361, Fold4=0.7904, Fold5=0.8692

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7396 | 0.3390 | 0.1916 | 0.7273 | 0.3590 |
| CrossAttn | 0.7521 | 0.2796 | 0.2092 | 0.7455 | 0.3636 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7672 | 0.4870 | 0.2624 | 0.5893 | 0.3784 |
| F | 163 | 0.7169 | 0.1879 | 0.1430 | 0.8221 | 0.3256 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7486 | 0.3267 | 0.3194 | 0.5000 | 0.3333 |
| F | 163 | 0.7349 | 0.3341 | 0.1334 | 0.9141 | 0.4615 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 179 | 66 |
| **True: Sarco**  | 9 | 21 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 185 | 60 |
| **True: Sarco**  | 10 | 20 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR & CrossAttn) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curves |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices (LR & CrossAttn) |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
