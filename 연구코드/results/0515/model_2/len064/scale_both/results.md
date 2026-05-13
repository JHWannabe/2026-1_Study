# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:04  |  5-Fold CV  |  Median best epoch: 5

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 357 | 297 | 83.2% | 60 | 16.8% |
| Train | F | 660 | 609 | 92.3% | 51 | 7.7% |
| Train | **All** | **1017** | **906** | **89.1%** | **111** | **10.9%** |
| Test | M | 99 | 82 | 82.8% | 17 | 17.2% |
| Test | F | 156 | 145 | 92.9% | 11 | 7.1% |
| Test | **All** | **255** | **227** | **89.0%** | **28** | **11.0%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 357 | 59.35 ± 12.80 | 18.00 | 60.00 | 88.00 |
| Train | F | 660 | 55.60 ± 11.74 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **1017** | **56.92 ± 12.25** | **14.00** | **57.00** | **91.00** |
| Test | M | 99 | 61.58 ± 10.97 | 34.00 | 61.00 | 89.00 |
| Test | F | 156 | 55.71 ± 13.31 | 11.00 | 55.00 | 86.00 |
| Test | **All** | **255** | **57.98 ± 12.78** | **11.00** | **59.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 357 | 24.17 ± 3.31 | 14.48 | 24.16 | 36.76 |
| Train | F | 660 | 23.01 ± 3.24 | 15.62 | 22.69 | 34.61 |
| Train | **All** | **1017** | **23.42 ± 3.31** | **14.48** | **23.24** | **36.76** |
| Test | M | 99 | 24.03 ± 3.22 | 16.80 | 24.16 | 33.87 |
| Test | F | 156 | 23.22 ± 3.52 | 14.40 | 22.71 | 36.24 |
| Test | **All** | **255** | **23.54 ± 3.43** | **14.40** | **23.53** | **36.24** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7500 | 0.2748 | 0.1760 | 0.7598 | 0.3467 |
| 2 | 0.6680 | 0.2693 | 0.1931 | 0.7255 | 0.2432 |
| 3 | 0.7639 | 0.2453 | 0.1981 | 0.6798 | 0.3434 |
| 4 | 0.8430 | 0.4831 | 0.1639 | 0.7882 | 0.4416 |
| 5 | 0.8594 | 0.4762 | 0.1825 | 0.7192 | 0.3871 |
| **Mean** | **0.7769** | **0.3497** | **0.1827** | **0.7345** | **0.3524** |
| **±Std** | 0.0692 | 0.1066 | 0.0122 | 0.0370 | 0.0651 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8244 | 0.3836 | 0.1445 | 0.7843 | 0.3714 |
| 2 | 0.6925 | 0.2318 | 0.1948 | 0.7206 | 0.3133 |
| 3 | 0.7961 | 0.2934 | 0.2638 | 0.5419 | 0.2791 |
| 4 | 0.7961 | 0.3676 | 0.1881 | 0.6946 | 0.3673 |
| 5 | 0.9219 | 0.6285 | 0.1532 | 0.7389 | 0.4421 |
| **Mean** | **0.8062** | **0.3810** | **0.1889** | **0.6961** | **0.3546** |
| **±Std** | 0.0733 | 0.1352 | 0.0422 | 0.0825 | 0.0557 |

CrossAttn best val AUC per fold: Fold1=0.8244, Fold2=0.6925, Fold3=0.7961, Fold4=0.7961, Fold5=0.9219

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8147 | 0.3195 | 0.1846 | 0.7412 | 0.4000 |
| CrossAttn | 0.8403 | 0.3367 | 0.2253 | 0.6784 | 0.3788 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7554 | 0.3357 | 0.2536 | 0.6364 | 0.4706 |
| F | 156 | 0.8107 | 0.3734 | 0.1408 | 0.8077 | 0.2857 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7812 | 0.3665 | 0.2662 | 0.6364 | 0.4706 |
| F | 156 | 0.8476 | 0.4301 | 0.1994 | 0.7051 | 0.2812 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 60 |
| **True: Sarco**  | 6 | 22 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 148 | 79 |
| **True: Sarco**  | 3 | 25 |

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
