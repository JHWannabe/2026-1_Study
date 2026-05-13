# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 17:35  |  5-Fold CV  |  Median best epoch: 4

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
| 1 | 0.6201 | 0.1876 | 0.2070 | 0.7010 | 0.1644 |
| 2 | 0.6010 | 0.1977 | 0.1874 | 0.7402 | 0.2535 |
| 3 | 0.6371 | 0.1547 | 0.2201 | 0.6798 | 0.2857 |
| 4 | 0.6688 | 0.2829 | 0.1991 | 0.6995 | 0.2651 |
| 5 | 0.7022 | 0.2882 | 0.1815 | 0.7488 | 0.3377 |
| **Mean** | **0.6458** | **0.2222** | **0.1990** | **0.7139** | **0.2613** |
| **±Std** | 0.0359 | 0.0537 | 0.0138 | 0.0262 | 0.0564 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8014 | 0.3922 | 0.1881 | 0.7647 | 0.3684 |
| 2 | 0.6918 | 0.2526 | 0.1722 | 0.7304 | 0.3038 |
| 3 | 0.8046 | 0.3388 | 0.1854 | 0.7291 | 0.3820 |
| 4 | 0.8009 | 0.3834 | 0.1381 | 0.7931 | 0.3000 |
| 5 | 0.9103 | 0.6604 | 0.1698 | 0.7094 | 0.4158 |
| **Mean** | **0.8018** | **0.4055** | **0.1707** | **0.7453** | **0.3540** |
| **±Std** | 0.0691 | 0.1367 | 0.0178 | 0.0298 | 0.0453 |

CrossAttn best val AUC per fold: Fold1=0.8014, Fold2=0.6918, Fold3=0.8046, Fold4=0.8009, Fold5=0.9103

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7486 | 0.2695 | 0.1946 | 0.7373 | 0.3366 |
| CrossAttn | 0.8177 | 0.3563 | 0.2219 | 0.6667 | 0.3511 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7209 | 0.3287 | 0.2223 | 0.7071 | 0.4082 |
| F | 156 | 0.7705 | 0.2737 | 0.1770 | 0.7564 | 0.2692 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7676 | 0.3723 | 0.2644 | 0.5758 | 0.4167 |
| F | 156 | 0.8169 | 0.4138 | 0.1950 | 0.7244 | 0.2712 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 171 | 56 |
| **True: Sarco**  | 11 | 17 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 147 | 80 |
| **True: Sarco**  | 5 | 23 |

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
