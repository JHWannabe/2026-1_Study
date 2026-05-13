# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 18:55  |  5-Fold CV  |  Median best epoch: 40

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
| 1 | 0.5704 | 0.1823 | 0.2029 | 0.6863 | 0.2195 |
| 2 | 0.6486 | 0.1975 | 0.2123 | 0.6667 | 0.2273 |
| 3 | 0.6750 | 0.2201 | 0.2107 | 0.6847 | 0.2889 |
| 4 | 0.7024 | 0.3045 | 0.1944 | 0.7438 | 0.3500 |
| 5 | 0.7730 | 0.2986 | 0.1877 | 0.7094 | 0.3371 |
| **Mean** | **0.6739** | **0.2406** | **0.2016** | **0.6982** | **0.2846** |
| **±Std** | 0.0663 | 0.0512 | 0.0094 | 0.0266 | 0.0540 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8432 | 0.3867 | 0.1951 | 0.7353 | 0.4000 |
| 2 | 0.7060 | 0.2448 | 0.2453 | 0.6324 | 0.3119 |
| 3 | 0.8016 | 0.3720 | 0.2278 | 0.6158 | 0.3276 |
| 4 | 0.8207 | 0.3575 | 0.1762 | 0.7340 | 0.3721 |
| 5 | 0.9262 | 0.6804 | 0.1830 | 0.7389 | 0.4421 |
| **Mean** | **0.8195** | **0.4083** | **0.2055** | **0.6913** | **0.3707** |
| **±Std** | 0.0709 | 0.1450 | 0.0267 | 0.0551 | 0.0475 |

CrossAttn best val AUC per fold: Fold1=0.8432, Fold2=0.7060, Fold3=0.8016, Fold4=0.8207, Fold5=0.9262

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6946 | 0.2337 | 0.2028 | 0.7176 | 0.3077 |
| CrossAttn | 0.8140 | 0.3591 | 0.1656 | 0.7608 | 0.4078 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6714 | 0.3120 | 0.2430 | 0.6364 | 0.3333 |
| F | 156 | 0.6934 | 0.2006 | 0.1774 | 0.7692 | 0.2800 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7776 | 0.4546 | 0.2181 | 0.6566 | 0.4333 |
| F | 156 | 0.8006 | 0.2727 | 0.1324 | 0.8269 | 0.3721 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 167 | 60 |
| **True: Sarco**  | 12 | 16 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 173 | 54 |
| **True: Sarco**  | 7 | 21 |

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
