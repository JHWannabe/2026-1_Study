# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 20:10  |  5-Fold CV  |  Median best epoch: 7

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 309 | 261 | 84.5% | 48 | 15.5% |
| Train | F | 605 | 559 | 92.4% | 46 | 7.6% |
| Train | **All** | **914** | **820** | **89.7%** | **94** | **10.3%** |
| Test | M | 83 | 70 | 84.3% | 13 | 15.7% |
| Test | F | 146 | 133 | 91.1% | 13 | 8.9% |
| Test | **All** | **229** | **203** | **88.6%** | **26** | **11.4%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 309 | 59.60 ± 12.50 | 18.00 | 60.00 | 89.00 |
| Train | F | 605 | 55.60 ± 11.93 | 18.00 | 55.00 | 91.00 |
| Train | **All** | **914** | **56.95 ± 12.27** | **18.00** | **57.00** | **91.00** |
| Test | M | 83 | 59.20 ± 12.71 | 28.00 | 60.00 | 88.00 |
| Test | F | 146 | 54.76 ± 11.42 | 23.00 | 55.00 | 86.00 |
| Test | **All** | **229** | **56.37 ± 12.09** | **23.00** | **57.00** | **88.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 309 | 24.01 ± 3.23 | 14.48 | 24.07 | 35.20 |
| Train | F | 605 | 23.00 ± 3.25 | 14.40 | 22.72 | 36.24 |
| Train | **All** | **914** | **23.34 ± 3.27** | **14.40** | **23.24** | **36.24** |
| Test | M | 83 | 24.36 ± 2.96 | 18.37 | 24.39 | 33.87 |
| Test | F | 146 | 22.97 ± 3.08 | 16.87 | 22.65 | 34.23 |
| Test | **All** | **229** | **23.47 ± 3.11** | **16.87** | **23.28** | **34.23** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7372 | 0.2447 | 0.1490 | 0.7869 | 0.3158 |
| 2 | 0.7651 | 0.3432 | 0.1716 | 0.7541 | 0.3284 |
| 3 | 0.8312 | 0.4143 | 0.1582 | 0.7596 | 0.4054 |
| 4 | 0.7840 | 0.3739 | 0.1770 | 0.7377 | 0.3684 |
| 5 | 0.7209 | 0.2719 | 0.1949 | 0.7033 | 0.2895 |
| **Mean** | **0.7677** | **0.3296** | **0.1701** | **0.7483** | **0.3415** |
| **±Std** | 0.0385 | 0.0630 | 0.0158 | 0.0275 | 0.0409 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7721 | 0.2666 | 0.1484 | 0.7705 | 0.3636 |
| 2 | 0.8543 | 0.5059 | 0.2293 | 0.6503 | 0.3469 |
| 3 | 0.9166 | 0.5266 | 0.2360 | 0.6503 | 0.3600 |
| 4 | 0.7924 | 0.3595 | 0.2412 | 0.6120 | 0.3238 |
| 5 | 0.8388 | 0.4317 | 0.1381 | 0.8242 | 0.4667 |
| **Mean** | **0.8348** | **0.4181** | **0.1986** | **0.7014** | **0.3722** |
| **±Std** | 0.0506 | 0.0960 | 0.0455 | 0.0813 | 0.0492 |

CrossAttn best val AUC per fold: Fold1=0.7721, Fold2=0.8543, Fold3=0.9166, Fold4=0.7924, Fold5=0.8388

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7101 | 0.2672 | 0.1767 | 0.7511 | 0.2963 |
| CrossAttn | 0.7622 | 0.2971 | 0.2004 | 0.6900 | 0.3238 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 83 | 0.7253 | 0.3819 | 0.2132 | 0.6988 | 0.3902 |
| F | 146 | 0.6732 | 0.1653 | 0.1558 | 0.7808 | 0.2000 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 83 | 0.7725 | 0.4073 | 0.2254 | 0.6988 | 0.4444 |
| F | 146 | 0.7386 | 0.2257 | 0.1861 | 0.6849 | 0.2333 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 160 | 43 |
| **True: Sarco**  | 14 | 12 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 141 | 62 |
| **True: Sarco**  | 9 | 17 |

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
