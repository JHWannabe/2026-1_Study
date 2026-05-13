# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:56  |  5-Fold CV  |  Median best epoch: 5

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 311 | 262 | 84.2% | 49 | 15.8% |
| Train | F | 604 | 556 | 92.1% | 48 | 7.9% |
| Train | **All** | **915** | **818** | **89.4%** | **97** | **10.6%** |
| Test | M | 86 | 73 | 84.9% | 13 | 15.1% |
| Test | F | 143 | 132 | 92.3% | 11 | 7.7% |
| Test | **All** | **229** | **205** | **89.5%** | **24** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 311 | 59.30 ± 12.73 | 18.00 | 60.00 | 88.00 |
| Train | F | 604 | 55.54 ± 11.62 | 14.00 | 55.50 | 91.00 |
| Train | **All** | **915** | **56.82 ± 12.14** | **14.00** | **57.00** | **91.00** |
| Test | M | 86 | 61.66 ± 10.80 | 34.00 | 61.00 | 89.00 |
| Test | F | 143 | 55.24 ± 13.24 | 11.00 | 54.00 | 86.00 |
| Test | **All** | **229** | **57.65 ± 12.77** | **11.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 311 | 23.91 ± 3.12 | 14.48 | 24.03 | 35.20 |
| Train | F | 604 | 22.91 ± 3.10 | 15.62 | 22.64 | 34.29 |
| Train | **All** | **915** | **23.25 ± 3.14** | **14.48** | **23.09** | **35.20** |
| Test | M | 86 | 24.09 ± 3.18 | 16.80 | 24.42 | 33.87 |
| Test | F | 143 | 23.04 ± 3.49 | 14.40 | 22.53 | 36.24 |
| Test | **All** | **229** | **23.44 ± 3.42** | **14.40** | **23.38** | **36.24** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7346 | 0.2754 | 0.1752 | 0.7377 | 0.2727 |
| 2 | 0.7465 | 0.4069 | 0.1773 | 0.7322 | 0.3099 |
| 3 | 0.7153 | 0.2257 | 0.1685 | 0.7541 | 0.3077 |
| 4 | 0.7920 | 0.4322 | 0.1977 | 0.6557 | 0.3077 |
| 5 | 0.7617 | 0.2939 | 0.1726 | 0.7596 | 0.3714 |
| **Mean** | **0.7500** | **0.3268** | **0.1783** | **0.7279** | **0.3139** |
| **±Std** | 0.0259 | 0.0793 | 0.0101 | 0.0374 | 0.0319 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7417 | 0.2358 | 0.2031 | 0.7049 | 0.3415 |
| 2 | 0.8030 | 0.2966 | 0.2769 | 0.5628 | 0.3103 |
| 3 | 0.7596 | 0.2885 | 0.1933 | 0.6940 | 0.3333 |
| 4 | 0.8623 | 0.5615 | 0.1799 | 0.7268 | 0.3750 |
| 5 | 0.8160 | 0.4032 | 0.1225 | 0.8361 | 0.5000 |
| **Mean** | **0.7965** | **0.3571** | **0.1951** | **0.7049** | **0.3720** |
| **±Std** | 0.0427 | 0.1157 | 0.0495 | 0.0872 | 0.0673 |

CrossAttn best val AUC per fold: Fold1=0.7417, Fold2=0.8030, Fold3=0.7596, Fold4=0.8623, Fold5=0.8160

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8014 | 0.3209 | 0.1814 | 0.7293 | 0.3542 |
| CrossAttn | 0.7990 | 0.3125 | 0.1678 | 0.7598 | 0.4086 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 86 | 0.7534 | 0.3572 | 0.2349 | 0.6512 | 0.4000 |
| F | 143 | 0.8127 | 0.3408 | 0.1492 | 0.7762 | 0.3043 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 86 | 0.7977 | 0.3441 | 0.1883 | 0.7326 | 0.4889 |
| F | 143 | 0.7879 | 0.3911 | 0.1554 | 0.7762 | 0.3333 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 150 | 55 |
| **True: Sarco**  | 7 | 17 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 155 | 50 |
| **True: Sarco**  | 5 | 19 |

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
