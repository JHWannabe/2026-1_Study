# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 17:07  |  5-Fold CV  |  Median best epoch: 17

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 373 | 309 | 82.8% | 64 | 17.2% |
| Train | F | 666 | 616 | 92.5% | 50 | 7.5% |
| Train | **All** | **1039** | **925** | **89.0%** | **114** | **11.0%** |
| Test | M | 98 | 82 | 83.7% | 16 | 16.3% |
| Test | F | 162 | 150 | 92.6% | 12 | 7.4% |
| Test | **All** | **260** | **232** | **89.2%** | **28** | **10.8%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 373 | 59.36 ± 12.75 | 18.00 | 59.00 | 89.00 |
| Train | F | 666 | 55.63 ± 12.02 | 14.00 | 55.00 | 87.00 |
| Train | **All** | **1039** | **56.97 ± 12.41** | **14.00** | **57.00** | **89.00** |
| Test | M | 98 | 61.47 ± 11.63 | 20.00 | 62.50 | 84.00 |
| Test | F | 162 | 55.33 ± 12.79 | 11.00 | 55.50 | 91.00 |
| Test | **All** | **260** | **57.64 ± 12.72** | **11.00** | **58.00** | **91.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 373 | 24.19 ± 3.35 | 14.48 | 24.22 | 36.76 |
| Train | F | 666 | 23.06 ± 3.39 | 14.40 | 22.75 | 39.49 |
| Train | **All** | **1039** | **23.46 ± 3.42** | **14.40** | **23.29** | **39.49** |
| Test | M | 98 | 24.06 ± 2.92 | 17.03 | 24.12 | 31.51 |
| Test | F | 162 | 23.12 ± 3.29 | 16.44 | 22.66 | 34.61 |
| Test | **All** | **260** | **23.47 ± 3.19** | **16.44** | **23.39** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7192 | 0.2735 | 0.1696 | 0.7596 | 0.3590 |
| 2 | 0.8428 | 0.3645 | 0.1893 | 0.7115 | 0.3750 |
| 3 | 0.7608 | 0.3589 | 0.1603 | 0.7644 | 0.3288 |
| 4 | 0.7340 | 0.2553 | 0.1971 | 0.6875 | 0.3434 |
| 5 | 0.7968 | 0.2901 | 0.1750 | 0.7343 | 0.3529 |
| **Mean** | **0.7707** | **0.3085** | **0.1783** | **0.7315** | **0.3518** |
| **±Std** | 0.0447 | 0.0449 | 0.0133 | 0.0290 | 0.0154 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8059 | 0.4094 | 0.2096 | 0.6827 | 0.3774 |
| 2 | 0.8961 | 0.6136 | 0.1542 | 0.7596 | 0.4318 |
| 3 | 0.8134 | 0.4522 | 0.1091 | 0.8413 | 0.4407 |
| 4 | 0.7523 | 0.2997 | 0.2362 | 0.6442 | 0.3273 |
| 5 | 0.8457 | 0.3176 | 0.0972 | 0.8744 | 0.1875 |
| **Mean** | **0.8227** | **0.4185** | **0.1613** | **0.7605** | **0.3529** |
| **±Std** | 0.0474 | 0.1128 | 0.0545 | 0.0884 | 0.0922 |

CrossAttn best val AUC per fold: Fold1=0.8059, Fold2=0.8961, Fold3=0.8134, Fold4=0.7523, Fold5=0.8457

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7597 | 0.2572 | 0.1880 | 0.7154 | 0.3148 |
| CrossAttn | 0.7503 | 0.2393 | 0.1588 | 0.7885 | 0.3373 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.7492 | 0.3290 | 0.2491 | 0.6122 | 0.3871 |
| F | 162 | 0.7111 | 0.1608 | 0.1510 | 0.7778 | 0.2174 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.6768 | 0.2337 | 0.2394 | 0.6429 | 0.3636 |
| F | 162 | 0.7478 | 0.2978 | 0.1101 | 0.8765 | 0.2857 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 169 | 63 |
| **True: Sarco**  | 11 | 17 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 191 | 41 |
| **True: Sarco**  | 14 | 14 |

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
