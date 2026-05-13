# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 16:39  |  5-Fold CV  |  Median best epoch: 23

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
| 1 | 0.5307 | 0.1465 | 0.2064 | 0.7402 | 0.1846 |
| 2 | 0.6245 | 0.1755 | 0.2144 | 0.7010 | 0.2278 |
| 3 | 0.5593 | 0.1331 | 0.2250 | 0.6995 | 0.2469 |
| 4 | 0.7235 | 0.3282 | 0.1944 | 0.7340 | 0.3077 |
| 5 | 0.6886 | 0.2452 | 0.1915 | 0.7389 | 0.2933 |
| **Mean** | **0.6253** | **0.2057** | **0.2064** | **0.7227** | **0.2521** |
| **±Std** | 0.0734 | 0.0724 | 0.0125 | 0.0185 | 0.0446 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8174 | 0.4285 | 0.1845 | 0.7402 | 0.3765 |
| 2 | 0.6786 | 0.2179 | 0.2089 | 0.7157 | 0.2750 |
| 3 | 0.7868 | 0.3250 | 0.1469 | 0.7882 | 0.3582 |
| 4 | 0.7855 | 0.3659 | 0.2148 | 0.6995 | 0.3579 |
| 5 | 0.9091 | 0.6699 | 0.1407 | 0.7783 | 0.4706 |
| **Mean** | **0.7955** | **0.4014** | **0.1791** | **0.7444** | **0.3676** |
| **±Std** | 0.0738 | 0.1507 | 0.0307 | 0.0344 | 0.0624 |

CrossAttn best val AUC per fold: Fold1=0.8174, Fold2=0.6786, Fold3=0.7868, Fold4=0.7855, Fold5=0.9091

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6654 | 0.2366 | 0.2036 | 0.7294 | 0.2887 |
| CrossAttn | 0.8447 | 0.4172 | 0.1749 | 0.7529 | 0.4220 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6270 | 0.3188 | 0.2594 | 0.6263 | 0.2745 |
| F | 156 | 0.6984 | 0.2138 | 0.1682 | 0.7949 | 0.3043 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.8077 | 0.4673 | 0.2240 | 0.6869 | 0.4918 |
| F | 156 | 0.8527 | 0.3348 | 0.1438 | 0.7949 | 0.3333 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 172 | 55 |
| **True: Sarco**  | 14 | 14 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 169 | 58 |
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
