# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 19:32  |  5-Fold CV  |  Median best epoch: 9

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
| 1 | 0.4900 | 0.1315 | 0.2193 | 0.7059 | 0.1429 |
| 2 | 0.6020 | 0.1634 | 0.2158 | 0.6961 | 0.1842 |
| 3 | 0.5321 | 0.1174 | 0.2297 | 0.6700 | 0.1299 |
| 4 | 0.6818 | 0.2623 | 0.2146 | 0.7044 | 0.2857 |
| 5 | 0.6243 | 0.1859 | 0.2006 | 0.7389 | 0.2535 |
| **Mean** | **0.5861** | **0.1721** | **0.2160** | **0.7031** | **0.1992** |
| **±Std** | 0.0679 | 0.0510 | 0.0093 | 0.0221 | 0.0611 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8267 | 0.3844 | 0.1530 | 0.7843 | 0.3889 |
| 2 | 0.7134 | 0.2836 | 0.1881 | 0.7451 | 0.3500 |
| 3 | 0.7735 | 0.3508 | 0.1985 | 0.6601 | 0.3301 |
| 4 | 0.8041 | 0.3850 | 0.2198 | 0.6601 | 0.3551 |
| 5 | 0.9119 | 0.6210 | 0.1434 | 0.7931 | 0.5000 |
| **Mean** | **0.8059** | **0.4050** | **0.1806** | **0.7285** | **0.3848** |
| **±Std** | 0.0652 | 0.1142 | 0.0285 | 0.0582 | 0.0606 |

CrossAttn best val AUC per fold: Fold1=0.8267, Fold2=0.7134, Fold3=0.7735, Fold4=0.8041, Fold5=0.9119

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6449 | 0.2041 | 0.2092 | 0.7098 | 0.2600 |
| CrossAttn | 0.8087 | 0.3404 | 0.1988 | 0.7294 | 0.3670 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6055 | 0.2876 | 0.2654 | 0.6263 | 0.2745 |
| F | 156 | 0.6759 | 0.1811 | 0.1736 | 0.7628 | 0.2449 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7640 | 0.3663 | 0.2310 | 0.6970 | 0.4444 |
| F | 156 | 0.8332 | 0.3470 | 0.1783 | 0.7500 | 0.2909 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 59 |
| **True: Sarco**  | 15 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 166 | 61 |
| **True: Sarco**  | 8 | 20 |

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
