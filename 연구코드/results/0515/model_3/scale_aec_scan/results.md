# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 17:15  |  5-Fold CV  |  Median best epoch: 50

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
| 1 | 0.7184 | 0.2737 | 0.1704 | 0.7596 | 0.3590 |
| 2 | 0.8449 | 0.3689 | 0.1889 | 0.7115 | 0.3750 |
| 3 | 0.7572 | 0.3563 | 0.1605 | 0.7596 | 0.3243 |
| 4 | 0.7311 | 0.2494 | 0.1976 | 0.6971 | 0.3368 |
| 5 | 0.7934 | 0.2841 | 0.1762 | 0.7295 | 0.3488 |
| **Mean** | **0.7690** | **0.3065** | **0.1787** | **0.7315** | **0.3488** |
| **±Std** | 0.0458 | 0.0474 | 0.0132 | 0.0252 | 0.0175 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7692 | 0.3989 | 0.1427 | 0.7788 | 0.3030 |
| 2 | 0.8085 | 0.4659 | 0.1708 | 0.7452 | 0.4045 |
| 3 | 0.7946 | 0.4153 | 0.1009 | 0.8750 | 0.4583 |
| 4 | 0.7109 | 0.2989 | 0.2171 | 0.6346 | 0.2549 |
| 5 | 0.8408 | 0.3028 | 0.1778 | 0.7536 | 0.4000 |
| **Mean** | **0.7848** | **0.3764** | **0.1618** | **0.7575** | **0.3642** |
| **±Std** | 0.0436 | 0.0655 | 0.0386 | 0.0768 | 0.0741 |

CrossAttn best val AUC per fold: Fold1=0.7692, Fold2=0.8085, Fold3=0.7946, Fold4=0.7109, Fold5=0.8408

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7603 | 0.2597 | 0.1883 | 0.7115 | 0.3119 |
| CrossAttn | 0.7040 | 0.2031 | 0.1556 | 0.7885 | 0.2857 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.7508 | 0.3299 | 0.2478 | 0.6122 | 0.3871 |
| F | 162 | 0.7111 | 0.1610 | 0.1523 | 0.7716 | 0.2128 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.6319 | 0.2116 | 0.2383 | 0.6224 | 0.3019 |
| F | 162 | 0.6767 | 0.2376 | 0.1055 | 0.8889 | 0.2500 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 64 |
| **True: Sarco**  | 11 | 17 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 194 | 38 |
| **True: Sarco**  | 17 | 11 |

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
