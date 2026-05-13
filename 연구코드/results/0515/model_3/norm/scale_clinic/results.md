# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:31  |  5-Fold CV  |  Median best epoch: 2

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
| 1 | 0.7063 | 0.2043 | 0.1789 | 0.7696 | 0.3380 |
| 2 | 0.6428 | 0.2984 | 0.1943 | 0.7157 | 0.2368 |
| 3 | 0.7438 | 0.2150 | 0.1990 | 0.7241 | 0.3636 |
| 4 | 0.8154 | 0.4361 | 0.1701 | 0.7635 | 0.3846 |
| 5 | 0.8225 | 0.3616 | 0.1792 | 0.7389 | 0.3765 |
| **Mean** | **0.7462** | **0.3031** | **0.1843** | **0.7424** | **0.3399** |
| **±Std** | 0.0677 | 0.0879 | 0.0107 | 0.0212 | 0.0539 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8019 | 0.4620 | 0.1954 | 0.7451 | 0.3810 |
| 2 | 0.6836 | 0.2775 | 0.1578 | 0.7549 | 0.3056 |
| 3 | 0.7978 | 0.2945 | 0.2215 | 0.6650 | 0.3462 |
| 4 | 0.7938 | 0.4202 | 0.1844 | 0.7241 | 0.3913 |
| 5 | 0.9184 | 0.5924 | 0.1732 | 0.7438 | 0.4348 |
| **Mean** | **0.7991** | **0.4093** | **0.1865** | **0.7266** | **0.3717** |
| **±Std** | 0.0743 | 0.1158 | 0.0215 | 0.0324 | 0.0435 |

CrossAttn best val AUC per fold: Fold1=0.8019, Fold2=0.6836, Fold3=0.7978, Fold4=0.7938, Fold5=0.9184

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7725 | 0.2913 | 0.1847 | 0.7294 | 0.3429 |
| CrossAttn | 0.8059 | 0.3199 | 0.1656 | 0.7686 | 0.4040 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7109 | 0.3121 | 0.2552 | 0.5859 | 0.3492 |
| F | 156 | 0.7831 | 0.3281 | 0.1400 | 0.8205 | 0.3333 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7561 | 0.3492 | 0.2051 | 0.7071 | 0.4727 |
| F | 156 | 0.8169 | 0.3294 | 0.1404 | 0.8077 | 0.3182 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 59 |
| **True: Sarco**  | 10 | 18 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 176 | 51 |
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
