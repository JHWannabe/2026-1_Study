# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 17:16  |  5-Fold CV  |  Median best epoch: 8

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
| 1 | 0.4600 | 0.1365 | 0.2192 | 0.6961 | 0.1143 |
| 2 | 0.6426 | 0.1823 | 0.2079 | 0.6912 | 0.2222 |
| 3 | 0.6339 | 0.1606 | 0.2057 | 0.6798 | 0.2353 |
| 4 | 0.6517 | 0.2961 | 0.2129 | 0.6995 | 0.2651 |
| 5 | 0.6816 | 0.1993 | 0.2002 | 0.7241 | 0.2821 |
| **Mean** | **0.6139** | **0.1949** | **0.2092** | **0.6981** | **0.2238** |
| **±Std** | 0.0786 | 0.0548 | 0.0065 | 0.0146 | 0.0587 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7955 | 0.4194 | 0.1921 | 0.7598 | 0.3951 |
| 2 | 0.6829 | 0.2675 | 0.2013 | 0.6912 | 0.3077 |
| 3 | 0.7627 | 0.2541 | 0.1742 | 0.7438 | 0.3810 |
| 4 | 0.8202 | 0.4402 | 0.1049 | 0.8966 | 0.4000 |
| 5 | 0.9043 | 0.5694 | 0.1272 | 0.8079 | 0.5063 |
| **Mean** | **0.7931** | **0.3901** | **0.1600** | **0.7799** | **0.3980** |
| **±Std** | 0.0724 | 0.1175 | 0.0375 | 0.0692 | 0.0636 |

CrossAttn best val AUC per fold: Fold1=0.7955, Fold2=0.6829, Fold3=0.7627, Fold4=0.8202, Fold5=0.9043

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6495 | 0.1865 | 0.2118 | 0.6863 | 0.2593 |
| CrossAttn | 0.8406 | 0.3644 | 0.1311 | 0.8235 | 0.4304 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.6248 | 0.2545 | 0.2505 | 0.6364 | 0.3077 |
| F | 156 | 0.6671 | 0.1464 | 0.1873 | 0.7179 | 0.2143 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7841 | 0.4119 | 0.1865 | 0.7273 | 0.4255 |
| F | 156 | 0.8527 | 0.3375 | 0.0959 | 0.8846 | 0.4375 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 161 | 66 |
| **True: Sarco**  | 14 | 14 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 193 | 34 |
| **True: Sarco**  | 11 | 17 |

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
