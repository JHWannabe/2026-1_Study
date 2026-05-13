# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 17:33  |  5-Fold CV  |  Median best epoch: 11

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
| 1 | 0.6933 | 0.2861 | 0.1919 | 0.7108 | 0.2532 |
| 2 | 0.6483 | 0.2333 | 0.2032 | 0.6912 | 0.2759 |
| 3 | 0.5819 | 0.1665 | 0.2434 | 0.6453 | 0.2000 |
| 4 | 0.5522 | 0.1623 | 0.2061 | 0.7143 | 0.2368 |
| 5 | 0.6848 | 0.2269 | 0.2094 | 0.7192 | 0.3133 |
| **Mean** | **0.6321** | **0.2150** | **0.2108** | **0.6962** | **0.2558** |
| **±Std** | 0.0560 | 0.0462 | 0.0173 | 0.0271 | 0.0379 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8179 | 0.3323 | 0.2006 | 0.6912 | 0.3636 |
| 2 | 0.7057 | 0.2334 | 0.2147 | 0.6863 | 0.2889 |
| 3 | 0.7941 | 0.2759 | 0.1851 | 0.7192 | 0.3596 |
| 4 | 0.7878 | 0.4118 | 0.1815 | 0.7340 | 0.3864 |
| 5 | 0.8983 | 0.5964 | 0.1585 | 0.7389 | 0.4421 |
| **Mean** | **0.8008** | **0.3700** | **0.1881** | **0.7139** | **0.3681** |
| **±Std** | 0.0617 | 0.1280 | 0.0189 | 0.0216 | 0.0494 |

CrossAttn best val AUC per fold: Fold1=0.8179, Fold2=0.7057, Fold3=0.7941, Fold4=0.7878, Fold5=0.8983

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6768 | 0.2155 | 0.2199 | 0.6667 | 0.2609 |
| CrossAttn | 0.7955 | 0.3116 | 0.1758 | 0.7412 | 0.3889 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.5940 | 0.2087 | 0.2605 | 0.5758 | 0.2759 |
| F | 156 | 0.7618 | 0.2890 | 0.1941 | 0.7244 | 0.2456 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 99 | 0.7611 | 0.3582 | 0.2211 | 0.6667 | 0.4407 |
| F | 156 | 0.7875 | 0.2683 | 0.1470 | 0.7885 | 0.3265 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 155 | 72 |
| **True: Sarco**  | 13 | 15 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 168 | 59 |
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
