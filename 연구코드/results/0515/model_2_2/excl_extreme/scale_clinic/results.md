# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-13 20:46  |  5-Fold CV  |  Median best epoch: 4

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 323 | 270 | 83.6% | 53 | 16.4% |
| Train | F | 592 | 545 | 92.1% | 47 | 7.9% |
| Train | **All** | **915** | **815** | **89.1%** | **100** | **10.9%** |
| Test | M | 90 | 74 | 82.2% | 16 | 17.8% |
| Test | F | 139 | 129 | 92.8% | 10 | 7.2% |
| Test | **All** | **229** | **203** | **88.6%** | **26** | **11.4%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 323 | 59.69 ± 12.79 | 18.00 | 60.00 | 88.00 |
| Train | F | 592 | 55.78 ± 11.79 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **915** | **57.16 ± 12.30** | **14.00** | **58.00** | **91.00** |
| Test | M | 90 | 61.49 ± 11.03 | 34.00 | 61.00 | 89.00 |
| Test | F | 139 | 55.97 ± 13.56 | 11.00 | 56.00 | 86.00 |
| Test | **All** | **229** | **58.14 ± 12.91** | **11.00** | **59.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 323 | 24.11 ± 3.23 | 14.48 | 24.16 | 36.76 |
| Train | F | 592 | 22.96 ± 3.16 | 15.62 | 22.69 | 34.61 |
| Train | **All** | **915** | **23.37 ± 3.23** | **14.48** | **23.23** | **36.76** |
| Test | M | 90 | 23.92 ± 3.19 | 16.80 | 24.05 | 33.87 |
| Test | F | 139 | 23.28 ± 3.55 | 14.40 | 22.89 | 36.24 |
| Test | **All** | **229** | **23.53 ± 3.43** | **14.40** | **23.53** | **36.24** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.5046 | 0.1130 | 0.2226 | 0.6831 | 0.1212 |
| 2 | 0.5988 | 0.1797 | 0.2145 | 0.7049 | 0.2500 |
| 3 | 0.5003 | 0.1098 | 0.2560 | 0.6393 | 0.1316 |
| 4 | 0.5267 | 0.1429 | 0.2499 | 0.6284 | 0.1707 |
| 5 | 0.5233 | 0.1237 | 0.2431 | 0.6503 | 0.1795 |
| **Mean** | **0.5307** | **0.1338** | **0.2372** | **0.6612** | **0.1706** |
| **±Std** | 0.0355 | 0.0257 | 0.0160 | 0.0285 | 0.0455 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7739 | 0.2550 | 0.1965 | 0.6721 | 0.3478 |
| 2 | 0.7282 | 0.3226 | 0.1915 | 0.6776 | 0.3059 |
| 3 | 0.8218 | 0.5260 | 0.1372 | 0.8087 | 0.3636 |
| 4 | 0.8310 | 0.4030 | 0.1601 | 0.7322 | 0.3288 |
| 5 | 0.7702 | 0.2809 | 0.2310 | 0.6230 | 0.3429 |
| **Mean** | **0.7850** | **0.3575** | **0.1833** | **0.7027** | **0.3378** |
| **±Std** | 0.0375 | 0.0980 | 0.0322 | 0.0633 | 0.0195 |

CrossAttn best val AUC per fold: Fold1=0.7739, Fold2=0.7282, Fold3=0.8218, Fold4=0.8310, Fold5=0.7702

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6357 | 0.2034 | 0.2203 | 0.7074 | 0.3093 |
| CrossAttn | 0.8213 | 0.3113 | 0.1838 | 0.7249 | 0.4112 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 90 | 0.4890 | 0.2322 | 0.2898 | 0.5667 | 0.2642 |
| F | 139 | 0.7938 | 0.2106 | 0.1753 | 0.7986 | 0.3636 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 90 | 0.7416 | 0.3050 | 0.2352 | 0.6556 | 0.4561 |
| F | 139 | 0.8566 | 0.4697 | 0.1504 | 0.7698 | 0.3600 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 147 | 56 |
| **True: Sarco**  | 11 | 15 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 144 | 59 |
| **True: Sarco**  | 4 | 22 |

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
