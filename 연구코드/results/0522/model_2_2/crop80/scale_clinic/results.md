# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:07  |  5-Fold CV  |  Median best epoch: 5

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 355 | 292 | 82.3% | 63 | 17.7% |
| Train | F | 661 | 614 | 92.9% | 47 | 7.1% |
| Train | **All** | **1016** | **906** | **89.2%** | **110** | **10.8%** |
| Test | M | 97 | 83 | 85.6% | 14 | 14.4% |
| Test | F | 158 | 145 | 91.8% | 13 | 8.2% |
| Test | **All** | **255** | **228** | **89.4%** | **27** | **10.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 355 | 59.92 ± 12.67 | 18.00 | 60.00 | 89.00 |
| Train | F | 661 | 55.55 ± 11.94 | 18.00 | 55.00 | 91.00 |
| Train | **All** | **1016** | **57.07 ± 12.38** | **18.00** | **57.00** | **91.00** |
| Test | M | 97 | 58.63 ± 12.43 | 28.00 | 59.00 | 88.00 |
| Test | F | 158 | 55.27 ± 11.46 | 23.00 | 56.00 | 86.00 |
| Test | **All** | **255** | **56.55 ± 11.95** | **23.00** | **57.00** | **88.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 355 | 24.22 ± 3.38 | 14.48 | 24.16 | 36.76 |
| Train | F | 661 | 23.14 ± 3.39 | 14.40 | 22.83 | 36.24 |
| Train | **All** | **1016** | **23.52 ± 3.42** | **14.40** | **23.37** | **36.76** |
| Test | M | 97 | 24.50 ± 3.14 | 18.37 | 24.49 | 35.68 |
| Test | F | 158 | 23.11 ± 3.24 | 16.87 | 22.72 | 34.23 |
| Test | **All** | **255** | **23.64 ± 3.27** | **16.87** | **23.34** | **35.68** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.5197 | 0.1231 | 0.2393 | 0.6814 | 0.1772 |
| 2 | 0.6781 | 0.1869 | 0.1932 | 0.7438 | 0.2973 |
| 3 | 0.6183 | 0.1982 | 0.2074 | 0.7192 | 0.1972 |
| 4 | 0.6986 | 0.3205 | 0.1956 | 0.7241 | 0.3488 |
| 5 | 0.6389 | 0.1602 | 0.2229 | 0.6502 | 0.2198 |
| **Mean** | **0.6307** | **0.1978** | **0.2117** | **0.7038** | **0.2481** |
| **±Std** | 0.0623 | 0.0666 | 0.0174 | 0.0335 | 0.0648 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7862 | 0.3938 | 0.1530 | 0.7843 | 0.4054 |
| 2 | 0.8938 | 0.6234 | 0.1186 | 0.8128 | 0.4722 |
| 3 | 0.8096 | 0.3397 | 0.1887 | 0.7291 | 0.3678 |
| 4 | 0.8672 | 0.5889 | 0.1870 | 0.7438 | 0.4222 |
| 5 | 0.8365 | 0.3459 | 0.1990 | 0.6847 | 0.3725 |
| **Mean** | **0.8387** | **0.4583** | **0.1693** | **0.7510** | **0.4080** |
| **±Std** | 0.0386 | 0.1226 | 0.0297 | 0.0444 | 0.0380 |

CrossAttn best val AUC per fold: Fold1=0.7862, Fold2=0.8938, Fold3=0.8096, Fold4=0.8672, Fold5=0.8365

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.5551 | 0.1493 | 0.2111 | 0.6824 | 0.1980 |
| CrossAttn | 0.7966 | 0.2600 | 0.2273 | 0.6706 | 0.3333 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6231 | 0.2332 | 0.1995 | 0.6907 | 0.2857 |
| F | 158 | 0.5029 | 0.1230 | 0.2182 | 0.6772 | 0.1356 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7917 | 0.3117 | 0.2791 | 0.6082 | 0.3871 |
| F | 158 | 0.7798 | 0.2083 | 0.1954 | 0.7089 | 0.2812 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 164 | 64 |
| **True: Sarco**  | 17 | 10 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 150 | 78 |
| **True: Sarco**  | 6 | 21 |

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
