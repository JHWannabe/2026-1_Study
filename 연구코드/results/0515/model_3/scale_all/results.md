# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 17:19  |  5-Fold CV  |  Median best epoch: 13

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
| 1 | 0.7187 | 0.2731 | 0.1699 | 0.7596 | 0.3590 |
| 2 | 0.8425 | 0.3649 | 0.1892 | 0.7115 | 0.3750 |
| 3 | 0.7584 | 0.3353 | 0.1613 | 0.7596 | 0.3243 |
| 4 | 0.7314 | 0.2507 | 0.1974 | 0.7019 | 0.3542 |
| 5 | 0.7939 | 0.2862 | 0.1764 | 0.7246 | 0.3448 |
| **Mean** | **0.7690** | **0.3020** | **0.1788** | **0.7315** | **0.3515** |
| **±Std** | 0.0449 | 0.0419 | 0.0130 | 0.0241 | 0.0167 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8223 | 0.3670 | 0.1410 | 0.7933 | 0.4110 |
| 2 | 0.8999 | 0.5538 | 0.1529 | 0.7933 | 0.4691 |
| 3 | 0.8150 | 0.4247 | 0.1244 | 0.8558 | 0.5000 |
| 4 | 0.7469 | 0.3758 | 0.2221 | 0.6442 | 0.3019 |
| 5 | 0.7870 | 0.3152 | 0.2054 | 0.6329 | 0.3091 |
| **Mean** | **0.8142** | **0.4073** | **0.1691** | **0.7439** | **0.3982** |
| **±Std** | 0.0504 | 0.0811 | 0.0379 | 0.0891 | 0.0810 |

CrossAttn best val AUC per fold: Fold1=0.8223, Fold2=0.8999, Fold3=0.8150, Fold4=0.7469, Fold5=0.7870

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7599 | 0.2574 | 0.1884 | 0.7115 | 0.3119 |
| CrossAttn | 0.7263 | 0.2262 | 0.1654 | 0.7654 | 0.2651 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.7477 | 0.3256 | 0.2496 | 0.6122 | 0.3871 |
| F | 162 | 0.7111 | 0.1610 | 0.1514 | 0.7716 | 0.2128 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.6715 | 0.2476 | 0.2534 | 0.5918 | 0.3103 |
| F | 162 | 0.6928 | 0.2359 | 0.1122 | 0.8704 | 0.1600 |

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
| **True: Normal** | 188 | 44 |
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
