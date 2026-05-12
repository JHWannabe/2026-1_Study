# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 16:44  |  5-Fold CV  |  Median best epoch: 78

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 402 | 332 | 82.6% | 70 | 17.4% |
| Train | F | 695 | 645 | 92.8% | 50 | 7.2% |
| Train | **All** | **1097** | **977** | **89.1%** | **120** | **10.9%** |
| Test | M | 112 | 95 | 84.8% | 17 | 15.2% |
| Test | F | 163 | 150 | 92.0% | 13 | 8.0% |
| Test | **All** | **275** | **245** | **89.1%** | **30** | **10.9%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 402 | 59.81 ± 12.51 | 18.00 | 60.00 | 89.00 |
| Train | F | 695 | 55.36 ± 12.15 | 11.00 | 55.00 | 91.00 |
| Train | **All** | **1097** | **56.99 ± 12.47** | **11.00** | **58.00** | **91.00** |
| Test | M | 112 | 59.05 ± 12.52 | 23.00 | 59.50 | 84.00 |
| Test | F | 163 | 56.52 ± 12.29 | 22.00 | 56.00 | 87.00 |
| Test | **All** | **275** | **57.55 ± 12.45** | **22.00** | **58.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 402 | 24.22 ± 3.26 | 14.48 | 24.19 | 36.76 |
| Train | F | 695 | 23.09 ± 3.43 | 14.40 | 22.70 | 39.49 |
| Train | **All** | **1097** | **23.51 ± 3.41** | **14.40** | **23.30** | **39.49** |
| Test | M | 112 | 24.07 ± 3.30 | 16.44 | 24.16 | 35.20 |
| Test | F | 163 | 22.99 ± 3.19 | 16.06 | 22.83 | 34.23 |
| Test | **All** | **275** | **23.43 ± 3.28** | **16.06** | **23.44** | **35.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.5512 | 0.1463 | 0.2465 | 0.6636 | 0.2128 |
| 2 | 0.6482 | 0.1913 | 0.1779 | 0.7591 | 0.2740 |
| 3 | 0.6662 | 0.2269 | 0.1950 | 0.7169 | 0.2619 |
| 4 | 0.6397 | 0.1946 | 0.1923 | 0.7306 | 0.2532 |
| 5 | 0.6316 | 0.1876 | 0.2120 | 0.7169 | 0.2439 |
| **Mean** | **0.6274** | **0.1893** | **0.2047** | **0.7174** | **0.2491** |
| **±Std** | 0.0398 | 0.0257 | 0.0235 | 0.0310 | 0.0207 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7651 | 0.3361 | 0.1945 | 0.7000 | 0.3400 |
| 2 | 0.8253 | 0.4046 | 0.1680 | 0.7818 | 0.3846 |
| 3 | 0.7692 | 0.3508 | 0.2285 | 0.6484 | 0.3186 |
| 4 | 0.6979 | 0.2423 | 0.2551 | 0.5799 | 0.2698 |
| 5 | 0.8825 | 0.5270 | 0.1503 | 0.7580 | 0.4301 |
| **Mean** | **0.7880** | **0.3721** | **0.1993** | **0.6936** | **0.3486** |
| **±Std** | 0.0622 | 0.0934 | 0.0384 | 0.0734 | 0.0550 |

CrossAttn best val AUC per fold: Fold1=0.7651, Fold2=0.8253, Fold3=0.7692, Fold4=0.6979, Fold5=0.8825

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6133 | 0.2317 | 0.2039 | 0.6873 | 0.2321 |
| CrossAttn | 0.7071 | 0.2936 | 0.1801 | 0.7200 | 0.3304 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.6495 | 0.3322 | 0.2387 | 0.6429 | 0.3103 |
| F | 163 | 0.5477 | 0.1627 | 0.1801 | 0.7178 | 0.1481 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.6774 | 0.3597 | 0.2495 | 0.5893 | 0.3429 |
| F | 163 | 0.7092 | 0.2638 | 0.1323 | 0.8098 | 0.3111 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 176 | 69 |
| **True: Sarco**  | 17 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 179 | 66 |
| **True: Sarco**  | 11 | 19 |

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
