# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 16:58  |  5-Fold CV  |  Median best epoch: 54

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
| 1 | 0.7260 | 0.2774 | 0.1690 | 0.7644 | 0.3636 |
| 2 | 0.8407 | 0.3619 | 0.1895 | 0.7115 | 0.3750 |
| 3 | 0.7553 | 0.3548 | 0.1613 | 0.7596 | 0.3243 |
| 4 | 0.7295 | 0.2469 | 0.1978 | 0.7019 | 0.3404 |
| 5 | 0.8032 | 0.3189 | 0.1709 | 0.7391 | 0.3571 |
| **Mean** | **0.7709** | **0.3120** | **0.1777** | **0.7353** | **0.3521** |
| **±Std** | 0.0445 | 0.0443 | 0.0137 | 0.0250 | 0.0178 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8230 | 0.4558 | 0.1129 | 0.8413 | 0.5075 |
| 2 | 0.8169 | 0.4964 | 0.1288 | 0.8606 | 0.5085 |
| 3 | 0.8148 | 0.4225 | 0.1473 | 0.7740 | 0.4051 |
| 4 | 0.7871 | 0.3329 | 0.3134 | 0.4471 | 0.2675 |
| 5 | 0.8597 | 0.3677 | 0.1711 | 0.7150 | 0.4040 |
| **Mean** | **0.8203** | **0.4150** | **0.1747** | **0.7276** | **0.4185** |
| **±Std** | 0.0233 | 0.0589 | 0.0720 | 0.1494 | 0.0885 |

CrossAttn best val AUC per fold: Fold1=0.8230, Fold2=0.8169, Fold3=0.8148, Fold4=0.7871, Fold5=0.8597

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7608 | 0.2516 | 0.1888 | 0.7115 | 0.3119 |
| CrossAttn | 0.7606 | 0.2351 | 0.2005 | 0.6846 | 0.3387 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.7470 | 0.3155 | 0.2487 | 0.6122 | 0.3871 |
| F | 162 | 0.7117 | 0.1606 | 0.1525 | 0.7716 | 0.2128 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.6791 | 0.2436 | 0.2822 | 0.5918 | 0.3939 |
| F | 162 | 0.7583 | 0.2948 | 0.1511 | 0.7407 | 0.2759 |

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
| **True: Normal** | 157 | 75 |
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
