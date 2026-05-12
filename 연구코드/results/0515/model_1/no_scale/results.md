# SMI Binary Classification — Results

Generated: 2026-05-12 16:40  |  5-Fold CV  |  ResNet1D median best epoch: 54

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
| 1 | 0.8329 | 0.3651 | 0.1963 | 0.6864 | 0.3670 |
| 2 | 0.8095 | 0.3370 | 0.1612 | 0.7591 | 0.3765 |
| 3 | 0.7784 | 0.3403 | 0.2040 | 0.6804 | 0.3137 |
| 4 | 0.7233 | 0.2329 | 0.1830 | 0.7306 | 0.3059 |
| 5 | 0.8500 | 0.5234 | 0.1726 | 0.7489 | 0.3956 |
| **Mean** | **0.7988** | **0.3598** | **0.1834** | **0.7211** | **0.3517** |
| **±Std** | 0.0448 | 0.0936 | 0.0155 | 0.0322 | 0.0355 |

### ResNet1D

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8703 | 0.4683 | 0.2194 | 0.6409 | 0.3577 |
| 2 | 0.8444 | 0.3939 | 0.3022 | 0.4955 | 0.2930 |
| 3 | 0.7806 | 0.3370 | 0.0961 | 0.8584 | 0.1622 |
| 4 | 0.8415 | 0.4100 | 0.1840 | 0.8265 | 0.4062 |
| 5 | 0.8517 | 0.3597 | 0.1172 | 0.8265 | 0.4412 |
| **Mean** | **0.8377** | **0.3938** | **0.1838** | **0.7296** | **0.3321** |
| **±Std** | 0.0303 | 0.0452 | 0.0740 | 0.1400 | 0.0984 |

ResNet1D best val AUC per fold: Fold1=0.8703, Fold2=0.8444, Fold3=0.7806, Fold4=0.8415, Fold5=0.8517

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.7527 | 0.3338 | 0.1992 | 0.7091 | 0.3333 |
| ResNet1D  | 0.7578 | 0.3240 | 0.3256 | 0.4691 | 0.2700 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7245 | 0.4141 | 0.2790 | 0.5357 | 0.3333 |
| F | 163 | 0.7682 | 0.2498 | 0.1444 | 0.8282 | 0.3333 |

#### ResNet1D

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 112 | 0.7152 | 0.3669 | 0.3875 | 0.3839 | 0.3030 |
| F | 163 | 0.7985 | 0.3133 | 0.2830 | 0.5276 | 0.2376 |

---

## 3. Confusion Matrices (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 175 | 70 |
| **True: Sarco**  | 10 | 20 |

### ResNet1D

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 102 | 143 |
| **True: Sarco**  | 3 | 27 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR & ResNet1D) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `confusion_matrices.png` | Test-set confusion matrices (overall) |
| `confusion_matrices_by_sex.png` | Test-set confusion matrices split by sex |
| `training_curves.png` | ResNet1D loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curves (overall) |
| `test_roc_by_sex.png` | Final test-set ROC curves split by sex |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
