# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-12 16:44  |  5-Fold CV  |  Median best epoch: 42

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
| 1 | 0.6350 | 0.2601 | 0.1896 | 0.7548 | 0.2609 |
| 2 | 0.7452 | 0.2260 | 0.1925 | 0.7404 | 0.3864 |
| 3 | 0.7058 | 0.2919 | 0.1691 | 0.7885 | 0.2667 |
| 4 | 0.5431 | 0.1507 | 0.2444 | 0.6587 | 0.2022 |
| 5 | 0.7177 | 0.2781 | 0.1782 | 0.7391 | 0.3077 |
| **Mean** | **0.6694** | **0.2414** | **0.1948** | **0.7363** | **0.2848** |
| **±Std** | 0.0728 | 0.0504 | 0.0262 | 0.0427 | 0.0609 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7986 | 0.3567 | 0.1268 | 0.8269 | 0.3793 |
| 2 | 0.8021 | 0.3632 | 0.1994 | 0.5962 | 0.3115 |
| 3 | 0.8195 | 0.4050 | 0.1189 | 0.8221 | 0.4127 |
| 4 | 0.7727 | 0.2650 | 0.2901 | 0.5337 | 0.3022 |
| 5 | 0.8025 | 0.3577 | 0.2365 | 0.6570 | 0.3364 |
| **Mean** | **0.7991** | **0.3496** | **0.1944** | **0.6872** | **0.3484** |
| **±Std** | 0.0150 | 0.0459 | 0.0651 | 0.1187 | 0.0418 |

CrossAttn best val AUC per fold: Fold1=0.7986, Fold2=0.8021, Fold3=0.8195, Fold4=0.7727, Fold5=0.8025

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6713 | 0.1958 | 0.1956 | 0.7154 | 0.2745 |
| CrossAttn | 0.7542 | 0.2410 | 0.1717 | 0.7423 | 0.2947 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.6707 | 0.2723 | 0.2233 | 0.6939 | 0.3750 |
| F | 162 | 0.6539 | 0.1399 | 0.1788 | 0.7284 | 0.1852 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 98 | 0.6890 | 0.2373 | 0.2437 | 0.6327 | 0.3571 |
| F | 162 | 0.7428 | 0.2432 | 0.1281 | 0.8086 | 0.2051 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 172 | 60 |
| **True: Sarco**  | 14 | 14 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 179 | 53 |
| **True: Sarco**  | 14 | 14 |

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
