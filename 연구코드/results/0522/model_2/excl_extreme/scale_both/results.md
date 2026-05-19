# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-19 13:08  |  5-Fold CV  |  Median best epoch: 7

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 214 | 179 | 83.6% | 35 | 16.4% |
| Train | F | 527 | 488 | 92.6% | 39 | 7.4% |
| Train | **All** | **741** | **667** | **90.0%** | **74** | **10.0%** |
| Test | M | 55 | 46 | 83.6% | 9 | 16.4% |
| Test | F | 130 | 121 | 93.1% | 9 | 6.9% |
| Test | **All** | **185** | **167** | **90.3%** | **18** | **9.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 214 | 60.95 ± 11.58 | 28.00 | 62.00 | 89.00 |
| Train | F | 527 | 55.22 ± 11.46 | 24.00 | 55.00 | 87.00 |
| Train | **All** | **741** | **56.88 ± 11.78** | **24.00** | **57.00** | **89.00** |
| Test | M | 55 | 63.82 ± 11.36 | 31.00 | 66.00 | 83.00 |
| Test | F | 130 | 57.32 ± 11.54 | 29.00 | 58.00 | 91.00 |
| Test | **All** | **185** | **59.25 ± 11.86** | **29.00** | **59.00** | **91.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 214 | 24.04 ± 2.74 | 15.22 | 24.33 | 31.94 |
| Train | F | 527 | 23.07 ± 2.94 | 15.63 | 22.92 | 31.93 |
| Train | **All** | **741** | **23.35 ± 2.92** | **15.22** | **23.31** | **31.94** |
| Test | M | 55 | 23.35 ± 2.07 | 19.29 | 23.28 | 27.62 |
| Test | F | 130 | 23.09 ± 2.70 | 16.80 | 22.98 | 30.05 |
| Test | **All** | **185** | **23.17 ± 2.53** | **16.80** | **23.02** | **30.05** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8269 | 0.3541 | 0.2907 | 0.4295 | 0.2478 |
| 2 | 0.7814 | 0.2952 | 0.1973 | 0.7230 | 0.3279 |
| 3 | 0.8386 | 0.4253 | 0.1507 | 0.7703 | 0.4138 |
| 4 | 0.8436 | 0.3102 | 0.1891 | 0.6689 | 0.3467 |
| 5 | 0.8852 | 0.5610 | 0.1394 | 0.8041 | 0.4727 |
| **Mean** | **0.8351** | **0.3892** | **0.1934** | **0.6791** | **0.3618** |
| **±Std** | 0.0333 | 0.0971 | 0.0534 | 0.1328 | 0.0767 |

CrossAttn best val AUC per fold: Fold1=0.8269, Fold2=0.7814, Fold3=0.8386, Fold4=0.8436, Fold5=0.8852

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7242 | 0.2936 | 0.2012 | 0.6757 | 0.2857 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 55 | 0.6739 | 0.4283 | 0.3062 | 0.5091 | 0.3415 |
| F | 130 | 0.7208 | 0.1861 | 0.1568 | 0.7462 | 0.2326 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 113 | 54 |
| **True: Sarco**  | 6 | 12 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (CrossAttn) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curve |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
