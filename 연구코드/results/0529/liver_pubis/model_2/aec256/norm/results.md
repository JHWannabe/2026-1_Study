# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-25 18:19  |  5-Fold CV  |  Median best epoch: 7

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 335 | 283 | 84.5% | 52 | 15.5% |
| Train | F | 595 | 548 | 92.1% | 47 | 7.9% |
| Train | **All** | **930** | **831** | **89.4%** | **99** | **10.6%** |
| Test | M | 85 | 71 | 83.5% | 14 | 16.5% |
| Test | F | 148 | 137 | 92.6% | 11 | 7.4% |
| Test | **All** | **233** | **208** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 335 | 59.78 ± 12.13 | 20.00 | 60.00 | 89.00 |
| Train | F | 595 | 55.26 ± 11.90 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **930** | **56.89 ± 12.18** | **14.00** | **57.00** | **91.00** |
| Test | M | 85 | 59.08 ± 12.34 | 29.00 | 60.00 | 84.00 |
| Test | F | 148 | 55.47 ± 11.64 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **233** | **56.79 ± 12.03** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 335 | 24.39 ± 2.93 | 17.33 | 24.24 | 32.67 |
| Train | F | 595 | 23.15 ± 3.17 | 16.00 | 22.96 | 34.61 |
| Train | **All** | **930** | **23.60 ± 3.15** | **16.00** | **23.45** | **34.61** |
| Test | M | 85 | 24.34 ± 3.40 | 14.34 | 24.26 | 32.56 |
| Test | F | 148 | 23.06 ± 3.53 | 12.02 | 22.64 | 32.48 |
| Test | **All** | **233** | **23.52 ± 3.53** | **12.02** | **23.41** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8120 | 0.3312 | 0.1608 | 0.7849 | 0.4286 |
| 2 | 0.8151 | 0.4026 | 0.1510 | 0.6344 | 0.3462 |
| 3 | 0.8416 | 0.3836 | 0.1408 | 0.7742 | 0.4615 |
| 4 | 0.8470 | 0.3868 | 0.1724 | 0.6720 | 0.3838 |
| 5 | 0.7759 | 0.3161 | 0.1762 | 0.6075 | 0.3178 |
| **Mean** | **0.8183** | **0.3640** | **0.1602** | **0.6946** | **0.3876** |
| **±Std** | 0.0253 | 0.0340 | 0.0131 | 0.0724 | 0.0525 |

CrossAttn best val AUC per fold: Fold1=0.8120, Fold2=0.8151, Fold3=0.8416, Fold4=0.8470, Fold5=0.7759

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8269 | 0.4151 | 0.1966 | 0.5193 | 0.3000 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 85 | 0.8078 | 0.4985 | 0.2099 | 0.5059 | 0.3824 |
| F | 148 | 0.8003 | 0.3101 | 0.1890 | 0.5270 | 0.2391 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 97 | 111 |
| **True: Sarco**  | 1 | 24 |

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
