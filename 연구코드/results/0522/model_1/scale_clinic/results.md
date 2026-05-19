# SMI Binary Classification — Results

Generated: 2026-05-19 13:25  |  5-Fold CV  |  Model 1 (Clinic Only, LR)

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 233 | 193 | 82.8% | 40 | 17.2% |
| Train | F | 575 | 533 | 92.7% | 42 | 7.3% |
| Train | **All** | **808** | **726** | **89.9%** | **82** | **10.1%** |
| Test | M | 68 | 53 | 77.9% | 15 | 22.1% |
| Test | F | 134 | 128 | 95.5% | 6 | 4.5% |
| Test | **All** | **202** | **181** | **89.6%** | **21** | **10.4%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 233 | 62.19 ± 11.47 | 28.00 | 63.00 | 89.00 |
| Train | F | 575 | 55.51 ± 11.30 | 29.00 | 55.00 | 91.00 |
| Train | **All** | **808** | **57.44 ± 11.75** | **28.00** | **57.00** | **91.00** |
| Test | M | 68 | 62.71 ± 10.68 | 29.00 | 62.00 | 81.00 |
| Test | F | 134 | 56.51 ± 12.51 | 24.00 | 58.00 | 86.00 |
| Test | **All** | **202** | **58.59 ± 12.28** | **24.00** | **60.00** | **86.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 233 | 23.96 ± 2.53 | 16.01 | 24.08 | 30.77 |
| Train | F | 575 | 23.10 ± 3.01 | 15.95 | 22.95 | 30.82 |
| Train | **All** | **808** | **23.34 ± 2.91** | **15.95** | **23.24** | **30.82** |
| Test | M | 68 | 23.94 ± 2.67 | 18.23 | 23.92 | 29.65 |
| Test | F | 134 | 23.30 ± 2.85 | 17.66 | 23.38 | 30.82 |
| Test | **All** | **202** | **23.51 ± 2.80** | **17.66** | **23.51** | **30.82** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8168 | 0.2829 | 0.1891 | 0.7346 | 0.3768 |
| 2 | 0.7501 | 0.2753 | 0.1909 | 0.7346 | 0.3582 |
| 3 | 0.7085 | 0.2596 | 0.2131 | 0.6420 | 0.2564 |
| 4 | 0.7409 | 0.2460 | 0.1944 | 0.6708 | 0.2740 |
| 5 | 0.8659 | 0.4718 | 0.1773 | 0.7453 | 0.4058 |
| **Mean** | **0.7765** | **0.3071** | **0.1930** | **0.7055** | **0.3342** |
| **±Std** | 0.0569 | 0.0833 | 0.0116 | 0.0413 | 0.0586 |

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8190 | 0.3730 | 0.1901 | 0.7030 | 0.3478 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 68 | 0.7862 | 0.4790 | 0.2569 | 0.6176 | 0.5000 |
| F | 134 | 0.7057 | 0.1095 | 0.1561 | 0.7463 | 0.1500 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 126 | 55 |
| **True: Sarco**  | 5 | 16 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `confusion_matrices.png` | Test-set confusion matrices (overall + by sex) |
| `test_roc_curves.png` | Final test-set ROC curve (overall) |
| `test_roc_by_sex.png` | Final test-set ROC curves split by sex |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
