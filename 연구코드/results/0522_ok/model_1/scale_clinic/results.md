# SMI Binary Classification — Results

Generated: 2026-05-20 20:00  |  5-Fold CV  |  Model 1 (Clinic Only, LR)

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 282 | 235 | 83.3% | 47 | 16.7% |
| Train | F | 405 | 370 | 91.4% | 35 | 8.6% |
| Train | **All** | **687** | **605** | **88.1%** | **82** | **11.9%** |
| Test | M | 74 | 66 | 89.2% | 8 | 10.8% |
| Test | F | 98 | 85 | 86.7% | 13 | 13.3% |
| Test | **All** | **172** | **151** | **87.8%** | **21** | **12.2%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 282 | 59.14 ± 11.64 | 23.00 | 59.00 | 89.00 |
| Train | F | 405 | 55.71 ± 11.75 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **687** | **57.12 ± 11.82** | **14.00** | **58.00** | **91.00** |
| Test | M | 74 | 60.66 ± 12.38 | 29.00 | 61.50 | 83.00 |
| Test | F | 98 | 58.45 ± 13.90 | 18.00 | 59.00 | 87.00 |
| Test | **All** | **172** | **59.40 ± 13.31** | **18.00** | **60.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 282 | 24.34 ± 3.06 | 14.34 | 24.22 | 32.67 |
| Train | F | 405 | 22.83 ± 3.30 | 12.02 | 22.66 | 34.20 |
| Train | **All** | **687** | **23.45 ± 3.29** | **12.02** | **23.33** | **34.20** |
| Test | M | 74 | 24.61 ± 2.84 | 17.51 | 24.43 | 32.56 |
| Test | F | 98 | 23.55 ± 3.36 | 16.92 | 23.23 | 34.61 |
| Test | **All** | **172** | **24.01 ± 3.19** | **16.92** | **23.62** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8532 | 0.5717 | 0.1512 | 0.7971 | 0.5000 |
| 2 | 0.7144 | 0.3461 | 0.1880 | 0.6957 | 0.3226 |
| 3 | 0.7800 | 0.2536 | 0.1952 | 0.7153 | 0.3607 |
| 4 | 0.8533 | 0.4983 | 0.1650 | 0.7518 | 0.4138 |
| 5 | 0.8352 | 0.5264 | 0.1912 | 0.6715 | 0.3284 |
| **Mean** | **0.8072** | **0.4392** | **0.1781** | **0.7263** | **0.3851** |
| **±Std** | 0.0536 | 0.1198 | 0.0171 | 0.0441 | 0.0660 |

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8187 | 0.2970 | 0.1763 | 0.7326 | 0.3947 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 74 | 0.8333 | 0.2694 | 0.2405 | 0.6351 | 0.3721 |
| F | 98 | 0.8570 | 0.4462 | 0.1278 | 0.8061 | 0.4242 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 111 | 40 |
| **True: Sarco**  | 6 | 15 |

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
