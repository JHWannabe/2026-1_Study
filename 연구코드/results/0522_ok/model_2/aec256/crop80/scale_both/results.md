# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 18:27  |  5-Fold CV  |  Median best epoch: 36

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

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8177 | 0.4865 | 0.1599 | 0.7681 | 0.4074 |
| 2 | 0.7589 | 0.3623 | 0.1749 | 0.7536 | 0.4138 |
| 3 | 0.8394 | 0.3932 | 0.1946 | 0.6423 | 0.3636 |
| 4 | 0.8982 | 0.4575 | 0.1413 | 0.7883 | 0.4912 |
| 5 | 0.8611 | 0.5906 | 0.1783 | 0.7226 | 0.4062 |
| **Mean** | **0.8350** | **0.4580** | **0.1698** | **0.7350** | **0.4165** |
| **±Std** | 0.0465 | 0.0797 | 0.0180 | 0.0510 | 0.0414 |

CrossAttn best val AUC per fold: Fold1=0.8177, Fold2=0.7589, Fold3=0.8394, Fold4=0.8982, Fold5=0.8611

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7799 | 0.4636 | 0.2100 | 0.7326 | 0.3947 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 74 | 0.8295 | 0.4829 | 0.2657 | 0.6622 | 0.3902 |
| F | 98 | 0.7756 | 0.4821 | 0.1679 | 0.7857 | 0.4000 |

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
| `cv_roc_curves.png` | Per-fold ROC curves (CrossAttn) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curve |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
