# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 18:19  |  5-Fold CV  |  Median best epoch: 29

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
| 1 | 0.8639 | 0.5064 | 0.1578 | 0.7826 | 0.4828 |
| 2 | 0.7482 | 0.3270 | 0.2005 | 0.6957 | 0.3824 |
| 3 | 0.8156 | 0.3045 | 0.2302 | 0.6350 | 0.3421 |
| 4 | 0.9024 | 0.4894 | 0.1220 | 0.8394 | 0.5417 |
| 5 | 0.8518 | 0.5268 | 0.1653 | 0.7372 | 0.4194 |
| **Mean** | **0.8364** | **0.4308** | **0.1752** | **0.7380** | **0.4336** |
| **±Std** | 0.0521 | 0.0949 | 0.0371 | 0.0702 | 0.0711 |

CrossAttn best val AUC per fold: Fold1=0.8639, Fold2=0.7482, Fold3=0.8156, Fold4=0.9024, Fold5=0.8518

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7925 | 0.2624 | 0.1825 | 0.7151 | 0.3636 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 74 | 0.8390 | 0.2852 | 0.2520 | 0.6081 | 0.3556 |
| F | 98 | 0.8552 | 0.3600 | 0.1300 | 0.7959 | 0.3750 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 109 | 42 |
| **True: Sarco**  | 7 | 14 |

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
