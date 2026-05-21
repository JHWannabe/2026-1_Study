# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 19:59  |  5-Fold CV  |  Median best epoch: 2

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 334 | 282 | 84.4% | 52 | 15.6% |
| Train | F | 595 | 548 | 92.1% | 47 | 7.9% |
| Train | **All** | **929** | **830** | **89.3%** | **99** | **10.7%** |
| Test | M | 85 | 71 | 83.5% | 14 | 16.5% |
| Test | F | 148 | 137 | 92.6% | 11 | 7.4% |
| Test | **All** | **233** | **208** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 59.72 ± 12.11 | 20.00 | 60.00 | 89.00 |
| Train | F | 595 | 55.26 ± 11.90 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **929** | **56.86 ± 12.16** | **14.00** | **57.00** | **91.00** |
| Test | M | 85 | 59.26 ± 12.48 | 29.00 | 60.00 | 84.00 |
| Test | F | 148 | 55.47 ± 11.64 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **233** | **56.85 ± 12.09** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 24.38 ± 2.93 | 17.33 | 24.24 | 32.67 |
| Train | F | 595 | 23.15 ± 3.17 | 16.00 | 22.96 | 34.61 |
| Train | **All** | **929** | **23.59 ± 3.14** | **16.00** | **23.44** | **34.61** |
| Test | M | 85 | 24.33 ± 3.39 | 14.34 | 24.24 | 32.56 |
| Test | F | 148 | 23.06 ± 3.53 | 12.02 | 22.64 | 32.48 |
| Test | **All** | **233** | **23.52 ± 3.53** | **12.02** | **23.41** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8690 | 0.5372 | 0.1468 | 0.8011 | 0.4638 |
| 2 | 0.8066 | 0.4357 | 0.1968 | 0.7473 | 0.4198 |
| 3 | 0.8066 | 0.3349 | 0.2469 | 0.5860 | 0.3186 |
| 4 | 0.7732 | 0.3047 | 0.2628 | 0.5538 | 0.3025 |
| 5 | 0.8044 | 0.3563 | 0.2002 | 0.6919 | 0.3294 |
| **Mean** | **0.8120** | **0.3938** | **0.2107** | **0.6760** | **0.3668** |
| **±Std** | 0.0312 | 0.0838 | 0.0410 | 0.0938 | 0.0633 |

CrossAttn best val AUC per fold: Fold1=0.8690, Fold2=0.8066, Fold3=0.8066, Fold4=0.7732, Fold5=0.8044

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8250 | 0.4441 | 0.2295 | 0.6052 | 0.3235 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 85 | 0.8370 | 0.5662 | 0.2561 | 0.6118 | 0.4407 |
| F | 148 | 0.7823 | 0.2905 | 0.2143 | 0.6014 | 0.2338 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 119 | 89 |
| **True: Sarco**  | 3 | 22 |

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
