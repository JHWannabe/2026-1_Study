# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 22:04  |  5-Fold CV  |  Median best epoch: 30

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 246 | 212 | 86.2% | 34 | 13.8% |
| Train | F | 371 | 336 | 90.6% | 35 | 9.4% |
| Train | **All** | **617** | **548** | **88.8%** | **69** | **11.2%** |
| Test | M | 64 | 53 | 82.8% | 11 | 17.2% |
| Test | F | 90 | 80 | 88.9% | 10 | 11.1% |
| Test | **All** | **154** | **133** | **86.4%** | **21** | **13.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 246 | 59.52 ± 11.62 | 23.00 | 59.00 | 85.00 |
| Train | F | 371 | 56.44 ± 11.98 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.67 ± 11.93** | **14.00** | **58.00** | **91.00** |
| Test | M | 64 | 60.75 ± 12.72 | 28.00 | 62.00 | 89.00 |
| Test | F | 90 | 54.97 ± 12.65 | 24.00 | 55.00 | 86.00 |
| Test | **All** | **154** | **57.37 ± 12.99** | **24.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 246 | 24.23 ± 2.78 | 16.39 | 24.20 | 32.56 |
| Train | F | 371 | 22.88 ± 3.11 | 12.02 | 22.78 | 31.50 |
| Train | **All** | **617** | **23.42 ± 3.05** | **12.02** | **23.31** | **32.56** |
| Test | M | 64 | 24.23 ± 3.36 | 17.33 | 24.12 | 32.33 |
| Test | F | 90 | 22.69 ± 2.83 | 16.51 | 22.61 | 30.63 |
| Test | **All** | **154** | **23.33 ± 3.15** | **16.51** | **23.25** | **32.33** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8110 | 0.4418 | 0.1388 | 0.6774 | 0.3939 |
| 2 | 0.7247 | 0.2259 | 0.1511 | 0.7500 | 0.3673 |
| 3 | 0.8971 | 0.5162 | 0.1396 | 0.8537 | 0.5909 |
| 4 | 0.7733 | 0.3696 | 0.2567 | 0.7398 | 0.4074 |
| 5 | 0.7636 | 0.2993 | 0.1802 | 0.6748 | 0.3548 |
| **Mean** | **0.7939** | **0.3706** | **0.1733** | **0.7391** | **0.4229** |
| **±Std** | 0.0584 | 0.1023 | 0.0443 | 0.0651 | 0.0861 |

CrossAttn best val AUC per fold: Fold1=0.8110, Fold2=0.7247, Fold3=0.8971, Fold4=0.7733, Fold5=0.7636

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8174 | 0.4456 | 0.1724 | 0.7013 | 0.4103 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 64 | 0.8165 | 0.5907 | 0.1991 | 0.5938 | 0.3810 |
| F | 90 | 0.8175 | 0.3154 | 0.1534 | 0.7778 | 0.4444 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 92 | 41 |
| **True: Sarco**  | 5 | 16 |

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
