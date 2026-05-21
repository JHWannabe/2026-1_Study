# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 18:51  |  5-Fold CV  |  Median best epoch: 5

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 254 | 213 | 83.9% | 41 | 16.1% |
| Train | F | 363 | 331 | 91.2% | 32 | 8.8% |
| Train | **All** | **617** | **544** | **88.2%** | **73** | **11.8%** |
| Test | M | 63 | 57 | 90.5% | 6 | 9.5% |
| Test | F | 91 | 78 | 85.7% | 13 | 14.3% |
| Test | **All** | **154** | **135** | **87.7%** | **19** | **12.3%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 254 | 59.18 ± 11.75 | 23.00 | 59.00 | 89.00 |
| Train | F | 363 | 56.05 ± 11.88 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.34 ± 11.93** | **14.00** | **58.00** | **91.00** |
| Test | M | 63 | 60.44 ± 12.65 | 29.00 | 61.00 | 83.00 |
| Test | F | 91 | 57.99 ± 14.06 | 18.00 | 59.00 | 87.00 |
| Test | **All** | **154** | **58.99 ± 13.56** | **18.00** | **59.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 254 | 24.44 ± 3.07 | 14.34 | 24.29 | 32.67 |
| Train | F | 363 | 22.85 ± 3.34 | 12.02 | 22.67 | 34.20 |
| Train | **All** | **617** | **23.50 ± 3.33** | **12.02** | **23.37** | **34.20** |
| Test | M | 63 | 24.71 ± 3.00 | 17.51 | 24.51 | 32.56 |
| Test | F | 91 | 23.49 ± 3.30 | 16.92 | 23.12 | 34.61 |
| Test | **All** | **154** | **23.98 ± 3.23** | **16.92** | **23.62** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8361 | 0.5026 | 0.1674 | 0.7742 | 0.4400 |
| 2 | 0.8636 | 0.4335 | 0.1513 | 0.7823 | 0.5091 |
| 3 | 0.7575 | 0.3701 | 0.1525 | 0.7561 | 0.3750 |
| 4 | 0.8309 | 0.4298 | 0.2034 | 0.7073 | 0.3793 |
| 5 | 0.8302 | 0.5799 | 0.1746 | 0.7073 | 0.4000 |
| **Mean** | **0.8237** | **0.4632** | **0.1698** | **0.7454** | **0.4207** |
| **±Std** | 0.0353 | 0.0719 | 0.0189 | 0.0323 | 0.0498 |

CrossAttn best val AUC per fold: Fold1=0.8361, Fold2=0.8636, Fold3=0.7575, Fold4=0.8309, Fold5=0.8302

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7996 | 0.3404 | 0.1463 | 0.7857 | 0.4407 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 63 | 0.7661 | 0.2730 | 0.1763 | 0.7302 | 0.3704 |
| F | 91 | 0.8294 | 0.4856 | 0.1256 | 0.8242 | 0.5000 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 108 | 27 |
| **True: Sarco**  | 6 | 13 |

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
