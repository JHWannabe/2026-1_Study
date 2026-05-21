# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 22:14  |  5-Fold CV  |  Median best epoch: 14

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 302 | 256 | 84.8% | 46 | 15.2% |
| Train | F | 533 | 493 | 92.5% | 40 | 7.5% |
| Train | **All** | **835** | **749** | **89.7%** | **86** | **10.3%** |
| Test | M | 80 | 69 | 86.2% | 11 | 13.8% |
| Test | F | 129 | 118 | 91.5% | 11 | 8.5% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 302 | 59.41 ± 12.04 | 20.00 | 59.00 | 89.00 |
| Train | F | 533 | 55.34 ± 11.98 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **835** | **56.81 ± 12.16** | **14.00** | **57.00** | **91.00** |
| Test | M | 80 | 58.96 ± 12.47 | 29.00 | 60.00 | 84.00 |
| Test | F | 129 | 55.14 ± 12.04 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **209** | **56.60 ± 12.35** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 302 | 24.39 ± 2.93 | 17.33 | 24.26 | 32.67 |
| Train | F | 533 | 23.14 ± 3.20 | 16.00 | 22.95 | 34.61 |
| Train | **All** | **835** | **23.59 ± 3.16** | **16.00** | **23.46** | **34.61** |
| Test | M | 80 | 24.14 ± 3.35 | 14.34 | 24.03 | 32.56 |
| Test | F | 129 | 22.96 ± 3.62 | 12.02 | 22.51 | 32.48 |
| Test | **All** | **209** | **23.41 ± 3.56** | **12.02** | **23.26** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8988 | 0.4771 | 0.1617 | 0.7066 | 0.4096 |
| 2 | 0.8133 | 0.4564 | 0.1596 | 0.8323 | 0.4615 |
| 3 | 0.7749 | 0.3102 | 0.2320 | 0.7305 | 0.4000 |
| 4 | 0.8024 | 0.4592 | 0.1508 | 0.7425 | 0.3944 |
| 5 | 0.8650 | 0.4831 | 0.1307 | 0.8743 | 0.5714 |
| **Mean** | **0.8309** | **0.4372** | **0.1670** | **0.7772** | **0.4474** |
| **±Std** | 0.0448 | 0.0643 | 0.0343 | 0.0645 | 0.0664 |

CrossAttn best val AUC per fold: Fold1=0.8988, Fold2=0.8133, Fold3=0.7749, Fold4=0.8024, Fold5=0.8650

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7863 | 0.3017 | 0.2021 | 0.7033 | 0.3261 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 80 | 0.7997 | 0.3487 | 0.2226 | 0.6625 | 0.4255 |
| F | 129 | 0.7435 | 0.2626 | 0.1894 | 0.7287 | 0.2222 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 132 | 55 |
| **True: Sarco**  | 7 | 15 |

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
