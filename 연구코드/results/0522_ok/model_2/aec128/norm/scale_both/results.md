# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:27  |  5-Fold CV  |  Median best epoch: 6

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 290 | 243 | 83.8% | 47 | 16.2% |
| Train | F | 400 | 365 | 91.2% | 35 | 8.8% |
| Train | **All** | **690** | **608** | **88.1%** | **82** | **11.9%** |
| Test | M | 67 | 59 | 88.1% | 8 | 11.9% |
| Test | F | 106 | 93 | 87.7% | 13 | 12.3% |
| Test | **All** | **173** | **152** | **87.9%** | **21** | **12.1%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 290 | 59.73 ± 11.52 | 28.00 | 60.00 | 89.00 |
| Train | F | 400 | 56.10 ± 12.16 | 14.00 | 56.50 | 91.00 |
| Train | **All** | **690** | **57.62 ± 12.03** | **14.00** | **58.00** | **91.00** |
| Test | M | 67 | 58.27 ± 12.85 | 23.00 | 59.00 | 83.00 |
| Test | F | 106 | 56.60 ± 12.49 | 23.00 | 56.50 | 86.00 |
| Test | **All** | **173** | **57.25 ± 12.65** | **23.00** | **58.00** | **86.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 290 | 24.38 ± 3.07 | 14.34 | 24.28 | 32.67 |
| Train | F | 400 | 22.97 ± 3.22 | 16.00 | 22.83 | 34.61 |
| Train | **All** | **690** | **23.56 ± 3.23** | **14.34** | **23.44** | **34.61** |
| Test | M | 67 | 24.59 ± 2.87 | 17.65 | 24.11 | 32.56 |
| Test | F | 106 | 23.02 ± 3.69 | 12.02 | 22.66 | 34.20 |
| Test | **All** | **173** | **23.63 ± 3.48** | **12.02** | **23.33** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8663 | 0.4985 | 0.2084 | 0.6522 | 0.3684 |
| 2 | 0.8560 | 0.4654 | 0.1227 | 0.8188 | 0.3902 |
| 3 | 0.7520 | 0.3471 | 0.2038 | 0.7029 | 0.3692 |
| 4 | 0.7278 | 0.2341 | 0.1843 | 0.7391 | 0.4194 |
| 5 | 0.8347 | 0.5183 | 0.2414 | 0.5725 | 0.3516 |
| **Mean** | **0.8074** | **0.4127** | **0.1921** | **0.6971** | **0.3798** |
| **±Std** | 0.0565 | 0.1072 | 0.0393 | 0.0827 | 0.0233 |

CrossAttn best val AUC per fold: Fold1=0.8663, Fold2=0.8560, Fold3=0.7520, Fold4=0.7278, Fold5=0.8347

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8315 | 0.3753 | 0.2024 | 0.6590 | 0.3789 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 67 | 0.7775 | 0.2840 | 0.2454 | 0.6269 | 0.3590 |
| F | 106 | 0.8859 | 0.6845 | 0.1751 | 0.6792 | 0.3929 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 96 | 56 |
| **True: Sarco**  | 3 | 18 |

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
