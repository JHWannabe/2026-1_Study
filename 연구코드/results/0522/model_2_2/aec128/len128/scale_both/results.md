# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:07  |  5-Fold CV  |  Median best epoch: 9

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 334 | 280 | 83.8% | 54 | 16.2% |
| Train | F | 598 | 553 | 92.5% | 45 | 7.5% |
| Train | **All** | **932** | **833** | **89.4%** | **99** | **10.6%** |
| Test | M | 86 | 74 | 86.0% | 12 | 14.0% |
| Test | F | 148 | 135 | 91.2% | 13 | 8.8% |
| Test | **All** | **234** | **209** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 59.81 ± 12.21 | 20.00 | 60.00 | 89.00 |
| Train | F | 598 | 55.43 ± 11.87 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **932** | **57.00 ± 12.17** | **14.00** | **57.00** | **91.00** |
| Test | M | 86 | 58.88 ± 11.98 | 28.00 | 60.50 | 84.00 |
| Test | F | 148 | 54.66 ± 11.70 | 23.00 | 54.00 | 87.00 |
| Test | **All** | **234** | **56.21 ± 11.98** | **23.00** | **56.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 24.31 ± 3.02 | 14.34 | 24.27 | 32.67 |
| Train | F | 598 | 23.26 ± 3.27 | 12.02 | 23.06 | 34.61 |
| Train | **All** | **932** | **23.64 ± 3.22** | **12.02** | **23.56** | **34.61** |
| Test | M | 86 | 24.67 ± 3.13 | 17.43 | 24.15 | 32.56 |
| Test | F | 148 | 22.65 ± 3.12 | 16.44 | 22.14 | 34.20 |
| Test | **All** | **234** | **23.39 ± 3.27** | **16.44** | **23.30** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8174 | 0.4521 | 0.2216 | 0.6578 | 0.3469 |
| 2 | 0.8003 | 0.4246 | 0.2682 | 0.5187 | 0.2742 |
| 3 | 0.8204 | 0.3677 | 0.2049 | 0.6882 | 0.3409 |
| 4 | 0.8139 | 0.3626 | 0.1336 | 0.8118 | 0.4262 |
| 5 | 0.8410 | 0.3754 | 0.1673 | 0.7419 | 0.4146 |
| **Mean** | **0.8186** | **0.3965** | **0.1991** | **0.6837** | **0.3606** |
| **±Std** | 0.0131 | 0.0355 | 0.0461 | 0.0977 | 0.0553 |

CrossAttn best val AUC per fold: Fold1=0.8174, Fold2=0.8003, Fold3=0.8204, Fold4=0.8139, Fold5=0.8410

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7889 | 0.3063 | 0.2246 | 0.6325 | 0.3175 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 86 | 0.7218 | 0.3222 | 0.2727 | 0.6279 | 0.3846 |
| F | 148 | 0.8205 | 0.4415 | 0.1966 | 0.6351 | 0.2703 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 128 | 81 |
| **True: Sarco**  | 5 | 20 |

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
