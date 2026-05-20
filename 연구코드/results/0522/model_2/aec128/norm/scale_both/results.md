# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:38  |  5-Fold CV  |  Median best epoch: 5

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
| 1 | 0.8425 | 0.4665 | 0.1644 | 0.7540 | 0.4250 |
| 2 | 0.8153 | 0.4788 | 0.1612 | 0.8021 | 0.3729 |
| 3 | 0.8153 | 0.3535 | 0.1929 | 0.7258 | 0.3704 |
| 4 | 0.8163 | 0.3002 | 0.2156 | 0.6989 | 0.3778 |
| 5 | 0.8244 | 0.3945 | 0.1844 | 0.7151 | 0.4045 |
| **Mean** | **0.8228** | **0.3987** | **0.1837** | **0.7392** | **0.3901** |
| **±Std** | 0.0105 | 0.0675 | 0.0199 | 0.0362 | 0.0213 |

CrossAttn best val AUC per fold: Fold1=0.8425, Fold2=0.8153, Fold3=0.8153, Fold4=0.8163, Fold5=0.8244

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7922 | 0.3032 | 0.1676 | 0.7564 | 0.3736 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 86 | 0.6813 | 0.2470 | 0.2544 | 0.6279 | 0.3600 |
| F | 148 | 0.8519 | 0.4742 | 0.1172 | 0.8311 | 0.3902 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 160 | 49 |
| **True: Sarco**  | 8 | 17 |

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
