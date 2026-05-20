# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:53  |  5-Fold CV  |  Median best epoch: 18

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 286 | 245 | 85.7% | 41 | 14.3% |
| Train | F | 374 | 335 | 89.6% | 39 | 10.4% |
| Train | **All** | **660** | **580** | **87.9%** | **80** | **12.1%** |
| Test | M | 66 | 53 | 80.3% | 13 | 19.7% |
| Test | F | 100 | 93 | 93.0% | 7 | 7.0% |
| Test | **All** | **166** | **146** | **88.0%** | **20** | **12.0%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 286 | 59.38 ± 11.37 | 23.00 | 60.00 | 83.00 |
| Train | F | 374 | 56.41 ± 12.15 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **660** | **57.70 ± 11.91** | **14.00** | **58.00** | **91.00** |
| Test | M | 66 | 59.38 ± 13.64 | 28.00 | 59.00 | 89.00 |
| Test | F | 100 | 56.60 ± 12.99 | 24.00 | 56.50 | 87.00 |
| Test | **All** | **166** | **57.70 ± 13.32** | **24.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 286 | 24.38 ± 3.00 | 14.34 | 24.35 | 32.67 |
| Train | F | 374 | 22.87 ± 3.33 | 12.02 | 22.74 | 34.61 |
| Train | **All** | **660** | **23.53 ± 3.27** | **12.02** | **23.45** | **34.61** |
| Test | M | 66 | 24.44 ± 3.12 | 17.57 | 24.00 | 32.56 |
| Test | F | 100 | 23.27 ± 3.32 | 16.44 | 22.81 | 34.20 |
| Test | **All** | **166** | **23.74 ± 3.29** | **16.44** | **23.35** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8637 | 0.4390 | 0.1548 | 0.7803 | 0.4912 |
| 2 | 0.7710 | 0.3479 | 0.2064 | 0.6591 | 0.3284 |
| 3 | 0.8244 | 0.4239 | 0.2047 | 0.6667 | 0.3714 |
| 4 | 0.8270 | 0.3324 | 0.2188 | 0.6288 | 0.3636 |
| 5 | 0.7732 | 0.4131 | 0.1651 | 0.7348 | 0.3636 |
| **Mean** | **0.8119** | **0.3913** | **0.1900** | **0.6939** | **0.3837** |
| **±Std** | 0.0353 | 0.0428 | 0.0252 | 0.0554 | 0.0558 |

CrossAttn best val AUC per fold: Fold1=0.8637, Fold2=0.7710, Fold3=0.8244, Fold4=0.8270, Fold5=0.7732

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8158 | 0.4120 | 0.1975 | 0.6566 | 0.3871 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 66 | 0.7939 | 0.5129 | 0.2211 | 0.6364 | 0.5000 |
| F | 100 | 0.7972 | 0.3011 | 0.1819 | 0.6700 | 0.2667 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 91 | 55 |
| **True: Sarco**  | 2 | 18 |

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
