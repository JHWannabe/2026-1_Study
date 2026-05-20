# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:49  |  5-Fold CV  |  Median best epoch: 4

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
| 1 | 0.8470 | 0.3801 | 0.1741 | 0.7424 | 0.4333 |
| 2 | 0.7786 | 0.2492 | 0.1800 | 0.7348 | 0.3636 |
| 3 | 0.8120 | 0.4800 | 0.1950 | 0.7424 | 0.4333 |
| 4 | 0.8227 | 0.4103 | 0.2131 | 0.6818 | 0.3824 |
| 5 | 0.7829 | 0.4189 | 0.1845 | 0.6970 | 0.3750 |
| **Mean** | **0.8086** | **0.3877** | **0.1893** | **0.7197** | **0.3975** |
| **±Std** | 0.0255 | 0.0765 | 0.0137 | 0.0254 | 0.0298 |

CrossAttn best val AUC per fold: Fold1=0.8470, Fold2=0.7786, Fold3=0.8120, Fold4=0.8227, Fold5=0.7829

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8137 | 0.3975 | 0.2165 | 0.6627 | 0.4043 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 66 | 0.7954 | 0.5046 | 0.2005 | 0.7273 | 0.5714 |
| F | 100 | 0.8387 | 0.3334 | 0.2271 | 0.6200 | 0.2692 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 91 | 55 |
| **True: Sarco**  | 1 | 19 |

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
