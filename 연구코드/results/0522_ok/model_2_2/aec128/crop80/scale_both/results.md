# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:12  |  5-Fold CV  |  Median best epoch: 5

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
| 1 | 0.8852 | 0.4785 | 0.2077 | 0.7101 | 0.3939 |
| 2 | 0.8478 | 0.4154 | 0.1317 | 0.8261 | 0.5000 |
| 3 | 0.7823 | 0.2955 | 0.1743 | 0.7174 | 0.3607 |
| 4 | 0.7487 | 0.2871 | 0.2530 | 0.6304 | 0.3377 |
| 5 | 0.8167 | 0.5553 | 0.2845 | 0.5145 | 0.3232 |
| **Mean** | **0.8162** | **0.4064** | **0.2102** | **0.6797** | **0.3831** |
| **±Std** | 0.0479 | 0.1039 | 0.0544 | 0.1034 | 0.0631 |

CrossAttn best val AUC per fold: Fold1=0.8852, Fold2=0.8478, Fold3=0.7823, Fold4=0.7487, Fold5=0.8167

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8236 | 0.3752 | 0.1782 | 0.7283 | 0.4198 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 67 | 0.7352 | 0.2681 | 0.2259 | 0.6567 | 0.3429 |
| F | 106 | 0.8701 | 0.6398 | 0.1480 | 0.7736 | 0.4783 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 109 | 43 |
| **True: Sarco**  | 4 | 17 |

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
