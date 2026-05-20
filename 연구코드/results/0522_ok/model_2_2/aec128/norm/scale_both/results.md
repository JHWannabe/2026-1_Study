# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 12:27  |  5-Fold CV  |  Median best epoch: 8

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
| 1 | 0.8494 | 0.4204 | 0.1487 | 0.7609 | 0.4211 |
| 2 | 0.8438 | 0.4706 | 0.2028 | 0.7174 | 0.4000 |
| 3 | 0.8299 | 0.3194 | 0.1918 | 0.6884 | 0.4110 |
| 4 | 0.7696 | 0.2634 | 0.1507 | 0.7681 | 0.3846 |
| 5 | 0.8721 | 0.6025 | 0.1762 | 0.6957 | 0.4167 |
| **Mean** | **0.8330** | **0.4152** | **0.1740** | **0.7261** | **0.4067** |
| **±Std** | 0.0345 | 0.1186 | 0.0216 | 0.0329 | 0.0131 |

CrossAttn best val AUC per fold: Fold1=0.8494, Fold2=0.8438, Fold3=0.8299, Fold4=0.7696, Fold5=0.8721

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8380 | 0.5394 | 0.1292 | 0.8035 | 0.4333 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 67 | 0.8284 | 0.5618 | 0.1756 | 0.7015 | 0.3333 |
| F | 106 | 0.8668 | 0.6342 | 0.0998 | 0.8679 | 0.5333 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 126 | 26 |
| **True: Sarco**  | 8 | 13 |

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
