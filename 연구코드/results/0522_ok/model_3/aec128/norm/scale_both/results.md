# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 12:28  |  5-Fold CV  |  Median best epoch: 29

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 281 | 238 | 84.7% | 43 | 15.3% |
| Train | F | 379 | 343 | 90.5% | 36 | 9.5% |
| Train | **All** | **660** | **581** | **88.0%** | **79** | **12.0%** |
| Test | M | 71 | 60 | 84.5% | 11 | 15.5% |
| Test | F | 95 | 85 | 89.5% | 10 | 10.5% |
| Test | **All** | **166** | **145** | **87.3%** | **21** | **12.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 281 | 59.44 ± 11.39 | 28.00 | 59.00 | 85.00 |
| Train | F | 379 | 56.28 ± 12.55 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **660** | **57.62 ± 12.17** | **14.00** | **58.00** | **91.00** |
| Test | M | 71 | 59.15 ± 13.44 | 23.00 | 60.00 | 89.00 |
| Test | F | 95 | 57.14 ± 11.39 | 29.00 | 57.00 | 87.00 |
| Test | **All** | **166** | **58.00 ± 12.35** | **23.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 281 | 24.32 ± 2.93 | 14.34 | 24.28 | 32.59 |
| Train | F | 379 | 22.96 ± 3.27 | 12.02 | 22.77 | 34.61 |
| Train | **All** | **660** | **23.54 ± 3.20** | **12.02** | **23.44** | **34.61** |
| Test | M | 71 | 24.66 ± 3.33 | 17.51 | 24.42 | 32.67 |
| Test | F | 95 | 22.96 ± 3.57 | 16.00 | 22.64 | 34.20 |
| Test | **All** | **166** | **23.69 ± 3.57** | **16.00** | **23.42** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7983 | 0.4090 | 0.1681 | 0.7424 | 0.3929 |
| 2 | 0.8152 | 0.3040 | 0.1500 | 0.8182 | 0.3684 |
| 3 | 0.8367 | 0.4978 | 0.1683 | 0.7273 | 0.4000 |
| 4 | 0.8782 | 0.5263 | 0.1206 | 0.8106 | 0.5098 |
| 5 | 0.8486 | 0.3576 | 0.1698 | 0.7576 | 0.4667 |
| **Mean** | **0.8354** | **0.4189** | **0.1554** | **0.7712** | **0.4275** |
| **±Std** | 0.0276 | 0.0834 | 0.0189 | 0.0366 | 0.0524 |

CrossAttn best val AUC per fold: Fold1=0.7983, Fold2=0.8152, Fold3=0.8367, Fold4=0.8782, Fold5=0.8486

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8046 | 0.4923 | 0.1819 | 0.7229 | 0.4250 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 71 | 0.8061 | 0.5659 | 0.2039 | 0.6901 | 0.4762 |
| F | 95 | 0.7871 | 0.3771 | 0.1654 | 0.7474 | 0.3684 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 103 | 42 |
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
