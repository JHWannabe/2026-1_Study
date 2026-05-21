# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 18:41  |  5-Fold CV  |  Median best epoch: 15

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 334 | 282 | 84.4% | 52 | 15.6% |
| Train | F | 595 | 548 | 92.1% | 47 | 7.9% |
| Train | **All** | **929** | **830** | **89.3%** | **99** | **10.7%** |
| Test | M | 85 | 71 | 83.5% | 14 | 16.5% |
| Test | F | 148 | 137 | 92.6% | 11 | 7.4% |
| Test | **All** | **233** | **208** | **89.3%** | **25** | **10.7%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 59.72 ± 12.11 | 20.00 | 60.00 | 89.00 |
| Train | F | 595 | 55.26 ± 11.90 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **929** | **56.86 ± 12.16** | **14.00** | **57.00** | **91.00** |
| Test | M | 85 | 59.26 ± 12.48 | 29.00 | 60.00 | 84.00 |
| Test | F | 148 | 55.47 ± 11.64 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **233** | **56.85 ± 12.09** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 334 | 24.38 ± 2.93 | 17.33 | 24.24 | 32.67 |
| Train | F | 595 | 23.15 ± 3.17 | 16.00 | 22.96 | 34.61 |
| Train | **All** | **929** | **23.59 ± 3.14** | **16.00** | **23.44** | **34.61** |
| Test | M | 85 | 24.33 ± 3.39 | 14.34 | 24.24 | 32.56 |
| Test | F | 148 | 23.06 ± 3.53 | 12.02 | 22.64 | 32.48 |
| Test | **All** | **233** | **23.52 ± 3.53** | **12.02** | **23.41** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8518 | 0.4343 | 0.1205 | 0.8226 | 0.4000 |
| 2 | 0.8419 | 0.4286 | 0.2314 | 0.6237 | 0.3396 |
| 3 | 0.8090 | 0.3459 | 0.2002 | 0.7043 | 0.3820 |
| 4 | 0.8217 | 0.3974 | 0.2083 | 0.6774 | 0.3750 |
| 5 | 0.7743 | 0.2747 | 0.2259 | 0.6486 | 0.3299 |
| **Mean** | **0.8197** | **0.3762** | **0.1973** | **0.6953** | **0.3653** |
| **±Std** | 0.0272 | 0.0597 | 0.0400 | 0.0692 | 0.0264 |

CrossAttn best val AUC per fold: Fold1=0.8518, Fold2=0.8419, Fold3=0.8090, Fold4=0.8217, Fold5=0.7743

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8081 | 0.3326 | 0.1944 | 0.7124 | 0.3964 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 85 | 0.8068 | 0.3782 | 0.2339 | 0.6941 | 0.5185 |
| F | 148 | 0.7545 | 0.2979 | 0.1717 | 0.7230 | 0.2807 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 144 | 64 |
| **True: Sarco**  | 3 | 22 |

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
