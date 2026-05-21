# SMI Binary Classification — Results

Generated: 2026-05-21 18:31  |  5-Fold CV  |  Model 1 (Clinic Only, LR)

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

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8096 | 0.4533 | 0.1618 | 0.7742 | 0.3824 |
| 2 | 0.8416 | 0.4623 | 0.1716 | 0.7473 | 0.3896 |
| 3 | 0.8259 | 0.3161 | 0.1936 | 0.7043 | 0.3529 |
| 4 | 0.8012 | 0.4279 | 0.1757 | 0.7688 | 0.4110 |
| 5 | 0.7470 | 0.2689 | 0.1973 | 0.7027 | 0.3210 |
| **Mean** | **0.8051** | **0.3857** | **0.1800** | **0.7395** | **0.3714** |
| **±Std** | 0.0322 | 0.0784 | 0.0134 | 0.0307 | 0.0313 |

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.8119 | 0.4103 | 0.1878 | 0.7167 | 0.3889 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 85 | 0.7726 | 0.4686 | 0.2460 | 0.5882 | 0.4262 |
| F | 148 | 0.7969 | 0.3912 | 0.1544 | 0.7905 | 0.3404 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 146 | 62 |
| **True: Sarco**  | 4 | 21 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `confusion_matrices.png` | Test-set confusion matrices (overall + by sex) |
| `test_roc_curves.png` | Final test-set ROC curve (overall) |
| `test_roc_by_sex.png` | Final test-set ROC curves split by sex |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
