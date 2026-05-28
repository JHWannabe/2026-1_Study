# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-27 16:09  |  5-Fold CV  |  Median best epoch: 6

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 245 | 211 | 86.1% | 34 | 13.9% |
| Train | F | 372 | 337 | 90.6% | 35 | 9.4% |
| Train | **All** | **617** | **548** | **88.8%** | **69** | **11.2%** |
| Test | M | 64 | 53 | 82.8% | 11 | 17.2% |
| Test | F | 90 | 80 | 88.9% | 10 | 11.1% |
| Test | **All** | **154** | **133** | **86.4%** | **21** | **13.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 245 | 59.47 ± 11.67 | 23.00 | 59.00 | 85.00 |
| Train | F | 372 | 56.48 ± 11.97 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.67 ± 11.95** | **14.00** | **58.00** | **91.00** |
| Test | M | 64 | 60.75 ± 12.72 | 28.00 | 62.00 | 89.00 |
| Test | F | 90 | 54.97 ± 12.65 | 24.00 | 55.00 | 86.00 |
| Test | **All** | **154** | **57.37 ± 12.99** | **24.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 245 | 24.20 ± 2.76 | 16.39 | 24.17 | 32.56 |
| Train | F | 372 | 22.89 ± 3.12 | 12.02 | 22.81 | 31.50 |
| Train | **All** | **617** | **23.41 ± 3.05** | **12.02** | **23.31** | **32.56** |
| Test | M | 64 | 24.23 ± 3.36 | 17.33 | 24.12 | 32.33 |
| Test | F | 90 | 22.69 ± 2.83 | 16.51 | 22.61 | 30.63 |
| Test | **All** | **154** | **23.33 ± 3.15** | **16.51** | **23.25** | **32.33** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8039 | 0.5432 | 0.1926 | 0.7177 | 0.4068 |
| 2 | 0.7227 | 0.2384 | 0.1694 | 0.7903 | 0.4091 |
| 3 | 0.8277 | 0.2856 | 0.1453 | 0.6585 | 0.4000 |
| 4 | 0.7739 | 0.3847 | 0.2594 | 0.7236 | 0.3929 |
| 5 | 0.8350 | 0.3648 | 0.2424 | 0.8130 | 0.4889 |
| **Mean** | **0.7926** | **0.3633** | **0.2018** | **0.7406** | **0.4195** |
| **±Std** | 0.0409 | 0.1044 | 0.0431 | 0.0552 | 0.0351 |

CrossAttn best val AUC per fold: Fold1=0.8039, Fold2=0.7227, Fold3=0.8277, Fold4=0.7739, Fold5=0.8350

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7852 | 0.3351 | 0.1616 | 0.7922 | 0.4286 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 64 | 0.8182 | 0.4470 | 0.1706 | 0.7812 | 0.5625 |
| F | 90 | 0.7338 | 0.2245 | 0.1552 | 0.8000 | 0.2500 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 110 | 23 |
| **True: Sarco**  | 9 | 12 |

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
