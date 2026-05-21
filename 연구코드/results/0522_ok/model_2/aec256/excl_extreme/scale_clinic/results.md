# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 19:12  |  5-Fold CV  |  Median best epoch: 9

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 246 | 212 | 86.2% | 34 | 13.8% |
| Train | F | 371 | 336 | 90.6% | 35 | 9.4% |
| Train | **All** | **617** | **548** | **88.8%** | **69** | **11.2%** |
| Test | M | 64 | 53 | 82.8% | 11 | 17.2% |
| Test | F | 90 | 80 | 88.9% | 10 | 11.1% |
| Test | **All** | **154** | **133** | **86.4%** | **21** | **13.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 246 | 59.52 ± 11.62 | 23.00 | 59.00 | 85.00 |
| Train | F | 371 | 56.44 ± 11.98 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **617** | **57.67 ± 11.93** | **14.00** | **58.00** | **91.00** |
| Test | M | 64 | 60.75 ± 12.72 | 28.00 | 62.00 | 89.00 |
| Test | F | 90 | 54.97 ± 12.65 | 24.00 | 55.00 | 86.00 |
| Test | **All** | **154** | **57.37 ± 12.99** | **24.00** | **58.00** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 246 | 24.23 ± 2.78 | 16.39 | 24.20 | 32.56 |
| Train | F | 371 | 22.88 ± 3.11 | 12.02 | 22.78 | 31.50 |
| Train | **All** | **617** | **23.42 ± 3.05** | **12.02** | **23.31** | **32.56** |
| Test | M | 64 | 24.23 ± 3.36 | 17.33 | 24.12 | 32.33 |
| Test | F | 90 | 22.69 ± 2.83 | 16.51 | 22.61 | 30.63 |
| Test | **All** | **154** | **23.33 ± 3.15** | **16.51** | **23.25** | **32.33** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8143 | 0.4258 | 0.2024 | 0.6774 | 0.3750 |
| 2 | 0.8227 | 0.4319 | 0.1790 | 0.6694 | 0.3279 |
| 3 | 0.8532 | 0.3986 | 0.1244 | 0.7805 | 0.3721 |
| 4 | 0.7700 | 0.2598 | 0.2416 | 0.6504 | 0.3385 |
| 5 | 0.7860 | 0.2375 | 0.2619 | 0.5285 | 0.2927 |
| **Mean** | **0.8092** | **0.3507** | **0.2018** | **0.6612** | **0.3412** |
| **±Std** | 0.0291 | 0.0844 | 0.0484 | 0.0804 | 0.0304 |

CrossAttn best val AUC per fold: Fold1=0.8143, Fold2=0.8227, Fold3=0.8532, Fold4=0.7700, Fold5=0.7860

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8833 | 0.4842 | 0.1748 | 0.7143 | 0.4762 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 64 | 0.8491 | 0.4578 | 0.2023 | 0.6719 | 0.5116 |
| F | 90 | 0.9025 | 0.6890 | 0.1553 | 0.7444 | 0.4390 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 90 | 43 |
| **True: Sarco**  | 1 | 20 |

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
