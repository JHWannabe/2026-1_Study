# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 18:36  |  5-Fold CV  |  Median best epoch: 24

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 282 | 235 | 83.3% | 47 | 16.7% |
| Train | F | 405 | 370 | 91.4% | 35 | 8.6% |
| Train | **All** | **687** | **605** | **88.1%** | **82** | **11.9%** |
| Test | M | 74 | 66 | 89.2% | 8 | 10.8% |
| Test | F | 98 | 85 | 86.7% | 13 | 13.3% |
| Test | **All** | **172** | **151** | **87.8%** | **21** | **12.2%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 282 | 59.14 ± 11.64 | 23.00 | 59.00 | 89.00 |
| Train | F | 405 | 55.71 ± 11.75 | 14.00 | 56.00 | 91.00 |
| Train | **All** | **687** | **57.12 ± 11.82** | **14.00** | **58.00** | **91.00** |
| Test | M | 74 | 60.66 ± 12.38 | 29.00 | 61.50 | 83.00 |
| Test | F | 98 | 58.45 ± 13.90 | 18.00 | 59.00 | 87.00 |
| Test | **All** | **172** | **59.40 ± 13.31** | **18.00** | **60.00** | **87.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 282 | 24.34 ± 3.06 | 14.34 | 24.22 | 32.67 |
| Train | F | 405 | 22.83 ± 3.30 | 12.02 | 22.66 | 34.20 |
| Train | **All** | **687** | **23.45 ± 3.29** | **12.02** | **23.33** | **34.20** |
| Test | M | 74 | 24.61 ± 2.84 | 17.51 | 24.43 | 32.56 |
| Test | F | 98 | 23.55 ± 3.36 | 16.92 | 23.23 | 34.61 |
| Test | **All** | **172** | **24.01 ± 3.19** | **16.92** | **23.62** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8639 | 0.5959 | 0.1587 | 0.7754 | 0.4746 |
| 2 | 0.7725 | 0.3659 | 0.2241 | 0.6232 | 0.3500 |
| 3 | 0.7944 | 0.2778 | 0.2195 | 0.6861 | 0.3768 |
| 4 | 0.8771 | 0.5213 | 0.1467 | 0.7737 | 0.4364 |
| 5 | 0.8223 | 0.4685 | 0.1982 | 0.7226 | 0.4062 |
| **Mean** | **0.8260** | **0.4459** | **0.1894** | **0.7162** | **0.4088** |
| **±Std** | 0.0398 | 0.1126 | 0.0315 | 0.0573 | 0.0438 |

CrossAttn best val AUC per fold: Fold1=0.8639, Fold2=0.7725, Fold3=0.7944, Fold4=0.8771, Fold5=0.8223

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7474 | 0.2312 | 0.1867 | 0.7442 | 0.3529 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 74 | 0.7576 | 0.2119 | 0.2397 | 0.6351 | 0.3077 |
| F | 98 | 0.7638 | 0.3373 | 0.1467 | 0.8265 | 0.4138 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 116 | 35 |
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
