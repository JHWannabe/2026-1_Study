# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-20 20:32  |  5-Fold CV  |  Median best epoch: 6

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 284 | 240 | 84.5% | 44 | 15.5% |
| Train | F | 376 | 340 | 90.4% | 36 | 9.6% |
| Train | **All** | **660** | **580** | **87.9%** | **80** | **12.1%** |
| Test | M | 68 | 58 | 85.3% | 10 | 14.7% |
| Test | F | 98 | 88 | 89.8% | 10 | 10.2% |
| Test | **All** | **166** | **146** | **88.0%** | **20** | **12.0%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 284 | 59.65 ± 12.06 | 23.00 | 60.00 | 89.00 |
| Train | F | 376 | 56.61 ± 12.46 | 14.00 | 58.00 | 91.00 |
| Train | **All** | **660** | **57.92 ± 12.38** | **14.00** | **59.00** | **91.00** |
| Test | M | 68 | 58.24 ± 10.73 | 32.00 | 58.00 | 82.00 |
| Test | F | 98 | 55.86 ± 11.81 | 29.00 | 54.50 | 84.00 |
| Test | **All** | **166** | **56.83 ± 11.44** | **29.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 284 | 24.38 ± 3.03 | 14.34 | 24.18 | 32.67 |
| Train | F | 376 | 22.90 ± 3.23 | 12.02 | 22.69 | 34.20 |
| Train | **All** | **660** | **23.53 ± 3.23** | **12.02** | **23.37** | **34.20** |
| Test | M | 68 | 24.45 ± 2.99 | 17.51 | 24.56 | 32.56 |
| Test | F | 98 | 23.18 ± 3.68 | 16.00 | 23.06 | 34.61 |
| Test | **All** | **166** | **23.70 ± 3.47** | **16.00** | **23.59** | **34.61** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8238 | 0.4367 | 0.2801 | 0.5455 | 0.3182 |
| 2 | 0.8901 | 0.4966 | 0.2114 | 0.6515 | 0.3947 |
| 3 | 0.8120 | 0.3795 | 0.1841 | 0.7803 | 0.4314 |
| 4 | 0.7721 | 0.3879 | 0.1487 | 0.7879 | 0.3636 |
| 5 | 0.8815 | 0.5360 | 0.1139 | 0.7955 | 0.5091 |
| **Mean** | **0.8359** | **0.4473** | **0.1877** | **0.7121** | **0.4034** |
| **±Std** | 0.0443 | 0.0609 | 0.0567 | 0.0988 | 0.0646 |

CrossAttn best val AUC per fold: Fold1=0.8238, Fold2=0.8901, Fold3=0.8120, Fold4=0.7721, Fold5=0.8815

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8014 | 0.4032 | 0.1655 | 0.7048 | 0.3797 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 68 | 0.7862 | 0.4210 | 0.1748 | 0.6765 | 0.4211 |
| F | 98 | 0.8125 | 0.4545 | 0.1590 | 0.7245 | 0.3415 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 102 | 44 |
| **True: Sarco**  | 5 | 15 |

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
