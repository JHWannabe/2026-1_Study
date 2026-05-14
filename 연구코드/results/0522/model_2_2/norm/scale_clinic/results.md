# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-14 19:43  |  5-Fold CV  |  Median best epoch: 4

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 355 | 292 | 82.3% | 63 | 17.7% |
| Train | F | 661 | 614 | 92.9% | 47 | 7.1% |
| Train | **All** | **1016** | **906** | **89.2%** | **110** | **10.8%** |
| Test | M | 97 | 83 | 85.6% | 14 | 14.4% |
| Test | F | 158 | 145 | 91.8% | 13 | 8.2% |
| Test | **All** | **255** | **228** | **89.4%** | **27** | **10.6%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 355 | 59.92 ± 12.67 | 18.00 | 60.00 | 89.00 |
| Train | F | 661 | 55.55 ± 11.94 | 18.00 | 55.00 | 91.00 |
| Train | **All** | **1016** | **57.07 ± 12.38** | **18.00** | **57.00** | **91.00** |
| Test | M | 97 | 58.63 ± 12.43 | 28.00 | 59.00 | 88.00 |
| Test | F | 158 | 55.27 ± 11.46 | 23.00 | 56.00 | 86.00 |
| Test | **All** | **255** | **56.55 ± 11.95** | **23.00** | **57.00** | **88.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 355 | 24.22 ± 3.38 | 14.48 | 24.16 | 36.76 |
| Train | F | 661 | 23.14 ± 3.39 | 14.40 | 22.83 | 36.24 |
| Train | **All** | **1016** | **23.52 ± 3.42** | **14.40** | **23.37** | **36.76** |
| Test | M | 97 | 24.50 ± 3.14 | 18.37 | 24.49 | 35.68 |
| Test | F | 158 | 23.11 ± 3.24 | 16.87 | 22.72 | 34.23 |
| Test | **All** | **255** | **23.64 ± 3.27** | **16.87** | **23.34** | **35.68** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### Logistic Regression

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7235 | 0.2431 | 0.1624 | 0.7794 | 0.3284 |
| 2 | 0.8400 | 0.4453 | 0.1372 | 0.7882 | 0.4110 |
| 3 | 0.8707 | 0.3946 | 0.1425 | 0.8030 | 0.4737 |
| 4 | 0.8430 | 0.4209 | 0.1708 | 0.7635 | 0.4146 |
| 5 | 0.8096 | 0.3328 | 0.1654 | 0.7685 | 0.4198 |
| **Mean** | **0.8174** | **0.3673** | **0.1557** | **0.7805** | **0.4095** |
| **±Std** | 0.0508 | 0.0726 | 0.0133 | 0.0141 | 0.0466 |

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7992 | 0.3731 | 0.1670 | 0.7500 | 0.3855 |
| 2 | 0.8403 | 0.4243 | 0.1644 | 0.7537 | 0.3902 |
| 3 | 0.8302 | 0.3538 | 0.1901 | 0.7094 | 0.3656 |
| 4 | 0.8528 | 0.5538 | 0.2603 | 0.6010 | 0.3193 |
| 5 | 0.8307 | 0.3338 | 0.2616 | 0.5813 | 0.3200 |
| **Mean** | **0.8307** | **0.4078** | **0.2087** | **0.6791** | **0.3561** |
| **±Std** | 0.0177 | 0.0790 | 0.0436 | 0.0737 | 0.0309 |

CrossAttn best val AUC per fold: Fold1=0.7992, Fold2=0.8403, Fold3=0.8302, Fold4=0.8528, Fold5=0.8307

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| Log. Reg. | 0.6589 | 0.1890 | 0.1742 | 0.7725 | 0.3095 |
| CrossAttn | 0.7703 | 0.2582 | 0.2056 | 0.6980 | 0.2936 |

### By Sex

#### Logistic Regression

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.6618 | 0.2296 | 0.2482 | 0.6907 | 0.4000 |
| F | 158 | 0.6207 | 0.1453 | 0.1288 | 0.8228 | 0.1765 |

#### CrossAttn

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 97 | 0.7685 | 0.3115 | 0.2617 | 0.6186 | 0.3729 |
| F | 158 | 0.7411 | 0.2086 | 0.1712 | 0.7468 | 0.2000 |

---

## 3. Confusion Matrix (Test Set)

### Logistic Regression

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 184 | 44 |
| **True: Sarco**  | 14 | 13 |

### CrossAttn

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 162 | 66 |
| **True: Sarco**  | 11 | 16 |

---

## 4. Figures

| File | Description |
|------|-------------|
| `data_distribution.png` | Train/Test class·Age·BMI distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (LR & CrossAttn) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curves |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices (LR & CrossAttn) |
| `calibration.png` | Calibration plot (reliability diagram) + Precision-Recall curve |
