# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 12:12  |  5-Fold CV  |  Median best epoch: 32

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 285 | 241 | 84.6% | 44 | 15.4% |
| Train | F | 402 | 364 | 90.5% | 38 | 9.5% |
| Train | **All** | **687** | **605** | **88.1%** | **82** | **11.9%** |
| Test | M | 71 | 60 | 84.5% | 11 | 15.5% |
| Test | F | 101 | 91 | 90.1% | 10 | 9.9% |
| Test | **All** | **172** | **151** | **87.8%** | **21** | **12.2%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 285 | 59.36 ± 11.63 | 23.00 | 59.00 | 85.00 |
| Train | F | 402 | 56.39 ± 12.21 | 14.00 | 57.00 | 91.00 |
| Train | **All** | **687** | **57.62 ± 12.06** | **14.00** | **58.00** | **91.00** |
| Test | M | 71 | 59.82 ± 12.49 | 28.00 | 61.00 | 89.00 |
| Test | F | 101 | 55.65 ± 12.38 | 24.00 | 56.00 | 86.00 |
| Test | **All** | **172** | **57.37 ± 12.59** | **24.00** | **57.50** | **89.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 285 | 24.43 ± 2.95 | 14.34 | 24.28 | 32.67 |
| Train | F | 402 | 22.98 ± 3.35 | 12.02 | 22.81 | 34.61 |
| Train | **All** | **687** | **23.58 ± 3.27** | **12.02** | **23.44** | **34.61** |
| Test | M | 71 | 24.27 ± 3.24 | 17.33 | 24.12 | 32.33 |
| Test | F | 101 | 22.90 ± 3.20 | 16.00 | 22.64 | 34.20 |
| Test | **All** | **172** | **23.47 ± 3.29** | **16.00** | **23.27** | **34.20** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8240 | 0.4339 | 0.1091 | 0.8333 | 0.3429 |
| 2 | 0.7978 | 0.3654 | 0.2137 | 0.6957 | 0.4000 |
| 3 | 0.8512 | 0.4176 | 0.1678 | 0.7226 | 0.4242 |
| 4 | 0.8414 | 0.4840 | 0.2002 | 0.7080 | 0.3939 |
| 5 | 0.8569 | 0.3663 | 0.1935 | 0.7080 | 0.4286 |
| **Mean** | **0.8343** | **0.4134** | **0.1768** | **0.7335** | **0.3979** |
| **±Std** | 0.0214 | 0.0446 | 0.0370 | 0.0506 | 0.0306 |

CrossAttn best val AUC per fold: Fold1=0.8240, Fold2=0.7978, Fold3=0.8512, Fold4=0.8414, Fold5=0.8569

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8146 | 0.3235 | 0.1877 | 0.7384 | 0.4578 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 71 | 0.8015 | 0.3879 | 0.2059 | 0.7042 | 0.4878 |
| F | 101 | 0.8187 | 0.2800 | 0.1749 | 0.7624 | 0.4286 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 108 | 43 |
| **True: Sarco**  | 2 | 19 |

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
