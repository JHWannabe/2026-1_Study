# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-21 12:46  |  5-Fold CV  |  Median best epoch: 20

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 303 | 257 | 84.8% | 46 | 15.2% |
| Train | F | 532 | 493 | 92.7% | 39 | 7.3% |
| Train | **All** | **835** | **750** | **89.8%** | **85** | **10.2%** |
| Test | M | 79 | 68 | 86.1% | 11 | 13.9% |
| Test | F | 130 | 119 | 91.5% | 11 | 8.5% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 303 | 59.42 ± 12.02 | 20.00 | 59.00 | 89.00 |
| Train | F | 532 | 55.34 ± 11.97 | 14.00 | 55.00 | 91.00 |
| Train | **All** | **835** | **56.82 ± 12.15** | **14.00** | **57.00** | **91.00** |
| Test | M | 79 | 58.90 ± 12.54 | 29.00 | 60.00 | 84.00 |
| Test | F | 130 | 55.08 ± 12.01 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **209** | **56.52 ± 12.35** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 303 | 24.39 ± 2.93 | 17.33 | 24.28 | 32.67 |
| Train | F | 532 | 23.15 ± 3.19 | 16.00 | 22.95 | 34.61 |
| Train | **All** | **835** | **23.60 ± 3.16** | **16.00** | **23.46** | **34.61** |
| Test | M | 79 | 24.06 ± 3.31 | 14.34 | 23.94 | 32.56 |
| Test | F | 130 | 22.95 ± 3.60 | 12.02 | 22.49 | 32.48 |
| Test | **All** | **209** | **23.37 ± 3.54** | **12.02** | **23.24** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.8792 | 0.4140 | 0.1768 | 0.7246 | 0.4103 |
| 2 | 0.8251 | 0.3413 | 0.1768 | 0.7425 | 0.4110 |
| 3 | 0.8859 | 0.3850 | 0.1133 | 0.8263 | 0.4082 |
| 4 | 0.7867 | 0.3363 | 0.1275 | 0.8024 | 0.3265 |
| 5 | 0.8498 | 0.4444 | 0.1217 | 0.8323 | 0.4615 |
| **Mean** | **0.8453** | **0.3842** | **0.1432** | **0.7856** | **0.4035** |
| **±Std** | 0.0365 | 0.0416 | 0.0278 | 0.0441 | 0.0434 |

CrossAttn best val AUC per fold: Fold1=0.8792, Fold2=0.8251, Fold3=0.8859, Fold4=0.7867, Fold5=0.8498

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.7910 | 0.2751 | 0.2496 | 0.6029 | 0.3252 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 79 | 0.7754 | 0.2919 | 0.2880 | 0.5949 | 0.4074 |
| F | 130 | 0.7846 | 0.3922 | 0.2263 | 0.6077 | 0.2609 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 106 | 81 |
| **True: Sarco**  | 2 | 20 |

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
