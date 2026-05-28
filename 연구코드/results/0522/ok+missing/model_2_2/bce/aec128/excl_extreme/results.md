# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-22 00:27  |  5-Fold CV  |  Median best epoch: 6

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
| 1 | 0.8710 | 0.3997 | 0.2019 | 0.7305 | 0.4304 |
| 2 | 0.8522 | 0.4306 | 0.2679 | 0.8683 | 0.5600 |
| 3 | 0.8447 | 0.4190 | 0.0949 | 0.8024 | 0.4590 |
| 4 | 0.7988 | 0.3599 | 0.1474 | 0.8263 | 0.4528 |
| 5 | 0.8224 | 0.3768 | 0.1608 | 0.7425 | 0.3944 |
| **Mean** | **0.8378** | **0.3972** | **0.1746** | **0.7940** | **0.4593** |
| **±Std** | 0.0250 | 0.0261 | 0.0578 | 0.0516 | 0.0552 |

CrossAttn best val AUC per fold: Fold1=0.8710, Fold2=0.8522, Fold3=0.8447, Fold4=0.7988, Fold5=0.8224

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8230 | 0.3938 | 0.2283 | 0.6507 | 0.3423 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 79 | 0.8289 | 0.4165 | 0.2739 | 0.6203 | 0.4231 |
| F | 130 | 0.7998 | 0.3924 | 0.2006 | 0.6692 | 0.2712 |

---

## 3. Confusion Matrix (Test Set)

|   | Pred: Normal | Pred: Sarco |
|---|-------------:|------------:|
| **True: Normal** | 117 | 70 |
| **True: Sarco**  | 3 | 19 |

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
