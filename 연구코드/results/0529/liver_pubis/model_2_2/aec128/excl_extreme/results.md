# SMI Binary Classification — CrossAttn Results

Generated: 2026-05-25 17:57  |  5-Fold CV  |  Median best epoch: 3

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 299 | 251 | 83.9% | 48 | 16.1% |
| Train | F | 537 | 494 | 92.0% | 43 | 8.0% |
| Train | **All** | **836** | **745** | **89.1%** | **91** | **10.9%** |
| Test | M | 77 | 66 | 85.7% | 11 | 14.3% |
| Test | F | 132 | 121 | 91.7% | 11 | 8.3% |
| Test | **All** | **209** | **187** | **89.5%** | **22** | **10.5%** |

### Age

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 299 | 59.85 ± 12.02 | 20.00 | 60.00 | 86.00 |
| Train | F | 537 | 55.17 ± 11.93 | 18.00 | 55.00 | 91.00 |
| Train | **All** | **836** | **56.84 ± 12.17** | **18.00** | **57.00** | **91.00** |
| Test | M | 77 | 59.21 ± 11.70 | 32.00 | 60.00 | 84.00 |
| Test | F | 132 | 55.37 ± 11.84 | 23.00 | 55.00 | 83.00 |
| Test | **All** | **209** | **56.78 ± 11.93** | **23.00** | **57.00** | **84.00** |

### BMI

| Split | Sex | n | Mean ± Std | Min | Median | Max |
|-------|-----|--:|----------:|----:|-------:|----:|
| Train | M | 299 | 24.44 ± 2.88 | 17.51 | 24.41 | 32.67 |
| Train | F | 537 | 23.11 ± 3.18 | 16.00 | 22.96 | 34.61 |
| Train | **All** | **836** | **23.58 ± 3.14** | **16.00** | **23.51** | **34.61** |
| Test | M | 77 | 24.52 ± 3.39 | 14.34 | 24.28 | 32.56 |
| Test | F | 132 | 22.99 ± 3.54 | 12.02 | 22.64 | 32.48 |
| Test | **All** | **209** | **23.55 ± 3.57** | **12.02** | **23.52** | **32.56** |

![Data Distribution](data_distribution.png)

---

## 1. Cross-Validation Summary

### CrossAttn

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.7842 | 0.4361 | 0.3612 | 0.7738 | 0.4242 |
| 2 | 0.7342 | 0.3407 | 0.1937 | 0.7006 | 0.3421 |
| 3 | 0.8881 | 0.5370 | 0.2206 | 0.7784 | 0.4789 |
| 4 | 0.8046 | 0.4359 | 0.1622 | 0.8263 | 0.4912 |
| 5 | 0.8326 | 0.3541 | 0.1441 | 0.7126 | 0.4000 |
| **Mean** | **0.8087** | **0.4208** | **0.2163** | **0.7584** | **0.4273** |
| **±Std** | 0.0511 | 0.0704 | 0.0770 | 0.0463 | 0.0543 |

CrossAttn best val AUC per fold: Fold1=0.7842, Fold2=0.7342, Fold3=0.8881, Fold4=0.8046, Fold5=0.8326

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| CrossAttn | 0.8155 | 0.3762 | 0.1473 | 0.8325 | 0.4068 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 77 | 0.8003 | 0.4065 | 0.1706 | 0.7922 | 0.5000 |
| F | 132 | 0.7994 | 0.3667 | 0.1337 | 0.8561 | 0.2963 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 162 | 25 |
| **True: Sarco**  | 10 | 12 |

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
