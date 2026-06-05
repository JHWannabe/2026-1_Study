# SMI Binary Classification — AECOnly Results

Generated: 2026-06-04 19:46  |  5-Fold CV  |  Median best epoch: 280

---

## 0. Dataset Distribution

### Class Distribution

| Split | Sex | n | Normal | Normal % | Sarco | Sarco % |
|-------|-----|--:|-------:|---------:|------:|--------:|
| Train | M | 324 | 274 | 84.6% | 50 | 15.4% |
| Train | F | 590 | 544 | 92.2% | 46 | 7.8% |
| Train | **All** | **914** | **818** | **89.5%** | **96** | **10.5%** |
| Test | M | 82 | 69 | 84.1% | 13 | 15.9% |
| Test | F | 147 | 136 | 92.5% | 11 | 7.5% |
| Test | **All** | **229** | **205** | **89.5%** | **24** | **10.5%** |

![Label Distribution](label_distribution.png)

---

## 1. Cross-Validation Summary

### AECOnly

| Fold | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|------|--------:|------:|------:|---------:|---:|
| 1 | 0.6098 | 0.1867 | 0.3113 | 0.5082 | 0.2500 |
| 2 | 0.7064 | 0.2589 | 0.2970 | 0.4809 | 0.2748 |
| 3 | 0.7019 | 0.2188 | 0.3068 | 0.6995 | 0.3373 |
| 4 | 0.6964 | 0.2181 | 0.2704 | 0.6448 | 0.3011 |
| 5 | 0.6416 | 0.1831 | 0.2919 | 0.7747 | 0.3279 |
| **Mean** | **0.6712** | **0.2131** | **0.2955** | **0.6216** | **0.2982** |
| **±Std** | 0.0386 | 0.0274 | 0.0143 | 0.1120 | 0.0325 |

AECOnly best val AUC per fold: Fold1=0.6098, Fold2=0.7064, Fold3=0.7019, Fold4=0.6964, Fold5=0.6416

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| AECOnly | 0.6006 | 0.2167 | 0.3071 | 0.4672 | 0.2179 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 82 | 0.6098 | 0.2306 | 0.2848 | 0.5610 | 0.3077 |
| F | 147 | 0.5976 | 0.2764 | 0.3196 | 0.4150 | 0.1731 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 90 | 115 |
| **True: Sarco**  | 7 | 17 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.6006 | 0.4708 | 0.7290 |
| AUPRC | 0.2167 | 0.1082 | 0.3825 |
| Brier | 0.3071 | 0.2950 | 0.3197 |
| Accuracy | 0.4672 | 0.4017 | 0.5328 |
| F1 | 0.2179 | 0.1333 | 0.3086 |

---

## 5. Figures

| File | Description |
|------|-------------|
| `label_distribution.png` | Train/Test class·sex distributions |
| `cv_roc_curves.png` | Per-fold ROC curves (AECOnly) |
| `cv_metric_distribution.png` | Boxplot of AUC / Acc / F1 across folds |
| `training_curves.png` | Loss & AUC training curves (mean ± std) |
| `test_roc_curves.png` | Final test-set ROC curve |
| `test_roc_by_sex.png` | Final test-set ROC curves by sex |
| `confusion_matrices.png` | Test-set confusion matrices |
| `calibration.png` | Calibration plot + Precision-Recall curve |
| `cam_aec_mean.png` | Grad-CAM mean ± std per class |
| `cam_aec_lines.png` | Grad-CAM individual samples per class |
| `cam_aec_heatmap.png` | Grad-CAM sample-level heatmap |
