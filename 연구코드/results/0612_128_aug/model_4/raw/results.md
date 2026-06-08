# SMI Binary Classification — AECOnly Results

Generated: 2026-06-08 18:27  |  5-Fold CV  |  Median best epoch: 389

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
| 1 | 0.6156 | 0.1805 | 0.3139 | 0.5137 | 0.2521 |
| 2 | 0.6733 | 0.2250 | 0.2916 | 0.4699 | 0.2707 |
| 3 | 0.7006 | 0.2065 | 0.3031 | 0.6995 | 0.3373 |
| 4 | 0.6906 | 0.2051 | 0.2701 | 0.6393 | 0.2979 |
| 5 | 0.6319 | 0.1692 | 0.2898 | 0.7692 | 0.3226 |
| **Mean** | **0.6624** | **0.1972** | **0.2937** | **0.6183** | **0.2961** |
| **±Std** | 0.0331 | 0.0199 | 0.0147 | 0.1120 | 0.0316 |

AECOnly best val AUC per fold: Fold1=0.6156, Fold2=0.6733, Fold3=0.7006, Fold4=0.6906, Fold5=0.6319

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| AECOnly | 0.6014 | 0.2251 | 0.3066 | 0.4498 | 0.2125 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 82 | 0.6254 | 0.2823 | 0.2843 | 0.5610 | 0.3077 |
| F | 147 | 0.5842 | 0.1865 | 0.3190 | 0.3878 | 0.1667 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 86 | 119 |
| **True: Sarco**  | 7 | 17 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.6014 | 0.4736 | 0.7282 |
| AUPRC | 0.2251 | 0.1097 | 0.4009 |
| Brier | 0.3066 | 0.2945 | 0.3188 |
| Accuracy | 0.4498 | 0.3885 | 0.5153 |
| F1 | 0.2125 | 0.1282 | 0.2994 |

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
