# SMI Binary Classification — AECOnly Results

Generated: 2026-06-11 13:55  |  5-Fold CV  |  Median best epoch: 377

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
| 1 | 0.6325 | 0.1794 | 0.3191 | 0.5246 | 0.2564 |
| 2 | 0.6868 | 0.2216 | 0.2995 | 0.4863 | 0.2656 |
| 3 | 0.7076 | 0.2175 | 0.3079 | 0.7049 | 0.3415 |
| 4 | 0.6996 | 0.2732 | 0.2924 | 0.6940 | 0.3000 |
| 5 | 0.6697 | 0.3068 | 0.3031 | 0.7747 | 0.3279 |
| **Mean** | **0.6792** | **0.2397** | **0.3044** | **0.6369** | **0.2983** |
| **±Std** | 0.0267 | 0.0449 | 0.0089 | 0.1115 | 0.0334 |

AECOnly best val AUC per fold: Fold1=0.6325, Fold2=0.6868, Fold3=0.7076, Fold4=0.6996, Fold5=0.6697

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| AECOnly | 0.5856 | 0.1882 | 0.3041 | 0.6507 | 0.2157 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 82 | 0.6243 | 0.3019 | 0.2806 | 0.6707 | 0.3077 |
| F | 147 | 0.5635 | 0.0946 | 0.3172 | 0.6395 | 0.1587 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 138 | 67 |
| **True: Sarco**  | 13 | 11 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.5856 | 0.4544 | 0.7105 |
| AUPRC | 0.1882 | 0.1021 | 0.3675 |
| Brier | 0.3041 | 0.2917 | 0.3164 |
| Accuracy | 0.6507 | 0.5852 | 0.7118 |
| F1 | 0.2157 | 0.1111 | 0.3273 |

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
