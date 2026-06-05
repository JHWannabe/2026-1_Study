# SMI Binary Classification — AECOnly Results

Generated: 2026-06-04 19:52  |  5-Fold CV  |  Median best epoch: 134

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
| 1 | 0.6071 | 0.1531 | 0.2737 | 0.7213 | 0.3014 |
| 2 | 0.5308 | 0.1230 | 0.3050 | 0.7760 | 0.2264 |
| 3 | 0.6271 | 0.1703 | 0.3097 | 0.5464 | 0.2523 |
| 4 | 0.5546 | 0.1309 | 0.2973 | 0.4262 | 0.2336 |
| 5 | 0.5082 | 0.1329 | 0.3148 | 0.8681 | 0.2000 |
| **Mean** | **0.5655** | **0.1421** | **0.3001** | **0.6676** | **0.2427** |
| **±Std** | 0.0450 | 0.0173 | 0.0144 | 0.1599 | 0.0338 |

AECOnly best val AUC per fold: Fold1=0.6071, Fold2=0.5308, Fold3=0.6271, Fold4=0.5546, Fold5=0.5082

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| AECOnly | 0.5392 | 0.1199 | 0.3005 | 0.8297 | 0.0488 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 82 | 0.5875 | 0.2110 | 0.2941 | 0.7683 | 0.0952 |
| F | 147 | 0.4418 | 0.0699 | 0.3040 | 0.8639 | 0.0000 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 189 | 16 |
| **True: Sarco**  | 23 | 1 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.5392 | 0.4215 | 0.6526 |
| AUPRC | 0.1199 | 0.0765 | 0.2001 |
| Brier | 0.3005 | 0.2954 | 0.3049 |
| Accuracy | 0.8297 | 0.7773 | 0.8777 |
| F1 | 0.0488 | 0.0000 | 0.1538 |

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
