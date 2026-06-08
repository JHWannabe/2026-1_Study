# SMI Binary Classification — AECOnly Results

Generated: 2026-06-08 18:39  |  5-Fold CV  |  Median best epoch: 129

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
| 1 | 0.6064 | 0.2238 | 0.2787 | 0.3934 | 0.2550 |
| 2 | 0.7243 | 0.2212 | 0.2645 | 0.6557 | 0.3368 |
| 3 | 0.7125 | 0.3586 | 0.2335 | 0.7869 | 0.4000 |
| 4 | 0.6845 | 0.2295 | 0.2909 | 0.6393 | 0.2826 |
| 5 | 0.6587 | 0.1941 | 0.3037 | 0.7912 | 0.3448 |
| **Mean** | **0.6773** | **0.2454** | **0.2742** | **0.6533** | **0.3239** |
| **±Std** | 0.0421 | 0.0579 | 0.0242 | 0.1446 | 0.0507 |

AECOnly best val AUC per fold: Fold1=0.6064, Fold2=0.7243, Fold3=0.7125, Fold4=0.6845, Fold5=0.6587

---

## 2. Test Set Performance

### Overall

| Model | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-------|--------:|------:|------:|---------:|---:|
| AECOnly | 0.6043 | 0.2076 | 0.3008 | 0.4236 | 0.2048 |

### By Sex

| Sex | n | AUC-ROC | AUPRC | Brier | Accuracy | F1 |
|-----|--:|--------:|------:|------:|---------:|---:|
| M | 82 | 0.6355 | 0.2412 | 0.2857 | 0.5122 | 0.2857 |
| F | 147 | 0.5916 | 0.2643 | 0.3092 | 0.3741 | 0.1636 |

---

## 3. Confusion Matrix (Test Set)

|  | Pred: Normal | Pred: Sarco |
|--|-------------:|------------:|
| **True: Normal** | 80 | 125 |
| **True: Sarco**  | 7 | 17 |

---

## 4. Bootstrap 95% CI  (n_boot=2000)

> Estimate: 원본 test set 기준. 95% CI: 2.5th–97.5th percentile.

| Metric | Estimate | CI Lower | CI Upper |
|--------|--------:|---------:|---------:|
| AUC-ROC | 0.6043 | 0.4819 | 0.7260 |
| AUPRC | 0.2076 | 0.1023 | 0.3578 |
| Brier | 0.3008 | 0.2894 | 0.3124 |
| Accuracy | 0.4236 | 0.3624 | 0.4891 |
| F1 | 0.2048 | 0.1234 | 0.2892 |

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
