# ResNet1D Report — SMI

**Model:** ResNet1D_ClinicOnly_scaleY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 4.5299 | 0.5166 | 0.8000 | 0.7291 | 3 |
| 2 | 4.4424 | 0.4944 | 0.7818 | 0.6715 | 3 |
| 3 | 4.1224 | 0.5900 | 0.7818 | 0.7348 | 3 |
| 4 | 4.3087 | 0.5129 | 0.7534 | 0.6683 | 3 |
| 5 | 4.5305 | 0.4787 | 0.7717 | 0.6533 | 3 |
| **Mean** | **4.3868** | **0.5185** | **0.7778** | **0.6914** | **3.0** |
| **Std** | **0.1551** | **0.0382** | **0.0152** | **0.0337** | |

## Test Set 성능 (Test 20%)

Test R² = **0.4977**
Test MAE = **4.5214**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.7055 (p=1.026e-42) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.9470 |
| 이진화 ACC (성별 기준) | 0.7855 |
| 이진화 AUC (성별 기준) | 0.6809 |
| 이진화 AUPRC (성별 기준) | 0.8610 |
| 이진화 Brier Score (성별 기준) | 0.2682 |

## 피처 선택 목록 (Fold별)

### Fold 1 (3개)

PatientSex, PatientAge, BMI

### Fold 2 (3개)

PatientSex, PatientAge, BMI

### Fold 3 (3개)

PatientSex, PatientAge, BMI

### Fold 4 (3개)

PatientSex, PatientAge, BMI

### Fold 5 (3개)

PatientSex, PatientAge, BMI

### 최종 모델 (Train 전체, 3개)

PatientSex, PatientAge, BMI

