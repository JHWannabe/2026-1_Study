# ResNet1D Report — SMI

**Model:** ResNet1D_ClinicOnly_scaleXY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 4.3945 | 0.5187 | 0.7773 | 0.7257 | 3 |
| 2 | 4.4851 | 0.4892 | 0.7682 | 0.6383 | 3 |
| 3 | 4.1760 | 0.5920 | 0.7864 | 0.7373 | 3 |
| 4 | 4.3681 | 0.4907 | 0.7626 | 0.6731 | 3 |
| 5 | 4.5251 | 0.4812 | 0.7626 | 0.6329 | 3 |
| **Mean** | **4.3898** | **0.5144** | **0.7714** | **0.6815** | **3.0** |
| **Std** | **0.1213** | **0.0408** | **0.0092** | **0.0433** | |

## Test Set 성능 (Test 20%)

Test R² = **0.5012**
Test MAE = **4.5415**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 45.81 |
| 여성 임계값 (25th pct) | 37.77 |
| Pearson r | 0.7082 (p=3.657e-43) |
| Shapiro-Wilk p | 0.0000 |
| Bias t-test p | 0.8208 |
| 이진화 ACC (성별 기준) | 0.7636 |
| 이진화 AUC (성별 기준) | 0.6893 |
| 이진화 AUPRC (성별 기준) | 0.8645 |
| 이진화 Brier Score (성별 기준) | 0.4006 |

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

