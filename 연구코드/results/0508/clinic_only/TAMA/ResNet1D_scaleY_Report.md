# ResNet1D Report — TAMA

**Model:** ResNet1D_ClinicOnly_scaleY

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 12.3115 | 0.6635 | 0.7818 | 0.6749 | 3 |
| 2 | 11.8915 | 0.6801 | 0.7909 | 0.5647 | 3 |
| 3 | 11.2978 | 0.7183 | 0.7909 | 0.5806 | 3 |
| 4 | 12.7326 | 0.6545 | 0.7626 | 0.6732 | 3 |
| 5 | 11.8071 | 0.6998 | 0.7717 | 0.6833 | 3 |
| **Mean** | **12.0081** | **0.6832** | **0.7796** | **0.6353** | **3.0** |
| **Std** | **0.4849** | **0.0233** | **0.0111** | **0.0515** | |

## Test Set 성능 (Test 20%)

Test R² = **0.7116**
Test MAE = **12.7457**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.8509 (p=2.697e-78) |
| Shapiro-Wilk p | 0.0427 |
| Bias t-test p | 0.6146 |
| 이진화 ACC (성별 기준) | 0.7709 |
| 이진화 AUC (성별 기준) | 0.6492 |
| 이진화 AUPRC (성별 기준) | 0.8524 |
| 이진화 Brier Score (성별 기준) | 0.2843 |

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

