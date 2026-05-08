# ResNet1D Report — TAMA

**Model:** ResNet1D_ClinicOnly_scaleX

## 5-Fold CV 성능 (Train 80%)

| Fold | MAE | R² | ACC | AUC | 피처 수 |
|---|---|---|---|---|---|
| 1 | 12.1993 | 0.6782 | 0.7864 | 0.6684 | 3 |
| 2 | 11.9418 | 0.6785 | 0.7818 | 0.5582 | 3 |
| 3 | 11.0996 | 0.7182 | 0.8000 | 0.5724 | 3 |
| 4 | 13.2280 | 0.6055 | 0.7808 | 0.6674 | 3 |
| 5 | 11.9102 | 0.6945 | 0.7534 | 0.6804 | 3 |
| **Mean** | **12.0758** | **0.6750** | **0.7805** | **0.6294** | **3.0** |
| **Std** | **0.6843** | **0.0377** | **0.0152** | **0.0527** | |

## Test Set 성능 (Test 20%)

Test R² = **0.7169**
Test MAE = **12.5709**

## 통계 분석 (성별별 임계값 적용)

| 지표 | 값 |
|---|---|
| 남성 임계값 (25th pct) | 130.00 |
| 여성 임계값 (25th pct) | 95.00 |
| Pearson r | 0.8488 (p=1.525e-77) |
| Shapiro-Wilk p | 0.0231 |
| Bias t-test p | 0.0759 |
| 이진화 ACC (성별 기준) | 0.7782 |
| 이진화 AUC (성별 기준) | 0.6329 |
| 이진화 AUPRC (성별 기준) | 0.8464 |
| 이진화 Brier Score (성별 기준) | 0.3620 |

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

