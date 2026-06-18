# AEC Virtual Phenotype Analysis — 구현 계획

## 핵심 목표

AEC curve를 low SMI binary classifier에 직접 넣지 않고,  
**AEC → virtual body-composition phenotype → low SMI prediction** 의 two-stage 구조로 분석.

## 분석 파이프라인

```
AEC curve (aec_1~aec_128)
    ↓ Feature Engineering (8-segment, global, DCT, relation, acquisition)
Stage 1: RF Regressor (multi-target)
    ↓ OOF predicted phenotype (TAMA, NAMA, IMATA, SMI, Weight, Height, BMI)
                        +
Clinical (Age, Sex, Weight, Height)  +  AEC-perceived discordance
    ↓
Stage 2: Logistic Regression → low SMI (binary)
```

## 파일 구조: `연구코드/code/virtual_phenotype/`

| 파일 | 역할 |
|------|------|
| `config.py` | 경로, 하이퍼파라미터, Yoon cutoff 등 상수 |
| `data.py` | metadata + aec_128 merge, NAMA 계산, Yoon/Q25 outcome |
| `features.py` | AECFeatureBuilder (8-seg, global, DCT, relation, acquisition) |
| `stage1.py` | RF/XGBoost multi-target regressor fit/predict/evaluate |
| `crossfit.py` | Outer 5-fold nested CV — OOF Stage 1 → Stage 2 전체 파이프라인 |
| `metrics.py` | AUC/AUPRC/Brier, calibration, paired bootstrap ΔAUC/p-value |
| `visualize.py` | Stage 1 scatter plot, ROC curve, calibration plot |
| `main.py` | 전체 orchestration + 결과 저장 |

## Cross-fitting 원칙 (§5 준수)

```
Outer 5-fold:
  1. outer train / outer val 분할
  2. outer train 내부에서 inner 5-fold CV
  3. inner CV로 outer train 환자들의 Stage 1 OOF predicted phenotype 생성
  4. outer train 전체로 Stage 1 재학습
  5. outer val 환자들의 predicted phenotype 생성
  6. outer train OOF predicted phenotype으로 Stage 2 학습
  7. outer val predicted phenotype으로 Stage 2 예측
  8. 모든 outer val 예측 집계 → 최종 OOF 성능
```

**금지**: 전체 데이터로 Stage 1 학습 후 같은 데이터의 prediction을 Stage 2에 사용

## Feature 구성 (§3)

| 그룹 | 개수 | 내용 |
|------|------|------|
| 8-segment | 32 | s1~s8 각각 mean, std, auc, slope |
| Global AEC | 9 | mean, std, cv, auc, min, max, range, skewness, kurtosis |
| Segment relation | 14 | upper/lower ratio, s1/s8 ratio, mirror diff, center_of_mass |
| DCT low-freq | 8 | dct_1 ~ dct_8 |
| Acquisition | 4+ | kVp, mAs, z_range, n_slices, scanner model one-hot |

**금지**: column-wise StandardScaler on raw AEC curve

## Stage 2 비교 모델 (§7)

| 모델 | 변수 |
|------|------|
| Model 1 (baseline) | Age + Sex + Weight + Height |
| Model 2 (AEC-added) | Model 1 + AEC_pred_TAMA/NAMA/IMATA/SMI + BMI/Weight/Height discordance |
| Sensitivity 1 | Age + Sex + BMI + Height |
| Sensitivity 2 | Age + Sex + Weight + Height + actual BMI |
| Negative control | Model 1 + shuffled AEC virtual phenotype |

## AEC-derived Variables for Stage 2 (§6-3)

```python
# Virtual muscle phenotype (Stage 1 OOF prediction)
AEC_predicted_TAMA, AEC_predicted_NAMA, AEC_predicted_IMATA, AEC_predicted_SMI

# Discordance (§6-3)
AEC_BMI_discordance    = AEC_predicted_BMI    - actual_BMI
AEC_Weight_discordance = AEC_predicted_Weight - actual_Weight
AEC_Height_discordance = AEC_predicted_Height - actual_Height

# Optional
AEC_predicted_NAMA_index = AEC_predicted_NAMA / actual_Height_m²
```

## Outcome 정의 (§2)

```python
NAMA = TAMA - IMATA
Height_m = Height_cm / 100
SMI = NAMA / Height_m²

# Primary: Yoon cutoff
low_SMI_yoon: Male SMI < 40.96, Female SMI < 30.60

# Secondary: sex-specific Q25
low_SMI_q25: 각 성별에서 SMI 하위 25%
```

## 평가 지표 (§8)

- AUC, AUPRC, Brier score (95% Bootstrap CI, n_boot=2000)
- Calibration intercept, slope, calibration plot
- **Paired bootstrap** ΔAUC, ΔAUPRC, ΔBrier with 95% CI + p-value
- DeLong test (Model 1 vs Model 2)

## Benchmark 목표 (§9, 강남 internal)

| Metric | Model 1 | Model 2 | Delta | 95% CI | p |
|--------|---------|---------|-------|--------|---|
| AUC | 0.807 | 0.826 | +0.019 | +0.001 ~ +0.037 | ~0.039 |
| AUPRC | 0.391 | 0.418 | — | — | — |
| Brier | 0.0790 | 0.0770 | — | — | — |

## 구현 체크리스트

- [x] plan.md 저장
- [x] config.py
- [x] data.py
- [x] features.py
- [x] stage1.py
- [x] crossfit.py
- [x] metrics.py
- [x] visualize.py
- [x] main.py
