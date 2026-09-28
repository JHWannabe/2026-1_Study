# Clinic + AEC 모델 아키텍처 변화 기록

## 개요

CT 촬영 시 기록되는 AEC(자동 노출 제어) 곡선을 활용한 근감소증 진단 모델의 아키텍처 변화를 커밋 순서로 정리한다.

---

## 1단계: 최초 구현 — Cross-Attention 기반 딥러닝

**커밋**: `dca12f0b` (2026-05-28, liver pubis results)  
**파일**: `연구코드/code/model/models.py`

단일 `models.py`에 PyTorch 모델 3종이 공존.

| 모델 | 클래스 | 구조 |
|------|--------|------|
| M1 | `ResNet1D` | Clinic만 입력. stem → ResBlock×N → AdaptiveAvgPool → FC |
| M2 | `ClinAECCrossAttn` | Clinic + AEC. **Bidirectional Cross-Attention** |
| M3 | `ClinAECScanCrossAttn` | Clinic + Scanner(MFR Embedding) + AEC. M2에 Scanner 토큰 추가 |

### M2/M3 공통 서브모듈

- `ScalarFeatureTokenizer`: 각 Clinic scalar → `(B, F, d_model)` 독립 토큰
- `ResNet1DEncoder`: AEC 시퀀스 → `(B, n_tokens, d_model)` (1D Conv 인코딩)
- `CrossAttentionBlock` × 2방향:
  - 방향 1: Clinical → Query, AEC → Key/Value
  - 방향 2: AEC → Query, Clinical → Key/Value
- 최종: 양방향 mean-pool 후 concat → MLP classifier
- 손실함수: `BCEWithLogitsLoss(pos_weight)`

---

## 2단계: AEC Only 모델 추가 + FocalLoss 도입

**커밋**: `2114eb00` (2026-06-05, Input Data 재확인 & Only AEC Model 추가)

```
AECOnlyNet (M4) 추가 — ResNet1D와 동일 구조, 입력만 AEC 128포인트
손실함수 전환: BCEWithLogitsLoss → FocalLossWithLogits(gamma)
```

- `AECOnlyNet`: 임상 특징 없이 AEC 128포인트만으로 근감소증 직접 분류
- Focal Loss 도입으로 클래스 불균형 대응 강화

---

## 3단계: Late Fusion 비교 실험

**커밋**: `85e98357` (2026-06-10, late fusion 적용)

Cross-Attention 대비 단순 비교군으로 Late Fusion 2종 추가.

| 모델 | 클래스 | Cross-Attention과의 차이 |
|------|--------|--------------------------|
| M2_LF | `ClinAECLateFusion` | Cross-Attention **없이** mean-pool → concat → MLP |
| M3_LF | `ClinAECScanLateFusion` | Scanner 포함 Late Fusion |

두 모달리티를 독립 인코더로 처리 후 최종 표현만 결합하는 구조.

---

## 4단계: Late Fusion 제거

**커밋**: `63545902` (2026-06-12, late fusion 제거 및 aec 후반 crop)

```
M2_LF, M3_LF 전부 제거 (Cross-Attention 대비 열등)
build_cross_attn_feat: hand-crafted AEC feature 수 11 → 60 으로 변경
```

---

## 5단계: M1/M2/M5만 유지

**커밋**: `e278110e` (2026-06-16, Model 1,2,5만 유지)

```
M4 (AECOnlyNet) 제거
M3 (ClinAECScanCrossAttn + MfrTokenizer + QuadDataset) 전부 제거
ClinAECCrossAttn → M2/M5 공통으로 재지정
```

---

## 6단계: 전면 재설계 — 모듈별 디렉토리 분리

**커밋**: `682b1952` (2026-06-18, 코드 일괄 재정의)

`models.py` 삭제 후 `model_aec_128/` 디렉토리로 재구성.

| 클래스 | 구조 | 특징 |
|--------|------|------|
| `AECFusionModel` | AEC(ResNet1D) + Clinic(MLP) → concat → classifier | AvgPool + MaxPool 이중 풀링 |
| `GatedAECFusionModel` | AEC + Clinic에 학습된 스칼라 게이트 α 적용 | α→1: AEC 우세, α→0: Clinic 우세. 환자별 기여도 수치 해석 가능 |
| `AECCorrModel` | AEC 128-dim → 자기상관행렬(128×128) → 2D CNN | outer product로 slice 간 전역 공분산 학습 |

---

## 7단계: 현재 최종 — 통계 기반 워크플로 (완전 전환)

**파일**: `study.py`  
**커밋**: 최신 (`4b76ef3a` 이후)

PyTorch 딥러닝에서 **sklearn 기반 임상우선 + AEC 보조 워크플로**로 방향 전환.

### 전체 흐름

```
임상 모델 (LogReg)
    나이 + 키 + 체중 + 성별(M=1)
    → 임상 점수 (확률)
              ↓
    회색지대 판정: |임상점수 - Youden임계값| ≤ 0.075
              ↓ (회색지대 환자만)
국소 쌍대 AEC 모델 (Propensity Matched LogReg)
    AEC 128포인트 → 피처 엔지니어링 → 점수 ≥ 0.80 → 근감소증 재판정
```

### 임상 모델

- `LogisticRegression(C=0.7, class_weight="balanced")`
- 입력: Age, Height, Weight, Sex
- 5-fold Stratified CV → 폴드별 Youden 임계값 결정

### AEC 피처 엔지니어링

| 단계 | 내용 |
|------|------|
| `smooth_log` | `log1p` → Savitzky-Golay 스무딩 (window=9, poly=2) |
| `patientwise_robust_z` | 환자별 중앙값/IQR 기반 정규화 (스캐너 절대값 차이 제거) |
| `segment_features` | bins=2/8/16 구간별 mean/std/min/max/range/기울기 등 8종 |
| `inverse_features_from_x` | 전역 형태 기술자 22종 (무게중심, 3분위 평균, 오목·볼록 지수 등) |
| `dct_features_from_x` | 저주파 DCT 계수 12개 + 1차미분 DCT 8개 |

### 국소 쌍대 AEC 모델

1. **Propensity score 매칭**: 같은 성별 + 같은 스캐너 + logit 거리 ≤ caliper(σ×5%)
2. **학습**: 케이스–대조군 차이 벡터 (`x_case - x_ctrl`) → SelectKBest(k=20) → LogReg
3. **채점**: 테스트 환자를 이웃 K=3명(케이스·대조군)과 비교 → 쌍대 확률 평균
4. **배치 최적화**: predict_proba를 단 1회 호출 (전체 차이 벡터를 모아 일괄 처리)

### 최종 판정 규칙

```
임상 점수가 Youden 임계값 ±0.075 이내 (회색지대)
  AND AEC 점수 ≥ 0.80
  → 근감소증으로 재판정
나머지 → 임상 판단 유지
```

---

## 전체 변화 흐름 요약

```
BCELoss + Cross-Attention (양방향)                  [dca12f0b]
  → FocalLoss 도입 + AEC-Only 실험                  [2114eb00]
    → Late Fusion 비교 실험                         [85e98357]
      → Late Fusion 제거 (결과 불량)                [63545902]
        → M3(Scanner 융합) 제거, M1/M2/M5 정리       [e278110e]
          → 전면 재설계: Gating/2D Corr 실험         [682b1952]
            → 통계 기반 워크플로로 완전 전환          [study.py 현재]
```

## 이전 vs 현재 비교

| 항목 | 이전 (models.py) | 현재 (study.py) |
|------|-----------------|-----------------|
| 프레임워크 | PyTorch | sklearn |
| AEC 처리 | ResNet1D / Cross-Attention | Propensity 매칭 쌍대 LogReg |
| 융합 방식 | Bidirectional Cross-Attn → Late Fusion → Gating | 회색지대 환자에만 AEC 적용 |
| 임상 입력 | 토큰화 후 어텐션 | LogReg 직접 입력 |
| 해석 가능성 | 어텐션 맵 | 쌍대 확률, McNemar p-값 |
| 학습 복잡도 | GPU 필요, 에폭 학습 | 단순 fit, 재현성 높음 |
| 외부 검증 | 단일 코호트 | 강남(내부 CV) + 신촌(외부 검증) |
