# `core_resnet1d.py` 코드 실행 과정 — 연구적 상세 설명

> 본 문서는 `core_resnet1d.py`의 전체 실행 흐름을 연구 관점에서 단계별로 해설합니다.  
> 대상 태스크: **표 형식(tabular) 임상 데이터로부터 근육량 지표(SMI 등)를 회귀 예측**하고, 성별별 임계값 기반으로 근감소증(Sarcopenia)을 이진 분류하는 이중 평가 파이프라인.  
> **v2 추가**: X·y StandardScaler 4가지 조합 비교 실험, AUPRC·Brier Score 지표, 8패널 시각화.

---

## 목차

1. [전역 설정 및 재현성 확보](#1-전역-설정-및-재현성-확보)
2. [데이터 파이프라인](#2-데이터-파이프라인)
3. [모델 아키텍처: ResNet1D](#3-모델-아키텍처-resnet1d)
   - 3-1. ResBlock1D — 잔차 블록
   - 3-2. ResNet1D — 전체 네트워크
4. [학습 루프](#4-학습-루프)
   - 4-1. 에폭 단위 학습
   - 4-2. 에폭 단위 평가
   - 4-3. `_fit` — 전체 학습 관리자
5. [성별별 임계값 로직](#5-성별별-임계값-로직)
6. [5-Fold 교차 검증 + 최종 모델 학습](#6-5-fold-교차-검증--최종-모델-학습)
   - 6-1. 전체 분할 구조
   - 6-2. X·y StandardScaler 4가지 조합 비교
   - 6-3. 조합별 상세 설명
   - 6-4. 피처 선택과 스케일링의 분리
   - 6-5. 최종 모델 학습
7. [결과 저장 및 시각화](#7-결과-저장-및-시각화)
8. [전체 실행 흐름 요약 다이어그램](#8-전체-실행-흐름-요약-다이어그램)
9. [설계 선택의 연구적 의미](#9-설계-선택의-연구적-의미)

---

## 1. 전역 설정 및 재현성 확보

```python
BATCH_SIZE = 32
EPOCHS     = 500
LR         = 1e-3
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
```

| 하이퍼파라미터 | 값 | 의미 |
|---|---|---|
| BATCH_SIZE | 32 | 미니배치 크기. 소규모 임상 데이터셋에서 gradient noise를 적절히 유지 |
| EPOCHS | 500 | 최대 반복 횟수. 실제 최적 모델은 검증 손실 기반 early-save로 결정됨 |
| LR | 1e-3 | AdamW 초기 학습률. CosineAnnealing으로 1e-6까지 감소 |
| SEED | 42 | PyTorch + NumPy 시드 고정으로 fold split, 가중치 초기화, 셔플 모두 재현 가능 |

**연구적 의미**: 임상 연구에서 재현성(reproducibility)은 필수 요건입니다. 두 라이브러리의 시드를 모두 고정하면 동일 환경에서 동일 결과를 보장합니다.

---

## 2. 데이터 파이프라인

### `_TabDataset`

```python
class _TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
```

- NumPy 배열을 PyTorch `float32` 텐서로 변환
- `y`에 `.unsqueeze(1)` → shape `(N,)` → `(N, 1)` : MSELoss 계산 시 broadcast 오류 방지
- `__getitem__`은 단순 인덱싱으로 DataLoader의 배치 조립에 사용됨

### `_loader`

```python
def _loader(X, y, shuffle):
    return DataLoader(_TabDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)
```

- **학습 시** `shuffle=True` → 매 에폭 순서 무작위화, 과적합 방지
- **평가 시** `shuffle=False` → 예측값과 실제값의 인덱스 일치 보장

---

## 3. 모델 아키텍처: ResNet1D

### 3-1. `ResBlock1D` — 잔차 블록

```
Input x ──────────────────────────────────────┐
  │                                            │ (shortcut)
  ├─ Conv1d(k=3, pad=1) → BN → ReLU          │
  ├─ Conv1d(k=3, pad=1) → BN                 │
  └──────────────────── + ←──────────────────┘
                         │
                       ReLU
                       Output
```

**핵심 설계 포인트**:

| 요소 | 값 | 역할 |
|---|---|---|
| Conv1d kernel=3, padding=1 | — | 시퀀스 길이 유지 (same-padding) |
| BatchNorm1d | — | 내부 공분산 이동(Internal Covariate Shift) 억제, 학습 안정화 |
| Shortcut (downsample) | 1×1 Conv + BN | `stride≠1` 또는 채널 불일치 시만 적용, 나머지는 항등 사상 |
| ReLU(inplace=True) | — | 메모리 절약 (in-place 연산) |

**잔차 연결의 연구적 의미**: 기울기 소실(Vanishing Gradient) 문제를 완화하며, 깊은 네트워크에서 "아무것도 안 하는 것"이 기본값이 되어 학습 초기 안정성 향상. 표형식 데이터를 1D 시퀀스로 취급할 때 특징 간 지역적 상호작용을 학습하는 귀납적 편향 제공.

### 3-2. `ResNet1D` — 전체 네트워크

```
Input: (N, 1, num_features)   ← 1채널 1D 시퀀스로 간주
    │
  [Stem]
    Conv1d(1→64, k=7, stride=2, pad=3) → BN → ReLU
    MaxPool1d(k=3, stride=2, pad=1)
    └─ 시퀀스 길이 ÷ 4
    │
  [Stage 1]  ResBlock1D×2  (64→64,   stride=1)
  [Stage 2]  ResBlock1D×2  (64→128,  stride=2)  ← 해상도 ½
  [Stage 3]  ResBlock1D×2  (128→256, stride=2)  ← 해상도 ½
  [Stage 4]  ResBlock1D×2  (256→512, stride=2)  ← 해상도 ½
    │
  [Head]
    AdaptiveAvgPool1d(1)  → (N, 512, 1)
    Flatten()             → (N, 512)
    Linear(512→256) → ReLU
    Dropout(0.3)
    Linear(256→1)         → 최종 스칼라 예측값
```

**채널 구성 (base=64)**:

| Stage | In channels | Out channels | Stride | 역할 |
|---|---|---|---|---|
| Stem | 1 | 64 | 2 | 저수준 특징 추출, 초기 다운샘플링 |
| Stage 1 | 64 | 64 | 1 | 동일 해상도에서 특징 정제 |
| Stage 2 | 64 | 128 | 2 | 공간 압축 + 채널 확장 |
| Stage 3 | 128 | 256 | 2 | 고수준 추상화 |
| Stage 4 | 256 | 512 | 2 | 최상위 표현 |

**Dropout(0.3)**: 분류 헤드에만 적용. 임상 소규모 데이터에서 과적합 방지의 핵심 정규화 수단.

**AdaptiveAvgPool1d(1)**: 입력 시퀀스 길이(피처 수)에 무관하게 고정된 벡터 출력 → 서로 다른 피처 셋으로 학습된 fold 모델들이 동일 헤드 구조를 공유 가능.

---

## 4. 학습 루프

### 4-1. `_train_epoch` — 에폭 단위 학습

```python
def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()           # Dropout·BN을 학습 모드로 전환
    for X_b, y_b in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)  # MSELoss
        loss.backward()                    # 역전파
        optimizer.step()                   # 가중치 업데이트
    return total / len(loader.dataset)     # 배치 가중 평균 손실 반환
```

- `model.train()` 호출로 Dropout 활성화 + BN이 미니배치 통계 사용
- `optimizer.zero_grad()` 위치: 배치 루프 시작 → 기울기 누적 방지

### 4-2. `_eval_epoch` — 에폭 단위 평가

```python
@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()   # Dropout 비활성화, BN이 학습된 running 통계 사용
    ...
    return loss, preds_array, trues_array
```

- `@torch.no_grad()`: 그래디언트 계산 그래프 비생성 → 메모리 절약 + 추론 속도 향상
- 배치별 예측값 누적 후 `np.concatenate`로 합쳐 전체 예측 배열 반환

### 4-3. `_fit` — 전체 학습 관리자

```python
def _fit(tr_loader, vl_loader, model, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    best_loss, best_state = float('inf'), None
    for _ in range(EPOCHS):
        _train_epoch(...)
        vl, _, _ = _eval_epoch(...)
        scheduler.step()
        if vl < best_loss:          # 검증 손실 기반 Best Model 저장
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    return best_state, criterion
```

**옵티마이저 선택 — AdamW**:
- Adam에 Weight Decay를 올바르게 분리(decoupled) 적용
- `weight_decay=1e-2`: L2 정규화로 과적합 억제
- Adam의 적응적 학습률 + 정규화의 조합 → 소규모 임상 데이터에 적합

**스케줄러 — CosineAnnealingLR**:

```
LR
1e-3 ─┐
      │  ╲
      │    ╲___
      │        ╲____
1e-6  └─────────────────→ EPOCH 500
```

- 학습 초기: 높은 LR로 손실 공간을 빠르게 탐색
- 학습 후기: 낮은 LR로 sharp minima를 피하고 flat minima 수렴 → 일반화 성능 향상
- `T_max=EPOCHS`: 전체 학습 기간을 주기 반주기로 사용

**Best Model 저장 (Implicit Early Stopping)**:
- 별도의 early stopping patience 없이, 전체 500 에폭을 돌면서 검증 손실 최소 시점의 가중치를 `.clone()`으로 복사 보관
- 학습 완료 후 best 가중치를 모델에 `load_state_dict`으로 복원 → 과적합 이전 최적 상태 사용

---

## 5. 성별별 임계값 로직

```python
def _get_gender_thresholds(trues, sex_arr):
    # PatientSex: 0=남성, 1=여성
    thr_m = np.percentile(trues[sex_arr == 0], 25)
    thr_f = np.percentile(trues[sex_arr == 1], 25)
    sample_thrs = np.where(sex_arr == 0, thr_m, thr_f)
    return thr_m, thr_f, sample_thrs
```

**연구적 배경**:

근감소증(Sarcopenia) 진단 기준(예: AWGS 2019)은 남녀의 근육량 기준값이 다릅니다.

- 남성의 정상 근육량 범위가 여성보다 높기 때문에, **동일한 절대적 임계값 사용은 진단 편향을 초래**합니다.
- 이 코드는 각 성별의 학습 데이터 분포에서 **하위 25번째 백분위수를 성별 특이적 임계값**으로 사용합니다.

```
남성 분포:       [──────|──────────────────]
                       ↑ thr_m (25th pct)

여성 분포:  [──────|────────────────────]
                  ↑ thr_f (25th pct)
```

**이진화 규칙**:
- `SMI < 임계값` → 클래스 0 (Low, 근감소증 위험)
- `SMI ≥ 임계값` → 클래스 1 (Normal)

**핵심 구현 세부사항**:
- `np.where(sex_arr == 0, thr_m, thr_f)`: 각 샘플에 성별에 맞는 임계값을 벡터 연산으로 일괄 할당
- Fold별 ACC/AUC 계산 시에도 **해당 fold의 학습 분할에서 산출한 임계값** 사용 → 데이터 누수(data leakage) 방지

---

## 6. 5-Fold 교차 검증 + 최종 모델 학습

### 6-1. 전체 분할 구조

```
전체 데이터 (N개)
    │
    ├── Train (80%, N×0.8)   ←── 5-Fold CV 수행
    │       │
    │       ├── Fold 1: Train 64% / Val 16%
    │       ├── Fold 2: Train 64% / Val 16%
    │       ├── Fold 3: Train 64% / Val 16%
    │       ├── Fold 4: Train 64% / Val 16%
    │       └── Fold 5: Train 64% / Val 16%
    │
    └── Test  (20%, N×0.2)   ←── 최종 모델 평가 전용 (CV 과정에서 미접촉)
```

---

### 6-2. X·y StandardScaler 4가지 조합 비교

현재 코드는 `SCALE_CONFIGS`에 정의된 4가지 조합을 **모두 자동으로 학습·평가**합니다.  
피처 선택은 스케일링 여부와 무관하므로 fold당 1회만 수행해 4가지 조합이 공유합니다.

```python
SCALE_CONFIGS = [
    ('noScale', False, False),   # X 미적용, y 미적용
    ('scaleX',  True,  False),   # X 적용,   y 미적용
    ('scaleY',  False, True),    # X 미적용, y 적용
    ('scaleXY', True,  True),    # X 적용,   y 적용
]
```

#### 4가지 조합 한눈에 비교

| 구분 | scale_X | scale_y | 출력 파일 | 특징 |
|---|:---:|:---:|---|---|
| **noScale** | ✗ | ✗ | `ResNet1D_Report.md` / `ResNet1D_Results.png` | 베이스라인. 역변환 없음. 기존 파일명 유지 |
| **scaleX** | ✓ | ✗ | `ResNet1D_scaleX_Report.md` / `ResNet1D_scaleX_Results.png` | 피처 스케일 통일. y는 원래 스케일 유지 |
| **scaleY** | ✗ | ✓ | `ResNet1D_scaleY_Report.md` / `ResNet1D_scaleY_Results.png` | 손실 스케일 정규화. 예측 후 y 역변환 필요 |
| **scaleXY** | ✓ | ✓ | `ResNet1D_scaleXY_Report.md` / `ResNet1D_scaleXY_Results.png` | X·y 모두 표준화. 역변환 후 원래 단위로 평가 |

> `noScale`은 기존 `5_generate_plots.py`와의 하위 호환성을 위해 원래 파일명을 사용합니다.

---

### 6-3. 조합별 상세 설명

#### noScale — X·y 모두 StandardScaler 미적용

```python
# X: shape 변환만 수행
X_tr = X_tr_raw[:, np.newaxis, :]
X_vl = X_vl_raw[:, np.newaxis, :]

# y: 원래 스케일 그대로 사용
# 예측값: 역변환 없이 바로 사용
preds = ps
```

| 대상 | 효과 | 주의점 |
|---|---|---|
| X | ResNet1D 내부 BatchNorm1d가 레이어별 정규화를 담당하므로 외부 스케일링 의존도 낮음 | 피처 간 스케일 차이가 매우 클 경우 초기 Conv 레이어에 영향 가능 |
| y | 코드 단순화, 역변환 오류 가능성 없음 | SMI 범위(~5~50)가 크면 MSELoss 값도 커져 학습률 민감도 증가 |

**연구적 위치**: 모든 정규화를 제거한 순수 베이스라인. 나머지 3가지 조합의 비교 기준.

---

#### scaleX — X만 StandardScaler 적용

```python
if scale_X:
    sc_X = StandardScaler().fit(X_tr_raw)   # fold 학습셋에만 fit
    X_tr_raw = sc_X.transform(X_tr_raw)
    X_vl_raw = sc_X.transform(X_vl_raw)    # 검증셋은 transform만

# y: 원래 스케일 그대로
```

| 대상 | 효과 | 주의점 |
|---|---|---|
| X | 피처 간 스케일 통일 → 초기 Conv 가중치 수렴 안정화 | fold별 fit이므로 fold 간 X 스케일 기준 상이 |
| y | 원래 단위 유지 → 손실 값의 직관적 해석 가능 | noScale과 손실 스케일 동일, y 관련 수렴 특성 불변 |

**연구적 의미**: X 정규화가 BatchNorm과 어떻게 상호작용하는지 분리 평가. BatchNorm이 있는 경우 scaleX와 noScale의 성능 차이가 작을 수 있음.

---

#### scaleY — y만 StandardScaler 적용

```python
if scale_y:
    sc_y = StandardScaler().fit(y_fold_tr.reshape(-1, 1))  # fold 학습셋에만 fit
    y_tr_s = sc_y.transform(y_fold_tr.reshape(-1, 1)).ravel()
    y_vl_s = sc_y.transform(y_fold_vl.reshape(-1, 1)).ravel()

# 예측 후 반드시 역변환
if sc_y is not None:
    ps = sc_y.inverse_transform(ps.reshape(-1, 1)).ravel()
    ts = y_fold_vl   # 역변환된 예측 vs 원래 y로 지표 계산
```

| 대상 | 효과 | 주의점 |
|---|---|---|
| X | 미처리 (noScale과 동일한 X 입력) | — |
| y | MSELoss를 ~N(0,1) 스케일로 제한 → 학습률 민감도 대폭 감소, 안정적 수렴 | **fold별 sc_y를 fit** → fold 간 y 표준화 기준이 다름. 최종 모델은 `y_train_all` 전체에 fit |

**역변환 흐름**:
```
y_fold_tr (원래 단위)
    → sc_y.fit_transform → y_tr_s (표준화, 학습에 사용)
    → 모델 예측: ps (표준화 공간)
    → sc_y.inverse_transform → ps (원래 단위, 지표 계산에 사용)
```

**연구적 의미**: y의 범위가 넓을수록(SMI: ~5~50, TAMA: 더 큰 범위) MSELoss 값이 커져 학습이 불안정해질 수 있습니다. scaleY는 이를 해소하는 가장 직접적인 방법입니다.

---

#### scaleXY — X·y 모두 StandardScaler 적용

```python
# X: fold 학습셋으로 fit, 검증/테스트는 transform
if scale_X:
    sc_X = StandardScaler().fit(X_tr_raw)
    X_tr_raw = sc_X.transform(X_tr_raw)
    X_vl_raw = sc_X.transform(X_vl_raw)

# y: fold 학습셋으로 fit
if scale_y:
    sc_y = StandardScaler().fit(y_fold_tr.reshape(-1, 1))
    y_tr_s = sc_y.transform(...).ravel()

# 예측 후 y 역변환
ps = sc_y.inverse_transform(ps.reshape(-1, 1)).ravel()
```

| 대상 | 효과 | 주의점 |
|---|---|---|
| X | scaleX와 동일 효과: 피처 스케일 통일 | fold별 기준 상이 |
| y | scaleY와 동일 효과: 손실 정규화 | fold별 기준 상이. 역변환 필수 |

**연구적 의미**: 두 정규화의 효과가 독립적으로 더해지는지, 아니면 상호작용이 있는지 확인. 일반적으로 딥러닝에서 가장 권장되는 전처리 조합이나, BatchNorm이 있는 경우 scaleY 단독과 유사한 성능을 보일 수 있음.

---

### 6-4. 피처 선택과 스케일링의 분리

```python
# 피처 선택: 4가지 조합 전에 1회만 수행 (스케일링 무관)
fold_feat_lists = []
for fold_i, (tr_idx, vl_idx) in enumerate(fold_splits, 1):
    sel = feature_selector(df_train_all.iloc[tr_idx])  # 원래 스케일 데이터
    fold_feat_lists.append(list(sel))

# 이후 4가지 SCALE_CONFIGS 루프에서 fold_feat_lists를 재사용
for scale_name, scale_X, scale_y in SCALE_CONFIGS:
    for fold_i, (tr_idx, vl_idx) in enumerate(fold_splits, 1):
        sel_feats = fold_feat_lists[fold_i - 1]   # 피처 선택 결과 재사용
        # ... 스케일링 적용 후 학습
```

**설계 의도**: 피처 선택(상관계수 필터 + VIF 제거)은 원래 스케일 데이터에서 수행합니다. 스케일링은 그 이후의 신경망 학습 단계에만 영향을 주므로, 4가지 조합이 **동일한 피처 집합을 공유**하여 정규화 효과만 순수하게 비교할 수 있습니다.

| 단계 | 의존성 | 수행 횟수 |
|---|---|---|
| 피처 선택 (상관계수·VIF) | 원래 스케일 데이터 | fold당 1회 (4가지 조합 공유) |
| X StandardScaler | 선택된 피처의 학습셋 | fold당 1회 (scale_X=True일 때만) |
| y StandardScaler | fold의 y 학습셋 | fold당 1회 (scale_y=True일 때만) |

---

### 6-5. 최종 모델 학습

각 스케일 조합에 대해 최종 모델을 독립적으로 학습합니다.

```python
for scale_name, scale_X, scale_y in SCALE_CONFIGS:
    # 최종 모델: train 전체로 학습 → test 평가
    X_tr_raw = df_train_all[sel_feats_final].values.astype(float)
    X_te_raw = df_test_all[sel_feats_final].values.astype(float)

    if scale_X:
        sc_X_final = StandardScaler().fit(X_tr_raw)   # train 전체에 fit
        X_tr_raw = sc_X_final.transform(X_tr_raw)
        X_te_raw = sc_X_final.transform(X_te_raw)     # test는 transform만

    if scale_y:
        sc_y_final = StandardScaler().fit(y_train_all.reshape(-1, 1))
        y_tr_s = sc_y_final.transform(...).ravel()
        # 예측 후 역변환
        ps_te = sc_y_final.inverse_transform(ps_te.reshape(-1, 1)).ravel()
```

**scaler fit 원칙** (데이터 누수 방지):

| Scaler | fit 데이터 | 이유 |
|---|---|---|
| X fold scaler | 해당 fold의 학습 분할만 | 검증셋 정보 노출 방지 |
| y fold scaler | 해당 fold의 y 학습 분할만 | 검증셋 y 분포 노출 방지 |
| X final scaler | `df_train_all` 전체 | test set 정보 노출 방지 |
| y final scaler | `y_train_all` 전체 | test set y 분포 노출 방지 |

**주의사항**: 최종 모델의 `_fit`에서 validation loader에 test 데이터를 사용합니다. 이는 최종 best state 선택의 기준으로만 사용되며, **실제 성능 지표는 이 과정 이후 별도 `_eval_epoch`에서 계산**됩니다. 이 설계는 test set의 레이블을 학습에 사용하지 않되, early stopping 신호로는 활용하는 실용적 타협점입니다.

---

## 7. 결과 저장 및 시각화

### `_save_md` — 마크다운 보고서 생성

각 스케일 조합마다 독립된 보고서 파일이 생성됩니다.

| 조합 | 보고서 파일 |
|---|---|
| noScale | `ResNet1D_Report.md` |
| scaleX | `ResNet1D_scaleX_Report.md` |
| scaleY | `ResNet1D_scaleY_Report.md` |
| scaleXY | `ResNet1D_scaleXY_Report.md` |

보고서 구조:

```
# ResNet1D Report — {target}

## 5-Fold CV 성능 (Train 80%)
| Fold | MAE | R² | ACC | AUC | 피처 수 |
| ...  |

## Test Set 성능 (Test 20%)
Test R²  = ...
Test MAE = ...

## 통계 분석 (성별별 임계값 적용)
| 남성 임계값 (25th pct)        | ...
| 여성 임계값 (25th pct)        | ...
| Pearson r                     | ...
| Shapiro-Wilk p                | ...
| Bias t-test p                 | ...
| 이진화 ACC (성별 기준)        | ...
| 이진화 AUC (성별 기준)        | ...
| 이진화 AUPRC (성별 기준)      | ...   ← 추가
| 이진화 Brier Score (성별 기준)| ...   ← 추가

## 피처 선택 목록 (Fold별)
```

**AUPRC 계산 방법**: 회귀 예측값을 `[0, 1]`로 정규화한 스코어(`score_norm`)를 분류 확률 대용으로 사용해 `average_precision_score`를 계산합니다.

```python
score_norm = (preds - preds.min()) / (np.ptp(preds) + 1e-8)
test_auprc = average_precision_score(y_bin, score_norm)
test_brier = brier_score_loss(y_bin, score_norm)
```

---

### `_plot_oof` — 8패널 시각화 (`ResNet1D_Results.png`)

각 스케일 조합마다 독립된 시각화 파일이 생성됩니다 (noScale은 `ResNet1D_Results.png`).

```
┌──────────────────────────────────────────────────────────────────┐
│  (1,1)              (1,2)              (1,3)              (1,4)  │
│  Predicted vs True  Residual Dist.     Bland-Altman       ROC    │
│                                        Plot               Curve  │
├──────────────────────────────────────────────────────────────────┤
│  (2,1)              (2,2)              (2,3)              (2,4)  │
│  Confusion Matrix   Q-Q Plot           PR Curve           Calib. │
│                                                           Plot   │
└──────────────────────────────────────────────────────────────────┘
```

| 패널 | 위치 | 내용 | 연구적 의미 |
|---|---|---|---|
| Predicted vs True | (1,1) | 예측값 vs 실제값 산점도 + y=x 직선 | 회귀 정확도 직관적 확인, r·R² 표기 |
| Residual Distribution | (1,2) | 잔차 히스토그램 | 오차의 분포 형태 확인 (정규성) |
| Bland-Altman Plot | (1,3) | (평균, 차이) 산점도 + ±1.96SD | 측정 일치도 평가 (임상 표준 방법론) |
| ROC Curve | (1,4) | FPR vs TPR, AUC | 성별별 임계값 기반 이진 분류 성능 |
| Confusion Matrix | (2,1) | TP/TN/FP/FN | 근감소증 탐지의 민감도/특이도 분해 |
| Q-Q Plot | (2,2) | 이론 vs 샘플 분위수 | 잔차 정규성 시각적 검증 (Shapiro-Wilk 보완) |
| **PR Curve** | **(2,3)** | **Recall vs Precision, AUPRC + 기저선** | **클래스 불균형 시 ROC보다 정보량이 많은 지표** |
| **Calibration Plot** | **(2,4)** | **예측 확률 vs 실제 양성 비율** | **모델이 확률을 얼마나 신뢰할 수 있는지 평가** |

**Bland-Altman Plot의 연구적 중요성**: 임상 측정 도구 비교에서 단순 상관관계보다 중요하게 쓰이는 방법론입니다. 예측 오차가 측정값의 크기에 따라 편향되는지(비례 오차)를 검출할 수 있습니다. `±1.96SD` 한계선 밖의 점은 임상적으로 의미 있는 오차를 보이는 케이스입니다.

**PR Curve와 Calibration Plot 추가 배경**:

| 지표 | 추가 이유 | 해석 기준 |
|---|---|---|
| AUPRC | 근감소증은 하위 25% 기준이므로 **클래스 불균형**(양성 25%)이 존재. 불균형 데이터에서 ROC AUC는 낙관적으로 나타날 수 있으므로 AUPRC를 병용 | 기저선(양성 비율, ~0.25)보다 높을수록 좋음 |
| Brier Score | 확률 추정의 평균 제곱 오차. 0에 가까울수록 좋고, 0.25가 무정보 기저선 | < 0.25 권장 |
| Calibration Plot | 모델이 "확률 0.7"이라고 할 때 실제로 70%가 양성인지 검증 | 대각선에 가까울수록 잘 교정됨 |

**통계 검정**:

| 검정 | 귀무가설 | 임상적 해석 |
|---|---|---|
| Pearson r (p값) | 선형 상관 없음 | p < 0.05 → 모델의 예측이 실제값과 유의미하게 연관 |
| Shapiro-Wilk | 잔차가 정규분포 | p > 0.05 → 잔차 정규성 만족 (선형 회귀 가정 충족 확인용) |
| 1-sample t-test (Bias) | 평균 잔차 = 0 | p < 0.05 → 체계적 과대/과소 추정 편향 존재 |

---

## 8. 전체 실행 흐름 요약 다이어그램

```
run_resnet1d(df, y_raw, target, results_dir, label, feature_selector)
│
├── [1] 데이터 분할
│       train_test_split(test_size=0.2, seed=42)
│       → df_train_all (80%), df_test_all (20%)
│
├── [2] 피처 선택 (4가지 조합 공통 — 1회만 수행)
│       KFold(n=5) → fold_splits 저장
│       Fold 1~5: feature_selector(fold_train) → fold_feat_lists[i]
│       feature_selector(df_train_all) → sel_feats_final
│
└── [3] SCALE_CONFIGS 루프  ×4회 반복
        ('noScale', False, False)
        ('scaleX',  True,  False)
        ('scaleY',  False, True )
        ('scaleXY', True,  True )
        │
        ├── [3-1] 5-Fold CV
        │         Fold 1~5:
        │           sel_feats = fold_feat_lists[i]   ← 피처 재사용
        │           X 처리: scale_X=True → StandardScaler.fit(fold_tr).transform
        │                   scale_X=False → shape 변환만 (N,F) → (N,1,F)
        │           y 처리: scale_y=True → StandardScaler.fit(y_fold_tr).transform
        │                   scale_y=False → 원래 스케일 그대로
        │           ResNet1D → _fit(500 epoch, AdamW+Cosine) → best_state
        │           _eval_epoch → ps, ts
        │           scale_y=True → ps = sc_y.inverse_transform(ps)
        │           → MAE, R², 성별별 ACC/AUC 기록
        │
        ├── [3-2] 최종 모델 학습
        │         sel_feats_final 사용
        │         X: scale_X=True → StandardScaler.fit(df_train_all)
        │         y: scale_y=True → StandardScaler.fit(y_train_all)
        │         ResNet1D → _fit(전체 train) → best_state_final
        │         _eval_epoch(test) → ps_te
        │         scale_y=True → ps_te = sc_y_final.inverse_transform(ps_te)
        │
        ├── [3-3] 결과 저장
        │         noScale  → ResNet1D_Report.md
        │         scaleX   → ResNet1D_scaleX_Report.md
        │         scaleY   → ResNet1D_scaleY_Report.md
        │         scaleXY  → ResNet1D_scaleXY_Report.md
        │         (ACC, AUC, AUPRC, Brier 모두 포함)
        │
        └── [3-4] 시각화 (8패널)
                  noScale  → ResNet1D_Results.png
                  scaleX   → ResNet1D_scaleX_Results.png
                  scaleY   → ResNet1D_scaleY_Results.png
                  scaleXY  → ResNet1D_scaleXY_Results.png
                  (PR Curve, Calibration Plot 포함)
```

---

## 9. 설계 선택의 연구적 의미

### 표형식 데이터에 1D CNN을 쓰는 이유

임상 데이터는 본질적으로 순서가 없는(unordered) 특징의 집합입니다. 이를 `(N, 1, F)` 형태로 변환해 1D CNN에 입력하면:

- **지역적 특징 상호작용 학습**: 인접한 특징 간의 관계를 커널이 포착
- **ResNet의 skip connection**: 깊은 네트워크에서도 원본 신호가 직접 전달 → 학습 안정성
- **MLP 대비 매개변수 공유**: Conv 가중치를 공유하므로 적은 파라미터로 표현력 확보

### Nested CV가 아닌 Holdout + CV 구조

완전한 Nested CV(외부 loop CV + 내부 loop 피처 선택)보다 계산 비용이 낮으면서, 분할된 test set을 완전히 격리함으로써 평가의 독립성을 보장합니다. 임상 연구 규모(수백~수천 명)에서 현실적인 선택입니다.

### 성별별 임계값의 데이터 누수 방지

Fold별 ACC/AUC 계산 시 임계값을 `y_train_all[tr_idx]` (해당 fold의 학습 분할)에서만 계산합니다. 검증셋의 실제값 분포를 임계값 계산에 사용하면 information leakage가 발생하므로 이를 엄격히 분리합니다.

### 피처 선택의 Fold별 수행

`feature_selector`가 각 fold의 학습셋에만 적용됩니다. 전체 데이터에 대해 피처를 선택한 후 CV를 수행하면 **선택 편향(selection bias)**이 발생해 CV 성능이 실제보다 낙관적으로 추정됩니다. 이 코드는 이를 올바르게 처리합니다.

### 4가지 스케일 조합 실험의 연구적 의미

StandardScaler를 X와 y에 각각 적용/미적용하는 조합을 모두 실험하는 이유는 다음과 같습니다.

**X 정규화 (scaleX) 효과**:
- 피처 간 단위가 다를 때(예: AEC 피처 값 범위가 불균일) Conv 레이어의 초기 수렴 속도에 영향
- BatchNorm1d가 내부적으로 정규화를 수행하므로, 깊은 네트워크에서는 scaleX 효과가 작을 수 있음
- 결과 비교로 "BatchNorm만으로 충분한가"를 데이터 기반으로 검증 가능

**y 정규화 (scaleY) 효과**:
- MSELoss = `(pred - true)²` 이므로, y의 절대 범위가 클수록 손실 값이 커지고 학습률 민감도 증가
- scaleY는 y를 `N(0,1)` 범위로 제한해 이를 해소
- **역변환 필수**: 지표(MAE, R², ACC, AUC)는 원래 단위로 계산해야 의미 있음
- y scaler는 반드시 학습셋에만 fit (test set y 분포가 학습에 영향 주면 leakage)

**예상 성능 순서 (일반론)**:

```
성능 향상 가능성 (높음) ←── scaleXY ≈ scaleY > scaleX ≈ noScale ──→ (낮음)

단, BatchNorm이 충분히 작동할 경우:
scaleXY ≈ scaleY > noScale ≈ scaleX
```

실제 데이터에서의 결과는 피처 수, 데이터 규모, y 값의 범위에 따라 달라질 수 있으며, 4가지 조합의 보고서를 비교해 최적 전처리 전략을 선택합니다.

### 이진 분류 지표 3종 사용 이유

| 지표 | 특성 | 사용 이유 |
|---|---|---|
| AUC-ROC | 임계값 무관, 클래스 불균형에 다소 낙관적 | 전통적 표준 지표, 비교 기반 |
| AUPRC | 양성 클래스 중심, 불균형에 민감 | 25% 양성 비율에서 AUC보다 정보량 많음 |
| Brier Score | 확률 추정의 정확도 (calibration + sharpness) | 모델의 확신도 평가, 임상 의사결정 지원 |

---

*작성일: 2026-05-08*  
*업데이트: v2 — StandardScaler 4가지 조합, AUPRC/Brier Score, 8패널 시각화 추가*  
*대상 파일: `연구코드/code/0508/core_resnet1d.py`*
