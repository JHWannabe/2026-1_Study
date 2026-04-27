"""
data_loader.py - 데이터 로드·전처리 공통 유틸리티
모든 분석 스크립트에서 import 하여 사용
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config as config


# ─────────────────────────────────────────────────────────────────────────────
# 1. 원시 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data() -> tuple:
    """
    'metadata' + 'features' 시트를 PatientID 기준으로 병합.
    - 중복 PatientID: 첫 번째 레코드만 유지
    - 결측치: 어느 컬럼이라도 값이 없는 행 전체 제거 (완전 제거법)

    반환: (df, cleaning_log)
      cleaning_log: 각 정제 단계별 환자 수 기록 list[dict]
    """
    meta  = pd.read_excel(config.EXCEL_PATH, sheet_name='metadata-value')
    feats = pd.read_excel(config.EXCEL_PATH, sheet_name='features')
    print(f"  [load] metadata: {len(meta)}명, features: {len(feats)}명 → 병합 중...")

    cleaning_log = [
        {'단계': '원시 데이터 (metadata 시트)', '환자수': len(meta)},
        {'단계': '원시 데이터 (features 시트)', '환자수': len(feats)},
    ]

    ids_meta  = set(meta['PatientID'])
    ids_feats = set(feats['PatientID'])
    only_meta  = sorted(ids_meta  - ids_feats)
    only_feats = sorted(ids_feats - ids_meta)
    if only_meta:
        print(f"  [load] metadata에만 존재 (features 없음): {len(only_meta)}명 → {only_meta}")
    if only_feats:
        print(f"  [load] features에만 존재 (metadata 없음): {len(only_feats)}명 → {only_feats}")
    if not only_meta and not only_feats:
        print(f"  [load] 모든 PatientID가 양쪽 시트에 존재 (누락 없음)")

    df = meta.merge(feats, on='PatientID', how='inner')
    cleaning_log.append({'단계': 'Inner join (공통 PatientID)', '환자수': len(df)})

    n_before = len(df)
    df = df.drop_duplicates(subset='PatientID', keep='first')
    if len(df) < n_before:
        print(f"  [load] 중복 PatientID 제거: {n_before - len(df)}건")
    cleaning_log.append({'단계': '중복 PatientID 제거 후', '환자수': len(df)})

    n_before = len(df)
    df = df.dropna()
    if len(df) < n_before:
        print(f"  [load] 결측치 행 제거: {n_before - len(df)}건 (전체 컬럼 기준)")
    cleaning_log.append({'단계': '결측치 행 제거 후 (최종)', '환자수': len(df)})

    df = df.reset_index(drop=True)
    print(f"  [load] 최종 데이터셋: {len(df)}명, {df.shape[1]}개 컬럼")
    return df, cleaning_log


# ─────────────────────────────────────────────────────────────────────────────
# 2. 개별 전처리 함수
# ─────────────────────────────────────────────────────────────────────────────

def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """PatientSex → Sex (M=1, F=0)."""
    df = df.copy()
    df['Sex'] = (df['PatientSex'] == 'M').astype(int)
    return df


def add_tama_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    성별 특이적 임계값으로 TAMA 이진화 (config.py 설정 참조).
    TAMA < 임계값 → 1 (low muscle), 이상 → 0 (normal)
    """
    df = df.copy()
    male   = df['PatientSex'] == 'M'
    female = df['PatientSex'] == 'F'
    df['TAMA_binary'] = 0
    df.loc[male   & (df['TAMA'] < config.TAMA_THRESHOLD_MALE),   'TAMA_binary'] = 1
    df.loc[female & (df['TAMA'] < config.TAMA_THRESHOLD_FEMALE), 'TAMA_binary'] = 1

    n_pos = int(df['TAMA_binary'].sum())
    n_tot = len(df)
    print(f"  [binary] TAMA_binary: 양성(low muscle) {n_pos}명 ({n_pos/n_tot*100:.1f}%)  "
          f"[M<{config.TAMA_THRESHOLD_MALE}, F<{config.TAMA_THRESHOLD_FEMALE} cm²]")
    return df


def standardize_cols(df: pd.DataFrame,
                     cols: list,
                     scaler: StandardScaler = None
                     ) -> tuple:
    """
    지정 컬럼을 Z-score 표준화. 새 컬럼명: <col>_z
    scaler=None 이면 새로 fit, 아니면 transform만 수행.
    반환: (변환된 DataFrame, StandardScaler)
    """
    df = df.copy()
    z_cols = [c + '_z' for c in cols]
    if scaler is None:
        scaler = StandardScaler()
        df[z_cols] = scaler.fit_transform(df[cols].values)
    else:
        df[z_cols] = scaler.transform(df[cols].values)
    return df, scaler


def add_model_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """ManufacturerModelName → one-hot 더미 변수 (drop_first=True)."""
    dummies = pd.get_dummies(df['ManufacturerModelName'], prefix='Model', drop_first=True)
    dummies = dummies.astype(int)
    return pd.concat([df.copy(), dummies], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 통합 전처리
# ─────────────────────────────────────────────────────────────────────────────

def prepare_full(mode: str = 'linear') -> pd.DataFrame:
    """
    전체 데이터 전처리 파이프라인 (전체 데이터셋 기준 scaler 적합).

    mode='linear'   → TAMA 연속형 그대로 사용
    mode='logistic' → TAMA_binary 컬럼 추가
    """
    # AEC feature 컬럼 존재 여부 확인
    missing = set(config.SELECTED_AEC_FEATURES)
    df, _ = load_raw_data()

    missing -= set(df.columns)
    if missing:
        raise KeyError(
            f"config.SELECTED_AEC_FEATURES에 지정된 컬럼이 데이터에 없습니다: {missing}\n"
            f"  → feature_selection.py 실행 후 config.py를 업데이트하세요."
        )

    # AEC features 결측치 행 제거 (표준화 전)
    n_before = len(df)
    df = df.dropna(subset=config.SELECTED_AEC_FEATURES)
    if len(df) < n_before:
        print(f"  [load] AEC feature 결측 제거: {n_before - len(df)}건")

    df = encode_sex(df)

    # Age 표준화
    df, _ = standardize_cols(df, ['PatientAge'])
    df = df.rename(columns={'PatientAge_z': 'Age_z'})

    # AEC features 표준화
    df, _ = standardize_cols(df, config.SELECTED_AEC_FEATURES)

    # KVP 표준화 (Case 3용) — KVP / kvp / kVp 컬럼 자동 탐지
    kvp_raw = next((c for c in ['KVP', 'kvp', 'kVp'] if c in df.columns), None)
    if kvp_raw:
        n_before = len(df)
        df = df.dropna(subset=[kvp_raw])
        if len(df) < n_before:
            print(f"  [load] {kvp_raw} 결측 제거: {n_before - len(df)}건")
        df, _ = standardize_cols(df, [kvp_raw])
        df = df.rename(columns={f'{kvp_raw}_z': 'kvp_z'})

    # ManufacturerModelName 더미 변수
    df = add_model_dummies(df)

    if mode == 'logistic':
        df = add_tama_binary(df)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Case별 feature 컬럼 목록
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(case: int, df: pd.DataFrame) -> list:
    """
    Case 번호에 따른 예측변수 컬럼 목록 반환.

    Case 1: [Sex, Age_z]
    Case 2: [Sex, Age_z] + 선택 AEC features (표준화)
    Case 3: [Sex, Age_z] + 선택 AEC features + KVP + ManufacturerModelName 더미
    """
    base  = ['Sex', 'Age_z']
    aec   = [f + '_z' for f in config.SELECTED_AEC_FEATURES]
    model = sorted([c for c in df.columns if c.startswith('Model_')])

    kvp_col = []
    if 'kvp_z' in df.columns:
        kvp_col = ['kvp_z']

    if case == 0:
        return aec                          # AEC 단독 (Sex, Age 없음)
    elif case == 1:
        return base
    elif case == 2:
        return base + aec
    elif case == 3:
        return base + aec + kvp_col + model
    else:
        raise ValueError(f"Case는 0, 1, 2, 3 중 하나여야 합니다 (입력값: {case})")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CV fold용 전처리 (data leakage 방지)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_cv_fold(df_raw: pd.DataFrame,
                    train_idx: np.ndarray,
                    test_idx: np.ndarray
                    ) -> tuple:
    """
    5-fold CV의 한 fold에 대해 data leakage 없이 전처리.
    scaler는 train fold에서만 fit → test fold에 transform.
    반환: (df_train, df_test)
    """
    df_tr = df_raw.iloc[train_idx].copy()
    df_te = df_raw.iloc[test_idx].copy()

    # Age 표준화
    df_tr, sc_age = standardize_cols(df_tr, ['PatientAge'])
    df_tr = df_tr.rename(columns={'PatientAge_z': 'Age_z'})
    df_te, _      = standardize_cols(df_te, ['PatientAge'], scaler=sc_age)
    df_te = df_te.rename(columns={'PatientAge_z': 'Age_z'})

    # AEC features 표준화
    aec = config.SELECTED_AEC_FEATURES
    df_tr, sc_aec = standardize_cols(df_tr, aec)
    df_te, _      = standardize_cols(df_te, aec, scaler=sc_aec)

    # KVP 표준화 (Case 3용) — KVP / kvp / kVp 컬럼 자동 탐지
    kvp_raw = next((c for c in ['KVP', 'kvp', 'kVp'] if c in df_tr.columns), None)
    if kvp_raw:
        df_tr, sc_kvp = standardize_cols(df_tr, [kvp_raw])
        df_te, _      = standardize_cols(df_te, [kvp_raw], scaler=sc_kvp)
        df_tr = df_tr.rename(columns={f'{kvp_raw}_z': 'kvp_z'})
        df_te = df_te.rename(columns={f'{kvp_raw}_z': 'kvp_z'})

    # ManufacturerModelName 더미 (전체 데이터셋 기준 카테고리 유지)
    df_tr = add_model_dummies(df_tr)
    df_te = add_model_dummies(df_te)

    # 카테고리 불일치 보완 (test fold에 없는 더미 컬럼을 0으로 채움)
    for col in df_tr.columns:
        if col.startswith('Model_') and col not in df_te.columns:
            df_te[col] = 0
    for col in df_te.columns:
        if col.startswith('Model_') and col not in df_tr.columns:
            df_tr[col] = 0

    df_tr = encode_sex(df_tr)
    df_te = encode_sex(df_te)

    return df_tr, df_te
