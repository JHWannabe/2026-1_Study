"""
data_loader.py - 데이터 로드·전처리 공통 유틸리티
모든 분석 스크립트에서 import 하여 사용
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. 원시 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """
    'metadata' + 'features' 시트를 PatientID 기준으로 병합.
    - 중복 PatientID: 첫 번째 레코드만 유지
    - PatientAge 결측치: 해당 행 제거
    """
    meta  = pd.read_excel(config.EXCEL_PATH, sheet_name='metadata-value')
    feats = pd.read_excel(config.EXCEL_PATH, sheet_name='features')

    df = meta.merge(feats, on='PatientID', how='inner')

    n_before = len(df)
    df = df.drop_duplicates(subset='PatientID', keep='first')
    if len(df) < n_before:
        print(f"  [load] 중복 PatientID 제거: {n_before - len(df)}건")

    n_before = len(df)
    df = df.dropna(subset=['PatientAge'])
    if len(df) < n_before:
        print(f"  [load] PatientAge 결측 제거: {n_before - len(df)}건")

    df = df.reset_index(drop=True)
    print(f"  [load] 최종 데이터셋: {len(df)}명, {df.shape[1]}개 컬럼")
    return df


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
    df = load_raw_data()

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
    Case 3: [Sex, Age_z] + 선택 AEC features + ManufacturerModelName 더미
    """
    base  = ['Sex', 'Age_z']
    aec   = [f + '_z' for f in config.SELECTED_AEC_FEATURES]
    model = sorted([c for c in df.columns if c.startswith('Model_')])

    if case == 1:
        return base
    elif case == 2:
        return base + aec
    elif case == 3:
        return base + aec + model
    else:
        raise ValueError(f"Case는 1, 2, 3 중 하나여야 합니다 (입력값: {case})")


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
