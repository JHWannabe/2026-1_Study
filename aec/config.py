"""
config.py - 연구 전체 설정값 관리
TAMA 예측 회귀분석 연구 (강남 데이터셋)
"""

import os
import pandas as pd

# ── 데이터 경로 ───────────────────────────────────────────────────────────────
SITE = "신촌"
EXCEL_PATH  = rf"C:\Users\jhjun\OneDrive\Desktop\aec\{SITE}_merged_features.xlsx"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(EXCEL_PATH)), f"results/{SITE}")

# ── Logistic regression: 성별 특이적 TAMA 이진화 임계값 ────────────────────────
# TAMA < 임계값 → low muscle (label=1), 이상 → normal (label=0)
# 성별별 TAMA 하위 25% (P25) 를 데이터에서 동적으로 계산
def _compute_tama_thresholds():
    _meta = pd.read_excel(EXCEL_PATH, sheet_name='metadata-value', usecols=['PatientSex', 'TAMA'])
    _p25  = _meta.groupby('PatientSex')['TAMA'].quantile(0.25)
    return int(_p25.get('M', 170)), int(_p25.get('F', 110))

TAMA_THRESHOLD_MALE, TAMA_THRESHOLD_FEMALE = _compute_tama_thresholds()

# ── Case 2·3에서 사용할 AEC 특징 변수 ─────────────────────────────────────────
# feature_selection.py 실행 후 상관계수 상위 변수로 업데이트 권장
# 기본값: 상관계수 상위 & 다중공선성 낮은 조합 (feature_selection.py 결과 기반)
#   p25           : Pearson r=0.37 (강양성, 하위 사분위 AEC 값)
#   CV            : Pearson r=-0.35 (상대 변동성, mean과 다른 정보 제공)
#   skewness      : Pearson r=-0.34 (AEC 곡선 비대칭성)
#   slope_abs_mean: 곡선 동역학 feature (다른 그룹)
# ※ mean과 AUC_normalized는 VIF>50000 → 동시 사용 시 다중공선성 심각 주의
SELECTED_AEC_FEATURES = ['p25', 'CV', 'skewness', 'slope_abs_mean', 'mean']

# ── 통계 분석 파라미터 ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS     = 5       # linear regression 5-fold CV용
N_BOOTSTRAP  = 1000    # logistic AUC 신뢰구간 bootstrap 반복수
