"""
config.py - 연구 전체 설정값 관리
TAMA 예측 회귀분석 연구 (강남 데이터셋)
"""

import os
from pathlib import Path
import pandas as pd

# ── 데이터 경로 ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent   # aec/ 루트

SITE = "강남"
EXCEL_PATH  = str(_ROOT / "data" / f"{SITE}_merged_features.xlsx")

RESULTS_DIR = str(_ROOT / "results" / SITE)

# ── Logistic regression: 성별 특이적 TAMA 이진화 임계값 ────────────────────────
# TAMA < 임계값 → low muscle (label=1), 이상 → normal (label=0)
# 성별별 TAMA 하위 25% (P25) 를 데이터에서 동적으로 계산
def _compute_tama_thresholds():
    _meta = pd.read_excel(EXCEL_PATH, sheet_name='metadata-value', usecols=['PatientSex', 'TAMA'])
    _p25  = _meta.groupby('PatientSex')['TAMA'].quantile(0.25)
    return int(_p25.get('M', 170)), int(_p25.get('F', 110))

TAMA_THRESHOLD_MALE, TAMA_THRESHOLD_FEMALE = _compute_tama_thresholds()

# ── Case 2·3에서 사용할 AEC 특징 변수 (SITE별 독립 관리) ──────────────────────
# feature_selection.py 실행 후 상관계수 상위 변수로 사이트별 업데이트 권장
# ※ mean과 AUC_normalized는 VIF>50000 → 동시 사용 시 다중공선성 심각 주의
_SELECTED_AEC_FEATURES_BY_SITE = {
    # amplitude 그룹(mean≈AUC_normalized≈p25≈peak_max_height) 중 mean 하나만 선택
    # 강남: mean(r=0.297) 포함, VIF max=1.58
    "강남": ['mean', 'CV', 'skewness', 'slope_abs_mean'],
    # 신촌: mean 포함, peak_max_height(r=0.943 with mean)은 중복 제거, VIF max=1.27
    "신촌": ['mean', 'CV', 'skewness', 'slope_abs_mean'],
}
SELECTED_AEC_FEATURES = _SELECTED_AEC_FEATURES_BY_SITE.get(SITE, ['mean', 'CV', 'skewness', 'slope_abs_mean'])

# ── 통계 분석 파라미터 ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS     = 5       # linear regression 5-fold CV용
N_BOOTSTRAP  = 1000    # logistic AUC 신뢰구간 bootstrap 반복수
