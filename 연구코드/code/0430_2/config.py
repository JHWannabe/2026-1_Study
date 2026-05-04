# -*- coding: utf-8 -*-
"""
Global configuration shared across all 0430 analysis modules.
Import this module first in every other module:
    from config import *
"""

import warnings
warnings.filterwarnings("ignore")

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
)
from sklearn.pipeline import Pipeline
from scipy import stats as scipy_stats
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as sm_vif
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from matplotlib.patches import Patch

# ────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
FS_RESULT  = SCRIPT_DIR.parent / "results" / "feature_selection"

# ────────────────────────────────────────────────
# CV settings
# ────────────────────────────────────────────────
CV_SPLITS = 5
CV_RANDOM = 42

# ────────────────────────────────────────────────
# AEC feature sets
# ────────────────────────────────────────────────
# AEC_PREV: 0424부터 사용해 온 4개의 고정 피처 세트.
AEC_PREV = ["mean", "CV", "skewness", "slope_abs_mean"]

# AEC_CANDIDATES: feature_selection 파이프라인이 분산+상관 필터링으로 도출한 후보 세트.
# 최종 선택은 CV fold 내 SelectKBest로 수행 (data leakage 방지).
fs_xlsx = FS_RESULT / "feature_selection_summary.xlsx"
if fs_xlsx.exists():
    AEC_CANDIDATES = pd.read_excel(
        str(fs_xlsx), sheet_name="candidate_features"
    )["candidate_features"].tolist()
    print(f"[AEC-candidates] Loaded {len(AEC_CANDIDATES)} features from pipeline: {AEC_CANDIDATES}")
else:
    AEC_CANDIDATES = [
        "IQR", "band2_energy", "dominant_freq", "mean", "slope_max",
        "spectral_energy", "wavelet_cD1_energy", "wavelet_cD2_energy",
        "wavelet_energy_ratio_D1",
    ]
    print(f"[AEC-candidates] Fallback hardcoded {len(AEC_CANDIDATES)} features")

print(f"[AEC-prev] {len(AEC_PREV)} features: {AEC_PREV}")

# AEC_SELECT_K: SelectKBest가 각 CV fold 내에서 AEC_CANDIDATES 중 선택할 개수.
AEC_SELECT_K = 10

# ────────────────────────────────────────────────
# Hospital discovery
# ────────────────────────────────────────────────
HOSPITALS = {}
for _fname in os.listdir(DATA_DIR):
    if not _fname.endswith(".xlsx") or "merged_features" not in _fname:
        continue
    if "강남" in _fname:
        HOSPITALS["gangnam"] = DATA_DIR / _fname
    # 신촌 데이터는 SMI 컬럼 부재로 분석 제외

print(f"[Data] Found hospitals: {list(HOSPITALS.keys())}")

# ────────────────────────────────────────────────
# Labels & colors
# ────────────────────────────────────────────────
CASE_LABELS = {
    "Case1_Clinical":                  "Case 1\nClinical",
    "Case2_Clinical+AEC_prev":         "Case 2\n+AEC_prev",
    "Case3_Clinical+AEC_new":          "Case 3\n+AEC_new",
    "Case4_Clinical+AEC_prev+Scanner": "Case 4\n+AEC_prev\n+Scanner",
    "Case5_Clinical+AEC_new+Scanner":  "Case 5\n+AEC_new\n+Scanner",
}
COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#1a5276", "#922b21"]
