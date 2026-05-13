"""
0520 실험 전역 설정.

새 실험 목록:
  Model 4   : AEC 핸드크래프트 피처 + LR / ResNet1D
  Model 5   : 성별 분리 CrossAttn (M-only, F-only)
  Model 6   : SMI 회귀 → 이진 분류
  Model 7   : 앙상블 (M1 LR + M2 CrossAttn)
  Calibration: Temperature Scaling (M2 CrossAttn 기반)
  Attention  : CrossAttn Attention 맵 시각화
"""
import os
import numpy as np
import torch

# ── 데이터 경로 ──────────────────────────────────────────────
DATA_PATH = "연구코드/data/강남/강남_merged_features.xlsx"
AEC_SHEET = "meta_aec256"
AEC_LEN   = 256

AEC_VARIANTS = [
    "len064", "len128", "len256",
    "crop80", "crop60", "norm", "excl_extreme",
]

# ── 재현성 ────────────────────────────────────────────────────
SEED             = 42
TEST_SIZE        = 0.2
AEC_SHUFFLE_SEED = 123

# ── 학습 하이퍼파라미터 ────────────────────────────────────────
N_FOLDS    = 5
BATCH_SIZE = 32
EPOCHS     = 200
LR_RATE    = 1e-3
HIDDEN     = 64
N_BLOCKS   = 4
N_HEADS    = 4

# ── 기기 ─────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Sarcopenia 기준 (AWGS 2019) ──────────────────────────────
SMI_THRESH_M  = 40.96   # 남성 기준 (cm²/m²)
SMI_THRESH_F  = 30.6    # 여성 기준
MIN_MFR_RATIO = 0.05    # 소수 제조사 제거 임계값

# ── 결과 경로 ────────────────────────────────────────────────
RESULTS_BASE      = "연구코드/results/0520"
RESULTS_DIR_M4    = f"{RESULTS_BASE}/model_4"       # AEC 해석가능 피처
RESULTS_DIR_M5_M  = f"{RESULTS_BASE}/model_5/male"  # 성별분리 — 남성
RESULTS_DIR_M5_F  = f"{RESULTS_BASE}/model_5/female"# 성별분리 — 여성
RESULTS_DIR_M6    = f"{RESULTS_BASE}/model_6"       # SMI 회귀
RESULTS_DIR_M7    = f"{RESULTS_BASE}/model_7"       # 앙상블
RESULTS_DIR_CALIB = f"{RESULTS_BASE}/calibration"   # 캘리브레이션
RESULTS_DIR_ATTN  = f"{RESULTS_BASE}/attention"     # Attention 시각화

for d in [
    RESULTS_DIR_M4, RESULTS_DIR_M5_M, RESULTS_DIR_M5_F,
    RESULTS_DIR_M6, RESULTS_DIR_M7, RESULTS_DIR_CALIB, RESULTS_DIR_ATTN,
]:
    os.makedirs(d, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
