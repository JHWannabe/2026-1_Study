"""
프로젝트 전역 하이퍼파라미터·경로·상수 정의.
"""
import os
import numpy as np
import torch

# ── 데이터 경로 ──────────────────────────────────────────────
DATA_PATH = "연구코드/data/강남/강남_liver_merged_features_ok.xlsx"
AEC_LEN   = 128
AEC_SHEET = "aec_128"

AEC_VARIANTS = [
    "raw",
    "norm",
    "global_zscore",
]

# ── 실험 재현성 ────────────────────────────────────────────────
SEED             = 42
TEST_SIZE        = 0.2
AEC_SHUFFLE_SEED = 123

# ── 하이퍼파라미터 ────────────────────────────────────────────
N_FOLDS     = 5
BATCH_SIZE  = 16
EPOCHS      = 500
FOCAL_GAMMA = 2.0

LR_RATE     = 1e-5
HIDDEN      = 16
N_HEADS     = 1
N_BLOCKS    = 2
GRAD_CLIP   = 0.0
N_CA_LAYERS = 2

# ── 기기 및 결과 경로 ─────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PARENT_DIR            = r"연구코드\results"
RESULTS_DIR           = f"{PARENT_DIR}/0619/"
RESULTS_MODEL_1_DIR   = RESULTS_DIR + "model_1"
RESULTS_MODEL_2_DIR   = RESULTS_DIR + "model_2"
RESULTS_MODEL_5_DIR   = RESULTS_DIR + "model_5"

# ── Sarcopenia 진단 기준 (SMI, cm²/m²) ───────────────────────
SMI_THRESH_M = 40.96
SMI_THRESH_F = 30.6

# ── 모델 출력 Threshold 수동 지정 ────────────────────────────
# None: CV fold 중앙값 자동 사용 / float: 해당 값으로 고정 (예: 0.4)
COMMON_THRESH = None
THRESH_M1    = COMMON_THRESH
THRESH_M2    = COMMON_THRESH
THRESH_M5    = COMMON_THRESH


# ── 초기화 ────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR,            exist_ok=True)
os.makedirs(RESULTS_MODEL_1_DIR,    exist_ok=True)
os.makedirs(RESULTS_MODEL_2_DIR,    exist_ok=True)
os.makedirs(RESULTS_MODEL_5_DIR,    exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)
