"""
프로젝트 전역 하이퍼파라미터·경로·상수 정의.

단계별 성능 비교를 위한 EXPERIMENT_STAGES 정의 포함.
환경변수 EXPERIMENT_STAGE (기본값 0)으로 실험 스테이지를 선택한다.
  예: EXPERIMENT_STAGE=2 python model/main.py       (직접 실행)
      python model/experiments.py --stages 0,1,2   (단계별 자동 실행)

스테이지 독립 비교표 (각 스테이지는 Stage 0 대비 변수 하나만 변경):
  Stage 0: Baseline — LR=1e-5, HIDDEN=16, N_HEADS=1, N_BLOCKS=1, GRAD_CLIP=0, N_CA_LAYERS=1
  Stage 1: HIDDEN 16 → 64
  Stage 2: N_HEADS 1 → 4
  Stage 3: N_BLOCKS 1 → 2
  Stage 4: GRAD_CLIP 0 → 1.0
  Stage 5: N_CA_LAYERS 1 → 2
"""
import os
import numpy as np
import torch

# ── 데이터 경로 ──────────────────────────────────────────────
DATA_PATH = "연구코드/data/강남/강남_liver_merged_features_ok.xlsx"
AEC_LEN   = 128
AEC_SHEET = "aec_128"

AEC_MODES = {
    "crop80": {"aec_len": 128, "aec_sheet": "aec_128", "crop_points": 103},
    "raw128": {"aec_len": 128, "aec_sheet": "aec_128", "crop_points": None},
}

AEC_VARIANTS = [
    "raw",
    "std_scaled",
    "norm",
    "global_zscore",
]

# ── 실험 재현성 ────────────────────────────────────────────────
SEED             = 42
TEST_SIZE        = 0.2
AEC_SHUFFLE_SEED = 123

# ── 고정 하이퍼파라미터 (모든 스테이지 공통) ──────────────────
N_FOLDS     = 5
BATCH_SIZE  = 32
EPOCHS      = 500
FOCAL_GAMMA = 2.0

# ── 단계별 실험 설정 ──────────────────────────────────────────
# 각 스테이지는 Stage 0 대비 변수 하나만 변경한다 (독립 비교).
EXPERIMENT_STAGES = {
    0: {
        "desc":        "N_BLOCKS: 1 → 2, N_CA_LAYERS: 1 → 2",
        "LR_RATE":     1e-5,
        "HIDDEN":      16,
        "N_HEADS":     1,
        "N_BLOCKS":    2,
        "GRAD_CLIP":   0.0,
        "N_CA_LAYERS": 2,
    },
    # 1: {
    #     "desc":        "HIDDEN: 16 → 64",
    #     "LR_RATE":     1e-5,
    #     "HIDDEN":      64,
    #     "N_HEADS":     1,
    #     "N_BLOCKS":    1,
    #     "GRAD_CLIP":   0.0,
    #     "N_CA_LAYERS": 1,
    # },
    # 2: {
    #     "desc":        "N_HEADS: 1 → 4",
    #     "LR_RATE":     1e-5,
    #     "HIDDEN":      16,
    #     "N_HEADS":     4,
    #     "N_BLOCKS":    1,
    #     "GRAD_CLIP":   0.0,
    #     "N_CA_LAYERS": 1,
    # },
    # 3: {
    #     "desc":        "N_BLOCKS: 1 → 2",
    #     "LR_RATE":     1e-5,
    #     "HIDDEN":      16,
    #     "N_HEADS":     1,
    #     "N_BLOCKS":    2,
    #     "GRAD_CLIP":   0.0,
    #     "N_CA_LAYERS": 1,
    # },
    # 4: {
    #     "desc":        "GradClip: 0 → 1.0",
    #     "LR_RATE":     1e-5,
    #     "HIDDEN":      16,
    #     "N_HEADS":     1,
    #     "N_BLOCKS":    1,
    #     "GRAD_CLIP":   1.0,
    #     "N_CA_LAYERS": 1,
    # },
    # 5: {
    #     "desc":        "N_CA_LAYERS: 1 → 2",
    #     "LR_RATE":     1e-5,
    #     "HIDDEN":      16,
    #     "N_HEADS":     1,
    #     "N_BLOCKS":    1,
    #     "GRAD_CLIP":   0.0,
    #     "N_CA_LAYERS": 2,
    # },
}

# ── 스테이지 적용 (환경변수 EXPERIMENT_STAGE, 기본값 0) ────────
CURRENT_STAGE      = int(os.environ.get("EXPERIMENT_STAGE", "0"))
_stage_cfg         = EXPERIMENT_STAGES.get(CURRENT_STAGE, EXPERIMENT_STAGES[0])
CURRENT_STAGE_DESC = _stage_cfg["desc"]

LR_RATE     = _stage_cfg["LR_RATE"]
HIDDEN      = _stage_cfg["HIDDEN"]
N_HEADS     = _stage_cfg["N_HEADS"]
N_BLOCKS    = _stage_cfg["N_BLOCKS"]
GRAD_CLIP   = _stage_cfg["GRAD_CLIP"]
N_CA_LAYERS = _stage_cfg["N_CA_LAYERS"]

# ── 기기 및 결과 경로 ─────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR           = f"연구코드/results/0605_2/stage{CURRENT_STAGE}_{LR_RATE}_epoch{EPOCHS}/"
RESULTS_MODEL_1_DIR   = RESULTS_DIR + "model_1"
RESULTS_MODEL_2_DIR   = RESULTS_DIR + "model_2"
RESULTS_MODEL_2_2_DIR = RESULTS_DIR + "model_2_2"
RESULTS_MODEL_3_DIR   = RESULTS_DIR + "model_3"

# ── Sarcopenia 진단 기준 (SMI, cm²/m²) ───────────────────────
SMI_THRESH_M = 40.96
SMI_THRESH_F = 30.6


# ── 초기화 ────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR,           exist_ok=True)
os.makedirs(RESULTS_MODEL_1_DIR,   exist_ok=True)
os.makedirs(RESULTS_MODEL_2_DIR,   exist_ok=True)
os.makedirs(RESULTS_MODEL_2_2_DIR, exist_ok=True)
os.makedirs(RESULTS_MODEL_3_DIR,   exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)
