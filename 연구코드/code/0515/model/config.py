import os
import numpy as np
import torch

DATA_PATH     = "연구코드/data/강남/강남_merged_features.xlsx"
AEC_SHEET     = "meta_aec256"   # sheet name containing aec_1 ~ aec_256
AEC_LEN       = 256

SEED        = 42
TEST_SIZE   = 0.2
N_FOLDS     = 5
BATCH_SIZE  = 32
EPOCHS      = 200
LR_RATE     = 1e-3
HIDDEN      = 64
N_BLOCKS    = 4
N_HEADS     = 4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR         = "연구코드/results/0515/model_1"
RESULTS_DIR_CROSS   = "연구코드/results/0515/model_2"
RESULTS_DIR_CROSS_2_2 = "연구코드/results/0515/model_2_2"
RESULTS_DIR_CROSS3  = "연구코드/results/0515/model_3"

# AEC 셔플용 seed — SEED(42)와 다른 값으로 설정하여 우연한 재정렬 방지
AEC_SHUFFLE_SEED = 123

SMI_THRESH_M = 40.96
SMI_THRESH_F = 30.6

# Model 3: 전체 데이터 대비 비율이 이 값 미만인 ManufacturerModelName은 사전 제거
MIN_MFR_RATIO = 0.05

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR_CROSS, exist_ok=True)
os.makedirs(RESULTS_DIR_CROSS_2_2, exist_ok=True)
os.makedirs(RESULTS_DIR_CROSS3, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)
