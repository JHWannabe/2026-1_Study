DATA_PATH = "연구코드/data/강남/강남_liver_merged_features_ok_pre_exclude.xlsx"

SEED      = 42
TEST_SIZE = 0.2
N_FOLDS   = 5

PARENT_DIR          = "연구코드/results/"
RESULTS_MODEL_1_DIR = PARENT_DIR + "model_aec_segment_8/"

SMI_THRESH_M = 40.96
SMI_THRESH_F = 30.6

SPLIT_TRAIN_ID_PATH = PARENT_DIR + "train_patient_ids.txt"
SPLIT_TEST_ID_PATH  = PARENT_DIR + "test_patient_ids.txt"

# None: CV fold 중앙값 자동 사용 / float: 해당 값으로 고정 (예: 0.4)
THRESH_M1 = None

# AEC 피처 정규화 방식: "global" | "rowwise" | "raw"
AEC_NORM = ["global", "rowwise", "raw"]

# AEC segment 설정
N_SEGMENTS   = 8    # 128-dim AEC를 8등분 (segment당 16 포인트)
N_STATS      = 4    # 각 segment: mean, std, auc, slope
# intra: 8×4=32, inter(adjacent diff): 7×4=28 → 총 60
AEC_FEAT_DIM = N_SEGMENTS * N_STATS + (N_SEGMENTS - 1) * N_STATS  # 60

import torch
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 500
LR         = 1e-4
BATCH_SIZE = 32
PATIENCE   = 15
AEC_OUT    = 16    # AEC branch 출력 차원 (60→64→32→16 압축)