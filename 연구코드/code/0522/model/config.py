"""
프로젝트 전역 하이퍼파라미터·경로·상수 정의.

모든 설정을 한 곳에서 관리해 실험 재현성을 확보한다.
각 모듈은 이 파일에서 필요한 값을 import해 사용하며, 직접 값을 하드코딩하지 않는다.
"""
import os
import numpy as np
import torch

# ── 데이터 경로 ──────────────────────────────────────────────
DATA_PATH = "연구코드/data/강남/강남_merged_features.xlsx"
AEC_LEN   = 128              # AEC 기본 시퀀스 길이 (default, 함수 파라미터로 오버라이드 가능)
AEC_SHEET = "aec_128"           # AEC_LEN에 따라 시트 자동 선택
AEC_SIZES = [128, 256]          # 비교할 AEC 보간 해상도 목록

# AEC 민감도 분석 변환 목록 — data.aec_variant() 참고
# 해상도(len*)·시간 범위(crop*)·정규화(norm)·이상치 제외(excl_extreme)를 비교한다
AEC_VARIANTS = [
    "len128",        # interpolated AEC 길이 128 (원본 길이와 무관하게 고정)
    "crop80",        # 중앙 80% 구간 (양끝 10% 제거)
    "crop60",        # 중앙 60% 구간 (양끝 20% 제거)
    "norm",          # 곡선 내 z-score 정규화 (스캐너 간 절대값 차이 제거)
    "excl_extreme",  # scan-length 상하위 5% 극단 샘플 제외
]

# ── 실험 재현성 ────────────────────────────────────────────────
SEED     = 42    # train/test split 및 모델 초기화 seed
TEST_SIZE = 0.2  # test set 비율

# AEC 셔플 seed를 SEED(42)와 다르게 설정해 Model 2_2 unmatching 실험에서
# 우연히 원본 순서와 일치하는 상황을 방지한다
AEC_SHUFFLE_SEED = 123

# ── 교차검증·학습 하이퍼파라미터 ──────────────────────────────
N_FOLDS    = 5      # Stratified K-Fold 수
BATCH_SIZE = 32
EPOCHS     = 200    # 각 fold 최대 epoch (best AUC epoch에서 조기 기록)
LR_RATE    = 1e-3
HIDDEN     = 64     # ResNet1D 채널 수 / Cross-Attention d_model
N_BLOCKS   = 4      # ResBlock1D 반복 수
N_HEADS    = 4      # Multi-head Attention head 수

# ── 기기 및 결과 경로 ─────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR           = "연구코드/results/0522/"
RESULTS_MODEL_1_DIR   = RESULTS_DIR + "model_1"   # M1 Clinic Only
RESULTS_MODEL_2_DIR   = RESULTS_DIR + "model_2"   # M2 Clinic+AEC (Matched)
RESULTS_MODEL_2_2_DIR = RESULTS_DIR + "model_2_2" # M2_2 Clinic+AEC (Unmatched)
RESULTS_MODEL_3_DIR   = RESULTS_DIR + "model_3"   # M3 Clinic+Scanner+AEC

# ── Sarcopenia 진단 기준 (SMI, cm²/m²) ───────────────────────
# AWGS 2019 기준: 남성 < 7.0 kg/m² → SMI 환산값
SMI_THRESH_M = 40.96  # 남성 sarcopenia 기준
SMI_THRESH_F = 30.6   # 여성 sarcopenia 기준

# ── Model 3 소수 제조사 필터 ──────────────────────────────────
# 전체 데이터 대비 비율이 MIN_MFR_RATIO 미만인 ManufacturerModelName 제거
# 소수 제조사는 Embedding 학습이 불안정하므로 사전 제거
MIN_MFR_RATIO = 0.05

# ── 초기화 ────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR,           exist_ok=True)
os.makedirs(RESULTS_MODEL_1_DIR,   exist_ok=True)
os.makedirs(RESULTS_MODEL_2_DIR,   exist_ok=True)
os.makedirs(RESULTS_MODEL_2_2_DIR, exist_ok=True)
os.makedirs(RESULTS_MODEL_3_DIR,   exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)
