DATA_PATH = "연구코드/data/강남/강남_liver_merged_features_ok.xlsx"

SEED      = 42
TEST_SIZE = 0.2
N_FOLDS   = 5

PARENT_DIR          = "연구코드/results/"
RESULTS_MODEL_1_DIR = PARENT_DIR + "baseline"

SMI_THRESH_M = 40.96
SMI_THRESH_F = 30.6

SPLIT_TRAIN_ID_PATH = PARENT_DIR + "train_patient_ids.txt"
SPLIT_TEST_ID_PATH  = PARENT_DIR + "test_patient_ids.txt"

# None: CV fold 중앙값 자동 사용 / float: 해당 값으로 고정 (예: 0.4)
THRESH_M1 = None