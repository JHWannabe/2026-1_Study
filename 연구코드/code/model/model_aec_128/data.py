import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, SEED, TEST_SIZE, SMI_THRESH_M, SMI_THRESH_F, SPLIT_TRAIN_ID_PATH, SPLIT_TEST_ID_PATH


def load_data():
    """AEC 128차원 + clinic 피처와 sarcopenia 레이블을 로드해
    (X_aec, X_clinic, y, sex, patient_ids) 반환."""
    aec_cols = [f"aec_{i}" for i in range(1, 129)]

    aec_df  = pd.read_excel(DATA_PATH, sheet_name="aec_128")[["PatientID", "SMI"] + aec_cols]
    meta_df = pd.read_excel(DATA_PATH, sheet_name="metadata")[
        ["PatientID", "PatientSex", "PatientAge", "BMI"]
    ]

    for d in (aec_df, meta_df):
        d["PatientID"] = d["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    df = aec_df.merge(meta_df, on="PatientID", how="inner").dropna().reset_index(drop=True)

    thresh = np.where(df["PatientSex"] == "M", SMI_THRESH_M, SMI_THRESH_F)
    df["label"]   = np.where(df["SMI"].values <= thresh, 1, 0)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(np.float32)

    X_aec       = df[aec_cols].values.astype(np.float32)                          # (N, 128)
    X_clinic    = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)  # (N, 3)
    y           = df["label"].values.astype(np.int64)
    sex         = df["PatientSex"].values
    patient_ids = df["PatientID"].values
    return X_aec, X_clinic, y, sex, patient_ids


def split_data(X_aec, X_clinic, y, sex, patient_ids):
    """txt 파일이 있으면 저장된 ID로 split 복원, 없으면 stratified split 후 저장."""
    if os.path.exists(SPLIT_TRAIN_ID_PATH) and os.path.exists(SPLIT_TEST_ID_PATH):
        print(f"[split_data] 저장된 split 파일 로드: {SPLIT_TRAIN_ID_PATH}")
        train_ids = _load_ids(SPLIT_TRAIN_ID_PATH)
        test_ids  = _load_ids(SPLIT_TEST_ID_PATH)
        cv_idx, te_idx = _ids_to_indices(patient_ids, train_ids, test_ids)
    else:
        print("[split_data] split 파일 없음 → 새로 분할 후 저장")
        cv_idx, te_idx = _stratified_split(y, sex)
        _save_split_ids(patient_ids[cv_idx], patient_ids[te_idx])

    return (
        X_aec[cv_idx], X_clinic[cv_idx], y[cv_idx], sex[cv_idx],
        X_aec[te_idx], X_clinic[te_idx], y[te_idx], sex[te_idx],
    )


def _stratified_split(y, sex):
    idx   = np.arange(len(y))
    base  = y * 2 + (sex == "M").astype(int)
    strat = base if pd.Series(base).value_counts().min() >= 2 else y
    return train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, stratify=strat)


def _load_ids(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _ids_to_indices(patient_ids, train_ids, test_ids):
    id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
    cv_idx = np.array([id_to_idx[pid] for pid in train_ids if pid in id_to_idx])
    te_idx = np.array([id_to_idx[pid] for pid in test_ids  if pid in id_to_idx])
    missing = (len(train_ids) - len(cv_idx)) + (len(test_ids) - len(te_idx))
    if missing:
        print(f"[split_data] 경고: 저장된 ID 중 {missing}개가 현재 데이터에 없음")
    return cv_idx, te_idx


def _save_split_ids(train_ids, test_ids):
    os.makedirs(os.path.dirname(SPLIT_TRAIN_ID_PATH), exist_ok=True)
    header = f"# SEED={SEED}  TEST_SIZE={TEST_SIZE}  n={{}}\n"
    with open(SPLIT_TRAIN_ID_PATH, "w", encoding="utf-8") as f:
        f.write(header.format(len(train_ids)))
        f.write("\n".join(train_ids) + "\n")
    with open(SPLIT_TEST_ID_PATH, "w", encoding="utf-8") as f:
        f.write(header.format(len(test_ids)))
        f.write("\n".join(test_ids) + "\n")


# ── Direction 4: Liver 후반부 집중 크롭 실험 ──────────────────────────────────

def crop_to_liver_region(X_aec, start_dim=51):
    """128-dim AEC 신호에서 간(liver) 집중 구간(후반부)만 추출 후 128-dim 선형 보간.

    현재 AEC 크롭 기준: liver_upper → pubis (128-dim 전체).
    간 조직은 크롭 기준점(liver_upper) 직후 40~60 dim 이내에 집중됨.
    start_dim=51 (~40%) 이후 구간만 사용 → 간 후반부 집중 실험.

    Parameters
    ----------
    X_aec : (N, 128) array  — 원본 AEC 신호
    start_dim : int         — 크롭 시작 인덱스 (0-based, 기본 51 = 40%)

    Returns
    -------
    (N, 128) array — 크롭 후 128-dim으로 선형 보간된 신호
    """
    N, D = X_aec.shape
    cropped = X_aec[:, start_dim:]           # (N, D - start_dim)
    crop_len = D - start_dim

    x_old = np.linspace(0.0, 1.0, crop_len)
    x_new = np.linspace(0.0, 1.0, D)
    out   = np.empty((N, D), dtype=X_aec.dtype)
    for i in range(N):
        out[i] = np.interp(x_new, x_old, cropped[i])
    return out
