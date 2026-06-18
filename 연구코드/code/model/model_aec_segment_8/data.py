import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, SEED, TEST_SIZE, SMI_THRESH_M, SMI_THRESH_F, SPLIT_TRAIN_ID_PATH, SPLIT_TEST_ID_PATH, N_SEGMENTS, N_STATS


def extract_segment_features(X_aec, n_segments=N_SEGMENTS):
    """128-dim AEC를 n_segments 등분하여 intra + inter segment 피처를 추출.

    Intra (segment당): mean, std, auc, slope  → n_segments × 4
    Inter (인접 segment 간 차이): Δmean, Δstd, Δauc, Δslope  → (n_segments-1) × 4
    총 차원: n_segments×4 + (n_segments-1)×4 = 8×4 + 7×4 = 60

    Returns (N, 60) array.
    """
    N, D = X_aec.shape
    seg_len = D // n_segments  # 128 // 8 = 16
    x_idx = np.arange(seg_len, dtype=np.float64)
    x_c = x_idx - x_idx.mean()
    x_c_sq_sum = (x_c ** 2).sum()

    # intra-segment 통계: shape (N, n_segments, 4) — [mean, std, auc, slope]
    stats = np.zeros((N, n_segments, N_STATS), dtype=np.float64)
    for s in range(n_segments):
        seg = X_aec[:, s * seg_len:(s + 1) * seg_len].astype(np.float64)
        stats[:, s, 0] = seg.mean(axis=1)
        stats[:, s, 1] = seg.std(axis=1)
        stats[:, s, 2] = np.trapz(seg, axis=1)
        stats[:, s, 3] = (seg * x_c).sum(axis=1) / x_c_sq_sum

    # intra 피처: (N, n_segments×4)
    intra = stats.reshape(N, n_segments * N_STATS)

    # inter 피처 (인접 segment 차이): (N, (n_segments-1)×4)
    inter = (stats[:, 1:, :] - stats[:, :-1, :]).reshape(N, (n_segments - 1) * N_STATS)

    return np.concatenate([intra, inter], axis=1).astype(np.float32)  # (N, 60)


def load_data():
    """AEC 128차원을 8-segment 피처(60-dim) + clinic 피처와 sarcopenia 레이블을 로드해
    (X_seg, X_clinic, y, sex, patient_ids) 반환. X_seg shape: (N, 60)."""
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

    X_aec_raw   = df[aec_cols].values.astype(np.float32)                          # (N, 128)
    X_seg       = extract_segment_features(X_aec_raw)                             # (N, 60)
    X_clinic    = df[["PatientAge", "sex_enc", "BMI"]].values.astype(np.float32)  # (N, 3)
    y           = df["label"].values.astype(np.int64)
    sex         = df["PatientSex"].values
    patient_ids = df["PatientID"].values
    return X_seg, X_clinic, y, sex, patient_ids, X_aec_raw


def split_data(X_aec, X_clinic, y, sex, patient_ids, X_aec_raw):
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
        X_aec_raw[cv_idx], X_aec_raw[te_idx],
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
