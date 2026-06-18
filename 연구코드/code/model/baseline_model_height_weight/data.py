import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, SEED, TEST_SIZE, SMI_THRESH_M, SMI_THRESH_F, SPLIT_TRAIN_ID_PATH, SPLIT_TEST_ID_PATH


def load_data():
    """임상 데이터(Age, Sex, Height, Weight)와 sarcopenia 레이블을 로드해 (X, y, sex, patient_ids) 반환."""
    cols = ["PatientID", "PatientAge", "PatientSex", "Height", "Weight", "SMI",
            "kVp", "ManufacturerModelName"]
    df = pd.read_excel(DATA_PATH, sheet_name="metadata")[cols].dropna().reset_index(drop=True)
    df["PatientID"] = df["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    thresh = np.where(df["PatientSex"] == "M", SMI_THRESH_M, SMI_THRESH_F)
    df["label"]   = np.where(df["SMI"].values <= thresh, 1, 0)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    X           = df[["PatientAge", "sex_enc", "Height", "Weight"]].values.astype(np.float32)
    y           = df["label"].values.astype(np.int64)
    sex         = df["PatientSex"].values
    patient_ids = df["PatientID"].values
    return X, y, sex, patient_ids


def split_data(X, y, sex, patient_ids):
    """txt 파일이 있으면 저장된 ID로 split 복원, 없으면 stratified split 후 저장.
    (X_cv, y_cv, sex_cv, X_te, y_te, sex_te) 반환."""
    if os.path.exists(SPLIT_TRAIN_ID_PATH) and os.path.exists(SPLIT_TEST_ID_PATH):
        print(f"[split_data] 저장된 split 파일 로드: {SPLIT_TRAIN_ID_PATH}")
        train_ids = _load_ids(SPLIT_TRAIN_ID_PATH)
        test_ids  = _load_ids(SPLIT_TEST_ID_PATH)
        cv_idx, te_idx = _ids_to_indices(patient_ids, train_ids, test_ids)
    else:
        print("[split_data] split 파일 없음 → 새로 분할 후 저장")
        cv_idx, te_idx = _stratified_split(X, y, sex)
        _save_split_ids(patient_ids[cv_idx], patient_ids[te_idx])

    return (
        X[cv_idx], y[cv_idx], sex[cv_idx],
        X[te_idx], y[te_idx], sex[te_idx],
    )


def _stratified_split(X, y, sex):
    idx = np.arange(len(y))
    age, bmi = X[:, 0], X[:, 2]
    base = y * 2 + (sex == "M").astype(int)

    strat = None
    for n in (3, 2):
        age_bin = pd.qcut(age, q=n, labels=False, duplicates="drop").astype(int)
        bmi_bin = pd.qcut(bmi, q=n, labels=False, duplicates="drop").astype(int)
        key = (base * n + age_bin) * n + bmi_bin
        if pd.Series(key).value_counts().min() >= 2:
            strat = key
            break
    if strat is None:
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