import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, SEED, TEST_SIZE, SPLIT_TRAIN_ID_PATH, SPLIT_TEST_ID_PATH


def load_data():
    """임상 데이터(Age, Sex, Height, Weight)와 SMI, IMATA를 로드해 (X, smi, imata, sex, patient_ids) 반환.
    레이블은 train split 이후 compute_thresholds() + make_labels()로 생성."""
    cols = ["PatientID", "PatientAge", "PatientSex", "Height", "Weight", "SMI", "IMATA",
            "kVp", "Manufacturer"]
    df = pd.read_excel(DATA_PATH, sheet_name="metadata")[cols].dropna().reset_index(drop=True)
    df["PatientID"] = df["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["sex_enc"] = (df["PatientSex"] == "M").astype(int)

    X           = df[["PatientAge", "sex_enc", "Height", "Weight"]].values.astype(np.float32)
    smi         = df["SMI"].values.astype(np.float32)
    imata       = df["IMATA"].values.astype(np.float32)
    sex         = df["PatientSex"].values
    patient_ids = df["PatientID"].values
    return X, smi, imata, sex, patient_ids


def split_data(X, smi, imata, sex, patient_ids):
    """txt 파일이 있으면 저장된 ID로 split 복원, 없으면 stratified split 후 저장.
    (X_cv, smi_cv, imata_cv, sex_cv, X_te, smi_te, imata_te, sex_te) 반환."""
    if os.path.exists(SPLIT_TRAIN_ID_PATH) and os.path.exists(SPLIT_TEST_ID_PATH):
        print(f"[split_data] 저장된 split 파일 로드: {SPLIT_TRAIN_ID_PATH}")
        train_ids = _load_ids(SPLIT_TRAIN_ID_PATH)
        test_ids  = _load_ids(SPLIT_TEST_ID_PATH)
        cv_idx, te_idx = _ids_to_indices(patient_ids, train_ids, test_ids)
    else:
        print("[split_data] split 파일 없음 → 새로 분할 후 저장")
        # stratification용 임시 레이블: 전체 데이터 25th percentile 기준
        y_tmp = _make_labels_tmp(smi, sex)
        cv_idx, te_idx = _stratified_split(X, y_tmp, sex)
        _save_split_ids(patient_ids[cv_idx], patient_ids[te_idx])

    return (
        X[cv_idx], smi[cv_idx], imata[cv_idx], sex[cv_idx],
        X[te_idx], smi[te_idx], imata[te_idx], sex[te_idx],
    )


def compute_thresholds(smi_cv, sex_cv):
    """Train set SMI의 성별별 하위 25th percentile을 임계값으로 반환."""
    thresh_m = float(np.percentile(smi_cv[sex_cv == "M"], 25))
    thresh_f = float(np.percentile(smi_cv[sex_cv == "F"], 25))
    print(f"[compute_thresholds] M Q1={thresh_m:.4f}, F Q1={thresh_f:.4f}")
    return thresh_m, thresh_f


def make_labels(smi, sex, thresh_m, thresh_f):
    """SMI ≤ 임계값이면 1(근감소증), 초과면 0."""
    thresh = np.where(sex == "M", thresh_m, thresh_f)
    return np.where(smi <= thresh, 1, 0).astype(np.int64)


def _make_labels_tmp(smi, sex):
    """전체 데이터 25th percentile 기준 임시 레이블 (stratification 전용)."""
    thresh_m = np.percentile(smi[sex == "M"], 25)
    thresh_f = np.percentile(smi[sex == "F"], 25)
    thresh = np.where(sex == "M", thresh_m, thresh_f)
    return np.where(smi <= thresh, 1, 0).astype(np.int64)


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