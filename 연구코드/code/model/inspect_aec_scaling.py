"""
AEC 스케일링 비교 — raw / StandardScaler / StandardScaler+row-norm 세 버전을
xlsx 파일의 각 시트에 저장한다.

출력 파일: 연구코드/results/aec_scaling_compare_aec{N}.xlsx
  시트 raw          : 원본 AEC 값
  시트 std_scaled   : StandardScaler 열 방향 표준화 적용
  시트 std_norm     : StandardScaler 후 행 방향 z-score 정규화 추가 적용

첫 3열(PatientID, label, sex)은 공통 식별자.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_PATH, AEC_SIZES, RESULTS_DIR
from data import load_data_with_aec


def _row_normalize(X: np.ndarray) -> np.ndarray:
    """행 방향 z-score 정규화."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mu) / sd).astype(np.float32)


def _aec_cols(n: int) -> list[str]:
    return [f"pos_{i+1}" for i in range(n)]


def _load_meta_ids(aec_sheet: str) -> pd.DataFrame:
    """merge 기준 PatientID·label·sex를 반환 (AEC 컬럼 제외)."""
    df_meta = pd.read_excel(DATA_PATH, sheet_name="metadata")
    df_meta["PatientID"] = df_meta["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    df_aec = pd.read_excel(DATA_PATH, sheet_name=aec_sheet)
    df_aec["PatientID"] = df_aec["PatientID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    from config import SMI_THRESH_M, SMI_THRESH_F
    base_cols = ["PatientID", "PatientAge", "PatientSex", "BMI", "SMI"]
    df_meta = df_meta[base_cols].dropna().reset_index(drop=True)

    df_merged = pd.merge(df_meta, df_aec[["PatientID"]], on="PatientID", how="inner").reset_index(drop=True)
    df_merged["label"] = df_merged.apply(
        lambda r: 1 if r["SMI"] <= (SMI_THRESH_M if r["PatientSex"] == "M" else SMI_THRESH_F) else 0,
        axis=1,
    )
    return df_merged[["PatientID", "label", "PatientSex"]]


def save_aec_comparison(aec_size: int) -> None:
    aec_sheet = f"aec_{aec_size}"
    print(f"[aec{aec_size}] Loading data ...")

    X_clin, X_aec_raw, y, sex = load_data_with_aec(aec_len=aec_size, aec_sheet=aec_sheet)
    meta_df = _load_meta_ids(aec_sheet)
    # load_data_with_aec 내부와 동일한 inner join 순서이므로 행 수가 일치
    assert len(meta_df) == len(X_aec_raw), "PatientID 행 수 불일치"

    col_names = _aec_cols(X_aec_raw.shape[1])

    # ── StandardScaler 열 방향 ─────────────────────────────────
    sc = StandardScaler()
    X_aec_std = sc.fit_transform(X_aec_raw).astype(np.float32)

    # ── StandardScaler 후 행 방향 z-score ─────────────────────
    X_aec_std_norm = _row_normalize(X_aec_std)

    # ── xlsx 저장 ──────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(RESULTS_DIR), "aec_inspection")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"aec_scaling_compare_aec{aec_size}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, X in [
            ("raw",       X_aec_raw),
            ("std_scaled", X_aec_std),
            ("std_norm",  X_aec_std_norm),
        ]:
            df_sheet = pd.concat(
                [meta_df.reset_index(drop=True),
                 pd.DataFrame(X, columns=col_names)],
                axis=1,
            )
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  sheet '{sheet_name}' saved  ({df_sheet.shape[0]} rows × {df_sheet.shape[1]} cols)")

    print(f"[aec{aec_size}] Saved → {out_path}\n")


if __name__ == "__main__":
    for aec_size in AEC_SIZES:
        save_aec_comparison(aec_size)
