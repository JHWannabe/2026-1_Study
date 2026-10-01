"""
aec_cropped_novatsat.xlsx의 aec_1..aec_N 배열 방향을 pubis->liver에서 liver->pubis로 뒤집는다
(Clinical_Study의 gangnam/sinchon_final_dataset.xlsx와 방향을 통일하기 위함).

aec_cropped/aec_cropped_contrast 시트는 행마다 n_slices_cropped만큼만 값이 있고 나머지는
NaN 패딩이므로, 유효 구간(aec_1..aec_{n_slices_cropped})만 뒤집고 패딩 위치는 그대로 둔다.
aec_128/aec_128_contrast는 고정 128포인트 보간값이라 전체를 그대로 뒤집는다.
"""

import os
import shutil

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_PATH  = os.path.join(PROJECT_ROOT, "aec", "aec_cropped_novatsat.xlsx")
BACKUP_PATH  = os.path.join(PROJECT_ROOT, "aec", "aec_cropped_novatsat_backup_pre_reverse.xlsx")

VARIABLE_LEN_SHEETS = ["aec_cropped", "aec_cropped_contrast"]
FIXED_LEN_SHEETS    = ["aec_128", "aec_128_contrast"]


def _lock_file_path(path: str) -> str:
    d, f = os.path.split(path)
    return os.path.join(d, f"~${f}")


def _aec_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("aec_") and c[4:].isdigit()]
    return sorted(cols, key=lambda c: int(c[4:]))


def reverse_variable_len(df: pd.DataFrame) -> pd.DataFrame:
    aec_cols = _aec_cols(df)
    arr = df[aec_cols].to_numpy(dtype=float)
    n_slices = df["n_slices_cropped"].to_numpy(dtype=int)
    for i, n in enumerate(n_slices):
        arr[i, :n] = arr[i, :n][::-1]
    df[aec_cols] = arr
    return df


def reverse_fixed_len(df: pd.DataFrame) -> pd.DataFrame:
    aec_cols = _aec_cols(df)
    df[aec_cols] = df[aec_cols].to_numpy(dtype=float)[:, ::-1]
    return df


def main():
    lock_path = _lock_file_path(TARGET_PATH)
    if os.path.exists(lock_path):
        raise RuntimeError(
            f"'{TARGET_PATH}' 파일이 Excel에서 열려 있는 것 같습니다 (lock 파일: {lock_path}). "
            "Excel에서 파일을 닫은 뒤 다시 실행하세요."
        )

    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(TARGET_PATH, BACKUP_PATH)
        print(f"백업 저장(최초 1회): {BACKUP_PATH}")

    sheets = pd.read_excel(TARGET_PATH, sheet_name=None)

    for sheet_name in VARIABLE_LEN_SHEETS:
        if sheet_name in sheets:
            sheets[sheet_name] = reverse_variable_len(sheets[sheet_name])
            print(f"[{sheet_name}] {len(sheets[sheet_name])}행 방향 반전 완료 (유효 구간만)")

    for sheet_name in FIXED_LEN_SHEETS:
        if sheet_name in sheets:
            sheets[sheet_name] = reverse_fixed_len(sheets[sheet_name])
            print(f"[{sheet_name}] {len(sheets[sheet_name])}행 방향 반전 완료 (128포인트 전체)")

    with pd.ExcelWriter(TARGET_PATH, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"저장 완료: {TARGET_PATH}")


if __name__ == "__main__":
    main()
