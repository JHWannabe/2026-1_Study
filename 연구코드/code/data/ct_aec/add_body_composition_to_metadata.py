"""
{SITE}_body_composition.xlsx (구간 합산 IMATA/NAMA/LAMA/내장지방/피하지방)를
{SITE}_liver_merged_features.xlsx 의 metadata 시트에 PatientID 기준으로 채워 넣는다.

body_composition_sum.py 배치 도중에도 반복 호출될 수 있으므로, 매번 metadata의 전체 행을
그대로 유지한 채(LEFT JOIN) 아직 seg_status=='ok'가 아닌 환자는 신규 컬럼을 NaN으로 남긴다.
"""

import os
import shutil

import pandas as pd

SITE = "강남"  # __main__ 직접 실행 시에만 사용 (다른 스크립트에서 import 시 merge()에 site를 명시할 것)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = rf"{PROJECT_ROOT}\data"
SITE_EN  = {"강남": "Gangnam", "신촌": "Sinchon"}

NEW_COLS = ["IMATA_sum_cm2", "NAMA_sum_cm2", "LAMA_sum_cm2", "내장지방_sum_cm2", "피하지방_sum_cm2"]


def _lock_file_path(path: str) -> str:
    d, f = os.path.split(path)
    return os.path.join(d, f"~${f}")


def merge(site: str = SITE, backup: bool = True, suffix: str = "", new_cols: list | None = None) -> None:
    cols           = new_cols if new_cols is not None else NEW_COLS
    site_dir       = rf"{DATA_DIR}\{SITE_EN[site]}"
    merged_path    = rf"{site_dir}\{SITE_EN[site]}.xlsx"
    body_comp_path = rf"{site_dir}\{site}_body_composition{suffix}.xlsx"
    backup_path    = rf"{site_dir}\{SITE_EN[site]}_backup_pre_body_comp{suffix}.xlsx"

    lock_path = _lock_file_path(merged_path)
    if os.path.exists(lock_path):
        raise RuntimeError(
            f"'{merged_path}' 파일이 Excel에서 열려 있는 것 같습니다 (lock 파일: {lock_path}). "
            "Excel에서 파일을 닫은 뒤 다시 실행하세요."
        )

    if backup and not os.path.exists(backup_path):
        shutil.copy2(merged_path, backup_path)
        print(f"백업 저장(최초 1회): {backup_path}")

    sheets  = pd.read_excel(merged_path, sheet_name=None)
    df_meta = sheets["metadata"].drop(columns=cols, errors="ignore")
    df_body = pd.read_excel(body_comp_path, sheet_name="body_composition")
    df_body_ok = df_body[df_body["seg_status"] == "ok"][["PatientID"] + cols]

    before = len(df_meta)
    merged = df_meta.merge(df_body_ok, on="PatientID", how="left")
    filled = int(merged[cols[0]].notna().sum())
    print(f"metadata {before}행 유지, 그 중 body_composition 반영 완료 {filled}행 "
          f"(body_composition ok {len(df_body_ok)}행)")

    sheets["metadata"] = merged
    with pd.ExcelWriter(merged_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"저장 완료: {merged_path}")


if __name__ == "__main__":
    merge(SITE)
