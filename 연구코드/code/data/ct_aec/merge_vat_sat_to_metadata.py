"""
vat_sat_volumetric.py가 만든 aec/vat_sat_series.xlsx (시리즈별 volumetric VAT/SAT)를
환자(PatientID)당 ok 시리즈 평균으로 집계하여, aec/aec_cropped_novatsat.xlsx의
metadata_cleaned/metadata_raw 시트에 vat_sum/sat_sum 컬럼으로 채워 넣는다.

환자당 여러 시리즈(Pre Contrast/With Contrast/Arterial/Portal/Delay 등)가 있는 경우
seg_status=='ok'인 시리즈들의 vat_sum/sat_sum(cm³) 평균을 사용한다.
metadata의 행/다른 컬럼은 그대로 유지한 채(LEFT JOIN) 매칭 안 되는 환자는 NaN으로 남긴다.
"""

import os
import shutil

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AEC_DIR      = os.path.join(PROJECT_ROOT, "aec")

SERIES_PATH  = os.path.join(AEC_DIR, "vat_sat_series.xlsx")
TARGET_PATH  = os.path.join(AEC_DIR, "aec_cropped_novatsat.xlsx")
BACKUP_PATH  = os.path.join(AEC_DIR, "aec_cropped_novatsat_backup_pre_vatsat.xlsx")

NEW_COLS      = ["vat_sum", "sat_sum"]
METADATA_SHEETS = ["metadata_cleaned", "metadata_raw"]


def _lock_file_path(path: str) -> str:
    d, f = os.path.split(path)
    return os.path.join(d, f"~${f}")


def build_patient_avg() -> pd.DataFrame:
    df = pd.read_excel(SERIES_PATH, sheet_name="vat_sat_series")
    df_ok = df[df["seg_status"] == "ok"]
    agg = (
        df_ok.groupby("PatientID")[NEW_COLS]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"PatientID": "patientID"})
    )
    return agg


def merge(backup: bool = True) -> None:
    lock_path = _lock_file_path(TARGET_PATH)
    if os.path.exists(lock_path):
        raise RuntimeError(
            f"'{TARGET_PATH}' 파일이 Excel에서 열려 있는 것 같습니다 (lock 파일: {lock_path}). "
            "Excel에서 파일을 닫은 뒤 다시 실행하세요."
        )

    if backup and not os.path.exists(BACKUP_PATH):
        shutil.copy2(TARGET_PATH, BACKUP_PATH)
        print(f"백업 저장(최초 1회): {BACKUP_PATH}")

    patient_avg = build_patient_avg()
    print(f"환자별 평균 집계: {len(patient_avg)}명 (vat_sat_series.xlsx ok 시리즈 기준)")

    sheets = pd.read_excel(TARGET_PATH, sheet_name=None)

    for sheet_name in METADATA_SHEETS:
        df_meta = sheets[sheet_name].drop(columns=NEW_COLS, errors="ignore")
        before  = len(df_meta)
        merged  = df_meta.merge(patient_avg, on="patientID", how="left")
        filled  = int(merged["vat_sum"].notna().sum())
        print(f"[{sheet_name}] {before}행 유지, 그 중 vat_sum/sat_sum 반영 완료 {filled}행")
        sheets[sheet_name] = merged

    with pd.ExcelWriter(TARGET_PATH, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"저장 완료: {TARGET_PATH}")


if __name__ == "__main__":
    merge()
