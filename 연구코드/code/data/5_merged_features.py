"""
[전체 목적]
SMI 메타데이터({SITE}_DLO_Results_SMI_kVp100.xlsx)와
AEC crop 데이터({SITE}_aec_cropped.xlsx)를 통합해
모델 학습에 사용할 최종 피처 파일({SITE}_liver_merged_features.xlsx)을 생성한다.
신촌 → 강남 순서로 순차 처리한다.

[처리 흐름]
  STEP 1. 데이터 로드 (SMI 메타데이터 + AEC 전체 시트)
  STEP 2. 공통 (PatientID, Manufacturer) 교집합 계산
  STEP 3. 환자 필터링
           ① n_slices_cropped < 100 제거 (AEC 구간 너무 짧음)
           ② z_range IQR 이상치 제거 (Q1-1.5IQR ~ Q3+1.5IQR 범위 밖)
  STEP 4. 최종 교집합으로 필터링 후 시트별 엑셀 저장

[출력 파일: {SITE}_liver_merged_features.xlsx]
  metadata 시트    : PatientID, Age, Sex, kVp, mAs, Manufacturer, TAMA, IMATA, BMI, Height, Weight, SMI
  aec_cropped 시트 : PatientID, Manufacturer, n_slices_cropped, z_range, aec_1 ~ aec_N
  aec_128 시트     : PatientID, Manufacturer, n_slices_cropped, z_range, aec_1 ~ aec_128

"""

import pandas as pd
from pathlib import Path


def process_site(site: str) -> None:
    print(f"\n{'='*60}")
    print(f"  SITE: {site}")
    print(f"{'='*60}")

    ROOT     = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data" / site

    smi_path = DATA_DIR / "metadata" / f"{site}_DLO_Results_SMI_kVp100.xlsx"
    aec_path = DATA_DIR / "aec" / f"{site}_aec_cropped.xlsx"
    out_path = DATA_DIR / f"{site}_liver_merged_features.xlsx"

    # ── STEP 1: 데이터 로드 ───────────────────────────────────────────────────

    df_smi = pd.read_excel(smi_path)

    # sheet_name=None → 모든 시트를 {시트명: DataFrame} 딕셔너리로 로드
    aec_sheets = pd.read_excel(aec_path, sheet_name=None)

    # ── STEP 2: 공통 (PatientID, Manufacturer) 교집합 계산 ───────────────────
    # 동일 PatientID가 장비별로 중복될 수 있어 Manufacturer도 키로 사용한다.
    # AEC 파일의 manufacturer_model과 SMI의 Manufacturer가 대응한다.

    smi_keys = set(zip(df_smi["PatientID"], df_smi["Manufacturer"]))

    common_keys = smi_keys
    for df_aec in aec_sheets.values():
        aec_keys = set(zip(df_aec["PatientID"], df_aec["manufacturer_model"]))
        common_keys &= aec_keys

    print(f"초기 공통 (PatientID, Manufacturer) 쌍: {len(common_keys)}개")

    # ── STEP 3: 환자 필터링 ──────────────────────────────────────────────────
    # ── ① n_slices_cropped < 100 제거 ───────────────────────────────────────

    df_aec_cropped = aec_sheets["aec_cropped"]
    df_aec_common  = df_aec_cropped[
        df_aec_cropped.apply(lambda r: (r["PatientID"], r["manufacturer_model"]) in common_keys, axis=1)
    ]
    slices_removed = (df_aec_common["n_slices_cropped"] < 100).sum()
    common_keys = set(
        zip(
            df_aec_common.loc[df_aec_common["n_slices_cropped"] >= 100, "PatientID"],
            df_aec_common.loc[df_aec_common["n_slices_cropped"] >= 100, "manufacturer_model"],
        )
    )

    print(f"n_slices_cropped < 100 제거 {slices_removed}개 → 최종 {len(common_keys)}개")

    # ── ② z_range 이상치 제거 ────────────────────────────────────────────────

    df_aec_common  = df_aec_cropped[
        df_aec_cropped.apply(lambda r: (r["PatientID"], r["manufacturer_model"]) in common_keys, axis=1)
    ]
    q1, q3         = df_aec_common["z_range"].quantile([0.25, 0.75])
    iqr            = q3 - q1
    lower, upper   = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    range_mask     = df_aec_common["z_range"].between(lower, upper)
    range_removed  = (~range_mask).sum()
    common_keys = set(
        zip(
            df_aec_common.loc[range_mask, "PatientID"],
            df_aec_common.loc[range_mask, "manufacturer_model"],
        )
    )

    print(f"z_range 이상치 제거 {range_removed}개 (범위: [{lower:.1f}, {upper:.1f}]) → 최종 {len(common_keys)}개")

    # ── STEP 4: 최종 필터링 및 저장 ──────────────────────────────────────────

    def in_common(row, pid_col, mfr_col):
        return (row[pid_col], row[mfr_col]) in common_keys

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # ── metadata 시트 ────────────────────────────────────────────────────
        mask = df_smi.apply(lambda r: in_common(r, "PatientID", "Manufacturer"), axis=1)
        df_smi_filtered = df_smi[mask].drop(columns=["SRC_Report"], errors="ignore").copy()
        df_smi_filtered.to_excel(writer, sheet_name="metadata", index=False)
        print(f"  [metadata] {len(df_smi_filtered)}행")

        # ── AEC 시트들 (aec_cropped, aec_128) ───────────────────────────────
        for sheet_name, df_aec in aec_sheets.items():
            mask = df_aec.apply(lambda r: in_common(r, "PatientID", "manufacturer_model"), axis=1)
            df_aec_filtered = df_aec[mask].rename(columns={"manufacturer_model": "Manufacturer"}).copy()
            df_aec_filtered.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  [{sheet_name}] {len(df_aec_filtered)}행")

    print(f"저장 완료: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

for site in ["강남"]:
    process_site(site)
