import os
import pydicom
import pandas as pd
from tqdm import tqdm

SITES = {
    "신촌": {
        "dicom_dir":  r"D:\영상제공\신촌\신촌_원본",
        "clinic_xlsx": r"D:\영상제공\신촌_merged_features.xlsx",
        "aec_xlsx":    r"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\신촌\aec\신촌_aec_total.xlsx",
    },
    "강남": {
        "dicom_dir":  r"D:\영상제공\강남\강남_원본",
        "clinic_xlsx": r"D:\영상제공\강남_merged_features.xlsx",
        "aec_xlsx":    r"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\강남\aec\강남_aec_total.xlsx",
    },
}

OUTPUT_XLSX = r"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\data_checklist.xlsx"

all_rows = []

for site_name, cfg in SITES.items():
    print(f"\n{'='*40}")
    print(f"[{site_name}] 처리 시작")

    # ── Clinic / AEC ID 세트 ──────────────────────────────────────
    clinic_ids = set(
        pd.read_excel(cfg["clinic_xlsx"])["PatientID"].astype(str)
    )
    aec_ids = set(
        pd.read_excel(cfg["aec_xlsx"])["PatientID"].astype(str)
    )

    # ── DICOM 원본 환자 목록 ──────────────────────────────────────
    dicom_patients = sorted(
        p for p in os.listdir(cfg["dicom_dir"])
        if os.path.isdir(os.path.join(cfg["dicom_dir"], p))
    )

    # 세 소스의 합집합을 기준 목록으로
    all_pids = sorted(set(dicom_patients) | clinic_ids | aec_ids)
    print(f"  DICOM 원본: {len(dicom_patients)}명 / Clinic: {len(clinic_ids)}명 / AEC: {len(aec_ids)}명")
    print(f"  전체(합집합): {len(all_pids)}명")

    # ── Scout 탐색 (DICOM 원본에 있는 환자만) ─────────────────────
    scout_patients = set()
    dicom_set = set(dicom_patients)

    for i, patient_id in enumerate(tqdm(dicom_patients, desc=f"{site_name} Scout 탐색"), start=1):
        patient_dir = os.path.join(cfg["dicom_dir"], patient_id)
        for root, dirs, files in os.walk(patient_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                    series_desc = str(getattr(ds, "SeriesDescription", "")).upper()
                    image_type  = [str(t).upper() for t in getattr(ds, "ImageType", [])]
                    if "SCOUT" in series_desc or "SCOUT" in image_type:
                        scout_patients.add(patient_id)
                        break  # 이 파일에서 발견됐으면 해당 환자 확정
                except Exception:
                    pass
            if patient_id in scout_patients:
                break  # 환자 폴더 탐색 중단

        if i % 50 == 0:
            print(f"  [{i}/{len(dicom_patients)}명] Scout 발견: {len(scout_patients)}명")

    print(f"  Scout 보유: {len(scout_patients)}명")

    # ── 행 생성 ──────────────────────────────────────────────────
    for pid in all_pids:
        all_rows.append({
            "Site":        site_name,
            "PatientID":   pid,
            "Clinic Data": "O" if pid in clinic_ids else "X",
            "AEC Data":    "O" if pid in aec_ids    else "X",
            "Scout Data":  "O" if pid in scout_patients else ("X" if pid in dicom_set else "-"),
        })

# ── 엑셀 저장 ────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    # 시트 1: 전체
    df.to_excel(writer, sheet_name="전체", index=False)

    # 시트 2~3: 사이트별
    for site_name in SITES:
        df[df["Site"] == site_name].to_excel(writer, sheet_name=site_name, index=False)

print(f"\n저장 완료: {OUTPUT_XLSX}")
print(f"전체 행수: {len(df)}")
print(df.groupby("Site")[["Clinic Data","AEC Data","Scout Data"]].apply(lambda g: g.apply(lambda c: c.value_counts().to_dict())).to_string())
