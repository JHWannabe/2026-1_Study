import os
import json
import pydicom
import pandas as pd
import numpy as np
from tqdm import tqdm

SITE = "강남"
DICOM_BASE = rf"D:/영상제공/{SITE}/{SITE}_axial"
META_PATH = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_metadata.xlsx"
AEC_PATH = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_aec_raw.xlsx"
CHECKPOINT_PATH = rf"C:/Users/jhjun/OneDrive/Desktop/2026-1_Study/연구코드/data/.meta_checkpoint.json"
BATCH_SIZE = 100

# PatientID → DICOM 폴더 경로 매핑 (폴더명: {idx}_{PatientID}_{date}_CT)
folder_map = {}
for folder_name in os.listdir(DICOM_BASE):
    parts = folder_name.split("_")
    if len(parts) >= 2:
        pid = parts[1]
        folder_map[pid] = os.path.join(DICOM_BASE, folder_name)


def get_z_range(patient_id):
    pid_str = str(patient_id)
    if pid_str not in folder_map:
        return np.nan

    patient_folder = folder_map[pid_str]
    subfolders = os.listdir(patient_folder)
    if not subfolders:
        return np.nan

    dcm_folder = os.path.join(patient_folder, subfolders[0])
    dcm_files = [f for f in os.listdir(dcm_folder) if not f.startswith(".")]

    z_positions = []
    for fname in dcm_files:
        try:
            dcm = pydicom.dcmread(
                os.path.join(dcm_folder, fname),
                stop_before_pixels=True,
            )
            if hasattr(dcm, "ImagePositionPatient"):
                z_positions.append(float(dcm.ImagePositionPatient[2]))
            elif hasattr(dcm, "SliceLocation"):
                z_positions.append(float(dcm.SliceLocation))
        except Exception:
            continue

    if len(z_positions) >= 2:
        return abs(max(z_positions) - min(z_positions))
    return np.nan


def save_checkpoint(processed_count, z_range_list):
    data = {
        "processed_count": processed_count,
        "z_range_list": [None if (isinstance(v, float) and np.isnan(v)) else v for v in z_range_list],
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, []
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    z_range_list = [np.nan if v is None else v for v in data["z_range_list"]]
    return data["processed_count"], z_range_list


def write_to_excel(meta_df, n_slices_all, z_range_partial):
    df_out = meta_df.copy()
    df_out["n_slices"] = n_slices_all
    n_pad = len(df_out) - len(z_range_partial)
    df_out["z_range_mm"] = z_range_partial + [np.nan] * n_pad
    df_out.to_excel(META_PATH, index=False)


# n_slices: aec_raw에서 비-NaN slice 컬럼 수
aec = pd.read_excel(AEC_PATH)
slice_cols = [c for c in aec.columns if c.startswith("slice_")]
aec_n_slices = aec.set_index("PatientID")[slice_cols].notna().sum(axis=1).rename("n_slices")

# metadata 로드
meta = pd.read_excel(META_PATH)

# aec에 없는 PatientID 확인
missing = set(meta["PatientID"]) - set(aec["PatientID"])
if missing:
    print(f"경고: aec_raw에 없는 PatientID {len(missing)}개: {list(missing)[:5]}")

# n_slices 전체 계산 (aec 기준, 즉시)
n_slices_all = meta["PatientID"].map(aec_n_slices).tolist()

# z_range_mm: DICOM에서 계산 (체크포인트 지원)
start_idx, z_range_list = load_checkpoint()
if start_idx > 0:
    print(f"체크포인트 감지: {start_idx}번째 행부터 재시작합니다.")

patient_ids = meta["PatientID"].tolist()
total = len(patient_ids)

for i in tqdm(range(start_idx, total), desc="DICOM z_range 계산", initial=start_idx, total=total):
    z = get_z_range(patient_ids[i])
    z_range_list.append(z)

    if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
        save_checkpoint(i + 1, z_range_list)
        write_to_excel(meta, n_slices_all, z_range_list)
        tqdm.write(f"  [{i + 1}/{total}] Excel 저장 완료")

print(f"\nn_slices NaN: {sum(1 for v in n_slices_all if pd.isna(v))}")
print(f"z_range_mm NaN: {sum(1 for v in z_range_list if isinstance(v, float) and np.isnan(v))}")

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

print("\n저장 완료:", META_PATH)
