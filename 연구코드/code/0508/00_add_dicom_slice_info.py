import os
import json
import pydicom
import pandas as pd
import numpy as np
from tqdm import tqdm

DICOM_BASE = r"D:/영상제공/신촌/신촌_axial"
EXCEL_PATH = r"C:/Users/jhjun/OneDrive/Desktop/2026-1_Study/연구코드/data/신촌_merged_features.xlsx"
CHECKPOINT_PATH = r"C:/Users/jhjun/OneDrive/Desktop/2026-1_Study/연구코드/data/.dicom_checkpoint.json"
BATCH_SIZE = 100

# 폴더명에서 PatientID → 폴더 경로 매핑 빌드
folder_map = {}
for folder_name in os.listdir(DICOM_BASE):
    parts = folder_name.split("_")
    if len(parts) >= 2:
        pid = parts[1]
        folder_map[pid] = os.path.join(DICOM_BASE, folder_name)


def get_slice_info(patient_id):
    pid_str = str(patient_id)
    if pid_str not in folder_map:
        return np.nan, np.nan

    patient_folder = folder_map[pid_str]
    subfolders = os.listdir(patient_folder)
    if not subfolders:
        return np.nan, np.nan

    dcm_folder = os.path.join(patient_folder, subfolders[0])
    dcm_files = [f for f in os.listdir(dcm_folder) if not f.startswith(".")]
    n_slices = len(dcm_files)

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
        z_range_mm = abs(max(z_positions) - min(z_positions))
    else:
        z_range_mm = np.nan

    return n_slices, z_range_mm


def save_checkpoint(processed_count, n_slices_list, z_range_list):
    data = {
        "processed_count": processed_count,
        "n_slices_list": [None if (isinstance(v, float) and np.isnan(v)) else v for v in n_slices_list],
        "z_range_list": [None if (isinstance(v, float) and np.isnan(v)) else v for v in z_range_list],
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, [], []
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_slices_list = [np.nan if v is None else v for v in data["n_slices_list"]]
    z_range_list = [np.nan if v is None else v for v in data["z_range_list"]]
    return data["processed_count"], n_slices_list, z_range_list


def write_batch_to_excel(df_bmi, n_slices_list, z_range_list):
    df_out = df_bmi.copy()
    # 처리된 행만 업데이트, 나머지는 NaN 유지
    n_pad = len(df_out) - len(n_slices_list)
    df_out["n_slices"] = n_slices_list + [np.nan] * n_pad
    df_out["z_range_mm"] = z_range_list + [np.nan] * n_pad
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_out.to_excel(writer, sheet_name="metadata-bmi_add", index=False)


# 시트 읽기
xl = pd.ExcelFile(EXCEL_PATH)
df_bmi = xl.parse("metadata-bmi_add")
df_feat = xl.parse("features")

assert len(df_bmi) == len(df_feat), "행 수 불일치"

# 체크포인트 로드
start_idx, n_slices_list, z_range_list = load_checkpoint()
if start_idx > 0:
    print(f"체크포인트 감지: {start_idx}번째 행부터 재시작합니다.")

patient_ids = df_feat["PatientID"].tolist()
total = len(patient_ids)

for i in tqdm(range(start_idx, total), desc="DICOM 처리", initial=start_idx, total=total):
    n, z = get_slice_info(patient_ids[i])
    n_slices_list.append(n)
    z_range_list.append(z)

    # 배치 단위로 저장
    if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
        save_checkpoint(i + 1, n_slices_list, z_range_list)
        write_batch_to_excel(df_bmi, n_slices_list, z_range_list)
        tqdm.write(f"  [{i + 1}/{total}] 저장 완료")

print(f"\nn_slices NaN: {sum(1 for v in n_slices_list if isinstance(v, float) and np.isnan(v))}")
print(f"z_range_mm NaN: {sum(1 for v in z_range_list if isinstance(v, float) and np.isnan(v))}")

# 체크포인트 삭제 (정상 완료)
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("체크포인트 파일 삭제 완료.")

print("\n저장 완료:", EXCEL_PATH)
