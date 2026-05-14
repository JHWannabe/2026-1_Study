import os
import json
import pydicom
import pandas as pd
import numpy as np
from tqdm import tqdm

SITE = "강남"
DICOM_BASE   = rf"D:/영상제공/{SITE}/{SITE}_axial"
META_PATH    = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_metadata.xlsx"
AECRAW_PATH  = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_aec_raw.xlsx"
CHECKPOINT   = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\.extract_checkpoint.json"
BATCH_SIZE   = 100

# 기존 metadata에서 임상 컬럼 보존 (DICOM에 없는 값들)
CLINICAL_COLS = ["TAMA", "IMATA", "신장", "체중", "BMI", "SMI"]

# ── PatientID → DICOM 폴더 매핑 ({idx}_{PatientID}_{date}_CT) ─────────────
folder_map = {}
for folder_name in os.listdir(DICOM_BASE):
    parts = folder_name.split("_")
    if len(parts) >= 2:
        folder_map[parts[1]] = os.path.join(DICOM_BASE, folder_name)

print(f"총 DICOM 폴더 수: {len(folder_map)}")


def extract_patient(pid_str):
    """
    반환: (meta_dict, aec_values_sorted_by_z)
    meta_dict keys: PatientAge, PatientSex, kVp, mAs, n_slices, z_range_mm,
                    SeriesDescription, ManufacturerModelName
    """
    if pid_str not in folder_map:
        return None, None

    patient_folder = folder_map[pid_str]
    subfolders = [s for s in os.listdir(patient_folder)]
    if not subfolders:
        return None, None

    dcm_dir = os.path.join(patient_folder, subfolders[0])
    dcm_files = [f for f in os.listdir(dcm_dir) if not f.startswith(".")]
    if not dcm_files:
        return None, None

    slice_data = []  # (z, mA)
    meta_dict = {}
    header_read = False

    for fname in dcm_files:
        try:
            dcm = pydicom.dcmread(
                os.path.join(dcm_dir, fname),
                stop_before_pixels=True,
            )
            # 스캔 레벨 메타데이터는 첫 슬라이스에서만 읽기
            if not header_read:
                age_raw = str(getattr(dcm, "PatientAge", ""))
                age_num = ''.join(filter(str.isdigit, age_raw))
                meta_dict["PatientAge"] = int(age_num) if age_num else np.nan
                meta_dict["PatientSex"] = str(getattr(dcm, "PatientSex", ""))
                meta_dict["kVp"] = float(getattr(dcm, "KVP", np.nan))
                meta_dict["SeriesDescription"] = str(getattr(dcm, "SeriesDescription", ""))
                meta_dict["ManufacturerModelName"] = str(getattr(dcm, "ManufacturerModelName", ""))
                header_read = True

            # z 위치
            if hasattr(dcm, "ImagePositionPatient"):
                z = float(dcm.ImagePositionPatient[2])
            elif hasattr(dcm, "SliceLocation"):
                z = float(dcm.SliceLocation)
            else:
                continue

            mA = float(getattr(dcm, "XRayTubeCurrent", np.nan))
            slice_data.append((z, mA))
        except Exception:
            continue

    if not slice_data:
        return None, None

    # z 기준 정렬
    slice_data.sort(key=lambda x: x[0])
    z_vals = [s[0] for s in slice_data]
    mA_vals = [s[1] for s in slice_data]

    meta_dict["n_slices"] = len(slice_data)
    meta_dict["z_range_mm"] = abs(max(z_vals) - min(z_vals)) if len(z_vals) >= 2 else np.nan

    return meta_dict, mA_vals


def save_checkpoint(processed, meta_rows, aec_rows):
    def clean(v):
        return None if isinstance(v, float) and np.isnan(v) else v

    data = {
        "processed": processed,
        "meta_rows": [
            {k: clean(v) for k, v in row.items()} for row in meta_rows
        ],
        "aec_rows": [
            [clean(v) for v in row] for row in aec_rows
        ],
    }
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return 0, [], []
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta_rows = [
        {k: (np.nan if v is None else v) for k, v in row.items()}
        for row in data["meta_rows"]
    ]
    aec_rows = [
        [(np.nan if v is None else v) for v in row]
        for row in data["aec_rows"]
    ]
    return data["processed"], meta_rows, aec_rows


def write_batch(patient_ids_all, meta_rows, aec_rows, clinical_df):
    n_done = len(meta_rows)

    # ── metadata ────────────────────────────────────────────────────────────
    meta_df = pd.DataFrame(meta_rows)
    meta_df.insert(0, "PatientID", patient_ids_all[:n_done])

    # 임상 컬럼 merge (PatientID 타입 통일)
    meta_df["PatientID"] = meta_df["PatientID"].astype(str)
    clin = clinical_df[["PatientID"] + CLINICAL_COLS].copy()
    clin["PatientID"] = clin["PatientID"].astype(str)
    meta_df = meta_df.merge(clin, on="PatientID", how="left")
    # 컬럼 순서 맞추기 (중복 없이)
    col_order = ["PatientID", "PatientAge", "PatientSex", "kVp",
                 "n_slices", "z_range_mm",
                 "SeriesDescription", "ManufacturerModelName"] + CLINICAL_COLS
    col_order = list(dict.fromkeys(c for c in col_order if c in meta_df.columns))
    remaining = [c for c in meta_df.columns if c not in col_order]
    meta_df = meta_df[col_order + remaining]
    float_cols = meta_df.select_dtypes(include="float").columns
    meta_df[float_cols] = meta_df[float_cols].round(2)
    meta_df.to_excel(META_PATH, index=False)

    # ── aec_raw ─────────────────────────────────────────────────────────────
    max_slices = max(len(r) for r in aec_rows) if aec_rows else 0
    slice_cols = [f"slice_{i+1}" for i in range(max_slices)]
    aec_data = []
    for pid, vals in zip(patient_ids_all[:n_done], aec_rows):
        padded = vals + [np.nan] * (max_slices - len(vals))
        aec_data.append([pid] + padded)
    aec_df = pd.DataFrame(aec_data, columns=["PatientID"] + slice_cols)
    float_cols = aec_df.select_dtypes(include="float").columns
    aec_df[float_cols] = aec_df[float_cols].round(2)
    aec_df.to_excel(AECRAW_PATH, index=False)


# ── 기존 metadata에서 임상 컬럼 로드 ─────────────────────────────────────────
clinical_df = pd.read_excel(META_PATH)

# ── 처리할 PatientID 목록 (기존 metadata 기준) ────────────────────────────────
all_pids = clinical_df["PatientID"].astype(str).tolist()
total = len(all_pids)

# ── 체크포인트 로드 ──────────────────────────────────────────────────────────
start_idx, meta_rows, aec_rows = load_checkpoint()
if start_idx > 0:
    print(f"체크포인트 감지: {start_idx}/{total}번째부터 재시작")

# ── 메인 루프 ────────────────────────────────────────────────────────────────
for i in tqdm(range(start_idx, total), desc="DICOM 추출", initial=start_idx, total=total):
    pid = all_pids[i]
    meta_dict, mA_vals = extract_patient(pid)

    if meta_dict is None:
        meta_dict = {
            "PatientAge": np.nan, "PatientSex": np.nan,
            "kVp": np.nan,
            "n_slices": np.nan, "z_range_mm": np.nan,
            "SeriesDescription": np.nan, "ManufacturerModelName": np.nan,
        }
        mA_vals = []

    meta_rows.append(meta_dict)
    aec_rows.append(mA_vals)

    if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
        save_checkpoint(i + 1, meta_rows, aec_rows)
        write_batch(all_pids, meta_rows, aec_rows, clinical_df)
        tqdm.write(f"  [{i + 1}/{total}] 저장 완료")

# ── 완료 ────────────────────────────────────────────────────────────────────
meta_final = pd.read_excel(META_PATH)
aec_final  = pd.read_excel(AECRAW_PATH)

print(f"\n메타데이터: {meta_final.shape}")
print(f"  n_slices NaN:    {meta_final['n_slices'].isna().sum()}")
print(f"  z_range_mm NaN:  {meta_final['z_range_mm'].isna().sum()}")
print(f"AEC raw: {aec_final.shape}")

if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

print("\n저장 완료")
print(f"  {META_PATH}")
print(f"  {AECRAW_PATH}")
