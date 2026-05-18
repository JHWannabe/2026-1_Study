"""
TotalSegmentator를 이용해 환자별 해부학적 경계를 DICOM Instance Number로 추출한다.

  t12_slice  : T12 상단 슬라이스의 DICOM Instance Number
  pubis_slice: 두덩뼈 하단 슬라이스의 DICOM Instance Number
  n_slices   : 스캔 전체 슬라이스 수
  slices     : 해부학적 클리핑 구간(T12 상단~두덩뼈 하단) 슬라이스 수
  z_t12      : T12 상단 슬라이스의 DICOM z 좌표
  z_pubis    : 두덩뼈 하단 슬라이스의 DICOM z 좌표

사용 구조:
  - 상단: vertebrae_T12 cranial end
  - 하단: hip_left + hip_right caudal end

출력:
  - {SITE}_z_bounds.xlsx
  - data/{SITE}/upper/{PatientID}.png   (T12 상단 슬라이스)
  - data/{SITE}/bottom/{PatientID}.png  (두덩뼈 하단 슬라이스)
"""

import os
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import pydicom
from PIL import Image
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from totalsegmentator.python_api import totalsegmentator

# ── 설정 ─────────────────────────────────────────────────────────────────────
SITE       = "강남"
DICOM_BASE = rf"D:/영상제공/{SITE}/{SITE}_axial"
META_PATH  = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_merged_features.xlsx"
OUT_PATH   = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_z_bounds.xlsx"
CHECKPOINT = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\.z_bounds_checkpoint.json"
UPPER_DIR  = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\upper"
BOTTOM_DIR = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\bottom"

ROI_SUBSET = ["vertebrae_T12", "hip_left", "hip_right"]
BATCH_SIZE = 5


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def build_folder_maps(dicom_base: str) -> tuple[dict, dict]:
    """PatientID → DICOM 폴더 경로, No 매핑."""
    folder_map: dict[str, str] = {}
    no_map: dict[str, int] = {}
    if not os.path.isdir(dicom_base):
        raise FileNotFoundError(f"DICOM 경로 없음: {dicom_base}")
    for folder_name in os.listdir(dicom_base):
        parts = folder_name.split("_")
        if len(parts) >= 2:
            pid = parts[1]
            folder_map[pid] = os.path.join(dicom_base, folder_name)
            try:
                no_map[pid] = int(parts[0])
            except ValueError:
                pass
    return folder_map, no_map


def dicom_to_nifti_with_mapping(dcm_dir: str, out_path: str) -> list | None:
    """DICOM을 NIfTI로 변환하고, NIfTI k-인덱스 순서와 일치하는 DICOM 파일 경로 리스트를 반환합니다."""
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dcm_dir)
        if not series_ids:
            return None
        
        # SimpleITK가 정렬하는 기준(지하학적 위치 z값 기준 정렬)으로 파일명 리스트 확보
        dicom_names = reader.GetGDCMSeriesFileNames(dcm_dir, series_ids[0])
        reader.SetFileNames(dicom_names)
        
        # NIfTI 저장 (SimpleITK는 기본적으로 z축 오름차순 또는 이미지 방향성에 맞춰 볼륨 생성)
        image = reader.Execute()
        sitk.WriteImage(image, out_path)
        
        # 변환된 NIfTI 배열의 k축 정렬 레이어와 dicom_names의 매핑 순서는 일치함
        return list(dicom_names)
    except Exception:
        return None


def get_k_indices(mask_path: str) -> np.ndarray:
    """마스크 NIfTI에서 양성 복셀의 k(슬라이스) 인덱스 배열 반환."""
    if not os.path.exists(mask_path):
        return np.array([])
    data = nib.load(mask_path).get_fdata()
    idx  = np.argwhere(data > 0.5)
    return idx[:, 2].astype(int) if len(idx) > 0 else np.array([])


def get_instance_number(dcm_path: str) -> int | None:
    """지정된 DICOM 파일에서 Instance Number 헤더값을 추출합니다."""
    try:
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return int(dcm.InstanceNumber)
    except Exception:
        return None


def _read_z(dcm_path: str) -> float:
    """DICOM 파일에서 z 좌표(ImagePositionPatient[2])를 반환. 실패 시 nan."""
    try:
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return float(dcm.ImagePositionPatient[2])
    except Exception:
        return float(np.nan)


def _window_to_uint8(pixels: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(pixels, low, high)
    return ((clipped - low) / max(high - low, 1e-6) * 255).astype(np.uint8)


def save_slice_png(dcm_path: str, out_path: str) -> bool:
    """DICOM 슬라이스 1장을 8-bit PNG로 저장."""
    try:
        dcm = pydicom.dcmread(dcm_path)
        pixels = dcm.pixel_array.astype(np.float32)
        slope = float(getattr(dcm, "RescaleSlope", 1) or 1)
        intercept = float(getattr(dcm, "RescaleIntercept", 0) or 0)
        pixels = pixels * slope + intercept

        wc = getattr(dcm, "WindowCenter", None)
        ww = getattr(dcm, "WindowWidth", None)
        if wc is not None and ww is not None:
            wc = float(wc[0]) if hasattr(wc, "__iter__") and not isinstance(wc, str) else float(wc)
            ww = float(ww[0]) if hasattr(ww, "__iter__") and not isinstance(ww, str) else float(ww)
            img = _window_to_uint8(pixels, wc, ww)
        else:
            lo, hi = np.percentile(pixels, [1, 99])
            img = _window_to_uint8(pixels, (lo + hi) / 2, hi - lo)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(img).save(out_path)
        return True
    except Exception:
        return False


def save_boundary_previews(
    pid_str: str,
    upper_dcm: str | None,
    bottom_dcm: str | None,
) -> tuple[bool, bool]:
    """T12 upper / pubis bottom 슬라이스를 각각 upper, bottom 폴더에 저장."""
    upper_ok = bottom_ok = False
    if upper_dcm:
        upper_ok = save_slice_png(upper_dcm, os.path.join(UPPER_DIR, f"{pid_str}.png"))
    if bottom_dcm:
        bottom_ok = save_slice_png(bottom_dcm, os.path.join(BOTTOM_DIR, f"{pid_str}.png"))
    return upper_ok, bottom_ok


def extract_bounds(pid_str: str, tmp_dir: str, folder_map: dict) -> dict:
    """환자 1명의 해부학적 경계면의 실제 DICOM Instance Number를 추출한다."""
    result = {
        "t12_slice":  np.nan,
        "pubis_slice": np.nan,
        "slices":     np.nan,
        "n_slices":   np.nan,
        "z_t12":      np.nan,
        "z_pubis":    np.nan,
        "z_range":    np.nan,
        "seg_status": "failed",
    }

    if pid_str not in folder_map:
        result["seg_status"] = "no_dicom"
        return result

    patient_folder = folder_map[pid_str]
    subfolders = [s for s in os.listdir(patient_folder)
                  if os.path.isdir(os.path.join(patient_folder, s))]
    if not subfolders:
        result["seg_status"] = "no_series"
        return result

    dcm_dir  = os.path.join(patient_folder, subfolders[0])
    nii_path = os.path.join(tmp_dir, f"{pid_str}.nii.gz")
    seg_dir  = os.path.join(tmp_dir, f"{pid_str}_seg")

    try:
        # 1) DICOM → NIfTI 변환 및 매핑용 파일 순서 획득
        dcm_mapping_files = dicom_to_nifti_with_mapping(dcm_dir, nii_path)
        if not dcm_mapping_files:
            result["seg_status"] = "nifti_fail"
            return result

        # 2) TotalSegmentator 실행
        os.makedirs(seg_dir, exist_ok=True)
        totalsegmentator(
            input=nii_path, output=seg_dir,
            fast=True, roi_subset=ROI_SUBSET,
            quiet=True, device="gpu",
        )

        n_k = int(nib.load(nii_path).shape[2])
        result["n_slices"] = n_k
        if n_k < 2 or len(dcm_mapping_files) != n_k:
            result["seg_status"] = "invalid_volume_match"
            return result

        # 3) 해부학적 마스크 좌표 추출 (정수형 k 인덱스 사용)
        t12_k = get_k_indices(os.path.join(seg_dir, "vertebrae_T12.nii.gz"))
        
        hip_k = np.array([], dtype=int)
        for side in ("hip_left", "hip_right"):
            p = os.path.join(seg_dir, f"{side}.nii.gz")
            if os.path.exists(p):
                hip_k = np.concatenate([hip_k, get_k_indices(p)]).astype(int)

        t12_found = len(t12_k) > 0
        hip_found = len(hip_k) > 0

        # 4) 해부학적 상단(Cranial)과 하단(Caudal)을 결정하기 위한 방향성 판별
        # 4) 방향 결정: T12(cranial)+hip(caudal) 둘 다 있으면 상대 위치로, 아니면 구조 위치 기반 휴리스틱
        if t12_found and hip_found:
            cranial_is_higher_k = np.median(t12_k) > np.median(hip_k)
        elif t12_found:
            # T12는 스캔의 cranial 절반에 있어야 함
            cranial_is_higher_k = np.median(t12_k) > (n_k / 2)
        elif hip_found:
            # Hip은 스캔의 caudal 절반에 있어야 함
            cranial_is_higher_k = np.median(hip_k) < (n_k / 2)
        else:
            cranial_is_higher_k = True  # 기본값 (양쪽 모두 없음)

        # 5) 각 경계 k 인덱스 결정 — missing 시 스캔 끝단 사용
        upper_dcm = bottom_dcm = None

        # T12 상단: 마스크 있으면 cranial 끝, 없으면 스캔 cranial 끝(k_max)
        if t12_found:
            apex_k = int(t12_k.max() if cranial_is_higher_k else t12_k.min())
        else:
            apex_k = n_k - 1 if cranial_is_higher_k else 0
        apex_k = max(0, min(apex_k, n_k - 1))
        upper_dcm = dcm_mapping_files[apex_k]
        inst_num  = get_instance_number(upper_dcm)
        result["z_t12"]    = _read_z(upper_dcm)
        result["t12_slice"] = inst_num if inst_num is not None else np.nan

        # 두덩뼈 하단: 마스크 있으면 caudal 끝, 없으면 스캔 caudal 끝(k_min)
        if hip_found:
            bottom_k = int(hip_k.min() if cranial_is_higher_k else hip_k.max())
        else:
            bottom_k = 0 if cranial_is_higher_k else n_k - 1
        bottom_k   = max(0, min(bottom_k, n_k - 1))
        bottom_dcm = dcm_mapping_files[bottom_k]
        inst_num   = get_instance_number(bottom_dcm)
        result["pubis_slice"] = inst_num if inst_num is not None else np.nan
        result["z_pubis"]     = _read_z(bottom_dcm)

        result["slices"] = abs(apex_k - bottom_k) + 1

        if pd.notna(result["z_t12"]) and pd.notna(result["z_pubis"]):
            result["z_range"] = round(abs(result["z_t12"] - result["z_pubis"]), 2)
        save_boundary_previews(pid_str, upper_dcm, bottom_dcm)

        # 6) 세그멘테이션 상태 기록
        if t12_found and hip_found:
            if pd.isna(result["t12_slice"]) or pd.isna(result["pubis_slice"]):
                result["seg_status"] = "instance_number_missing"
            else:
                result["seg_status"] = "ok"
        elif t12_found:
            result["seg_status"] = "hip_missing"
        elif hip_found:
            result["seg_status"] = "t12_missing"
        else:
            result["seg_status"] = "both_missing"

    except Exception as e:
        result["seg_status"] = f"error:{type(e).__name__}:{e}"

    finally:
        for path in (nii_path, seg_dir):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    return result


# ── 체크포인트 ────────────────────────────────────────────────────────────────

def save_checkpoint(processed: int, rows: list):
    def clean(v):
        return None if isinstance(v, float) and np.isnan(v) else v
    data = {"processed": processed,
            "rows": [{k: clean(v) for k, v in r.items()} for r in rows]}
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return 0, []
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = [{k: (np.nan if v is None else v) for k, v in r.items()}
            for r in data["rows"]]
    return data["processed"], rows


def write_output(all_pids: list, rows: list, no_map: dict):
    n  = len(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "PatientID", [int(p) for p in all_pids[:n]])
    df.insert(0, "No", [no_map.get(str(p), np.nan) for p in all_pids[:n]])
    int_cols = ["No", "t12_slice", "pubis_slice", "slices", "n_slices"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    col_order = ["No", "PatientID", "n_slices", "slices",
                 "t12_slice", "pubis_slice", "z_range", "z_t12", "z_pubis", "seg_status"]
    cols = ([c for c in col_order if c in df.columns] +
            [c for c in df.columns if c not in col_order])
    df = df[cols]
    df.to_excel(OUT_PATH, index=False)


def main():
    os.makedirs(UPPER_DIR, exist_ok=True)
    os.makedirs(BOTTOM_DIR, exist_ok=True)

    folder_map, no_map = build_folder_maps(DICOM_BASE)
    print(f"DICOM 폴더 수: {len(folder_map)}")
    print(f"미리보기 저장: {UPPER_DIR}")
    print(f"              {BOTTOM_DIR}")

    clinical_df = pd.read_excel(META_PATH, sheet_name="metadata")
    all_pids    = clinical_df["PatientID"].astype(str).tolist()
    total       = len(all_pids)

    start_idx, rows = load_checkpoint()
    if start_idx > 0:
        print(f"체크포인트 감지: {start_idx}/{total}번째부터 재시작")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in tqdm(range(start_idx, total), desc="경계 추출", initial=start_idx, total=total):
            pid    = all_pids[i]
            result = extract_bounds(pid, tmp_dir, folder_map)
            rows.append(result)

            if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
                save_checkpoint(i + 1, rows)
                write_output(all_pids, rows, no_map)
                status_counts = pd.Series([r["seg_status"] for r in rows]).value_counts().to_dict()
                tqdm.write(f"  [{i+1}/{total}] {status_counts}")

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    df_out = pd.read_excel(OUT_PATH)
    print(f"\n[완료] {OUT_PATH}")
    print(f"  shape : {df_out.shape}")
    print(f"  상태 :\n{df_out['seg_status'].value_counts().to_string()}")
    if not df_out.empty:
        if "t12_slice" in df_out.columns and not df_out["t12_slice"].isna().all():
            print(f"  t12_slice(Instance)    : median={df_out['t12_slice'].median():.0f}  std={df_out['t12_slice'].std():.1f}")
        if "pubis_slice" in df_out.columns and not df_out["pubis_slice"].isna().all():
            print(f"  pubis_slice(Instance)  : median={df_out['pubis_slice'].median():.0f}  std={df_out['pubis_slice'].std():.1f}")
    n_upper = len([f for f in os.listdir(UPPER_DIR) if f.endswith(".png")]) if os.path.isdir(UPPER_DIR) else 0
    n_bottom = len([f for f in os.listdir(BOTTOM_DIR) if f.endswith(".png")]) if os.path.isdir(BOTTOM_DIR) else 0
    print(f"  저장된 PNG: upper={n_upper}, bottom={n_bottom}")


def export_previews_from_bounds():
    """기존 z_bounds.xlsx의 Instance Number로 PNG만 재생성."""
    if not os.path.exists(OUT_PATH):
        raise FileNotFoundError(f"z_bounds 파일 없음: {OUT_PATH}")

    os.makedirs(UPPER_DIR, exist_ok=True)
    os.makedirs(BOTTOM_DIR, exist_ok=True)
    folder_map, _ = build_folder_maps(DICOM_BASE)
    df = pd.read_excel(OUT_PATH)

    def find_dcm_by_instance(dcm_dir: str, instance_num: int) -> str | None:
        for fname in os.listdir(dcm_dir):
            if fname.startswith("."):
                continue
            path = os.path.join(dcm_dir, fname)
            if not os.path.isfile(path):
                continue
            inst = get_instance_number(path)
            if inst == instance_num:
                return path
        return None

    saved = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PNG 재생성"):
        pid = str(int(row["PatientID"]))
        if pid not in folder_map:
            continue
        patient_folder = folder_map[pid]
        subfolders = [s for s in os.listdir(patient_folder)
                      if os.path.isdir(os.path.join(patient_folder, s))]
        if not subfolders:
            continue
        dcm_dir = os.path.join(patient_folder, subfolders[0])

        upper_dcm = bottom_dcm = None
        if pd.notna(row.get("t12_slice")):
            upper_dcm = find_dcm_by_instance(dcm_dir, int(row["t12_slice"]))
        if pd.notna(row.get("pubis_slice")):
            bottom_dcm = find_dcm_by_instance(dcm_dir, int(row["pubis_slice"]))

        u_ok, b_ok = save_boundary_previews(pid, upper_dcm, bottom_dcm)
        if u_ok or b_ok:
            saved += 1

    print(f"PNG 재생성 완료: {saved}명 (upper/bottom 폴더 확인)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--export-previews":
        export_previews_from_bounds()
    else:
        main()