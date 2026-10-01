"""
강남_z_bounds.xlsx 에 이미 계산된 pubis~liver 구간(k-index 범위) 내에서,
TotalSegmentator 척추 세그멘테이션으로 L3 슬라이스 1장을 자동으로 찾아
그 슬라이스 한 장의 체성분(IMATA/NAMA/LAMA/내장지방/피하지방) 면적을 계산한다.

[방법]
  z_bounds.xlsx의 series_description/manufacturer_model로 원래 세그멘테이션에 쓰인 것과
  동일한 시리즈를 다시 찾아 NIfTI로 변환 → liver_upper_k~pubis_k 범위로 볼륨을 크롭 →
  1) TotalSegmentator "total" task를 roi_subset=["vertebrae_L3"]로 실행해 L3 추골 마스크를 얻고,
     axial 단면적이 가장 큰 슬라이스를 L3 slice로 채택
  2) TotalSegmentator "tissue_4_types" task 실행
    (클래스: subcutaneous_fat=피하지방, torso_fat=내장지방, skeletal_muscle, intermuscular_fat=IMATA)
  → subcutaneous_fat/torso_fat/intermuscular_fat 마스크는 지방 HU(-190~-30)로 추가 필터링
  → skeletal_muscle 마스크 내부를 HU로 NAMA(+30~+150)/LAMA(-30~+30)로 재분류
  → L3 슬라이스 1장에서만 각 지표의 복셀 수 × 픽셀면적(cm²) = 슬라이스 면적(cm²)

[출력]
  {SITE}_body_composition_L3.xlsx : PatientID별 L3 슬라이스 면적 5개 컬럼 + seg_status
"""

import os
import io
import sys
import re
import shutil
import pickle
import logging
import tempfile
import contextlib
from difflib import SequenceMatcher
from typing import cast

import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import SimpleITK as sitk
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from totalsegmentator.python_api import totalsegmentator
import torch

import add_body_composition_to_metadata as merge_mod

torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

for _lg in ("nnunetv2", "totalsegmentator", "batchgeneratorsv2",
            "acvl_utils", "dynamic_network_architectures"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

# ── 설정 ──────────────────────────────────────────────────────────────────────

SITE = "신촌"
SITE_EN = {"강남": "Gangnam", "신촌": "Sinchon"}[SITE]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE_DIR     = rf"{PROJECT_ROOT}\data\{SITE_EN}"

DICOM_BASE   = rf"E:\영상제공\{SITE}\{SITE}_axial"
DATA_DIR     = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}"
ZBOUNDS_PATH = rf"{DATA_DIR}\aec\{SITE}_z_bounds.xlsx"
MERGED_PATH  = rf"{SITE_DIR}\{SITE_EN}.xlsx"
OUT_PATH     = rf"{SITE_DIR}\{SITE}_body_composition_L3.xlsx"

CHECKPOINT_PATH = rf"{PROJECT_ROOT}\aec\{SITE}_body_composition_L3_checkpoint.pkl"

TASK        = "tissue_4_types"
VERT_TASK   = "total"        # L3 위치 탐지용 (roi_subset=["vertebrae_L3"])
NAMA_HU     = (30, 150)     # 정상감쇠근육 (Normal Attenuation Muscle Area)
LAMA_HU     = (-30, 30)     # 저감쇠근육   (Low Attenuation Muscle Area)
FAT_HU      = (-190, -30)   # 지방 조직 (subcutaneous_fat/torso_fat/intermuscular_fat 공통 적용)

L3_COLS = ["IMATA_L3", "NAMA_L3", "LAMA_L3", "내장지방_L3", "피하지방_L3"]

BATCH_SIZE  = 1      # 몇 명 처리마다 체크포인트/중간 저장을 할지
MERGE_EVERY = 10     # 몇 명 처리마다 liver_merged_features.xlsx metadata 시트에 병합할지
TEST_N      = None   # 정수 지정 시 앞 N명만 처리 (파일럿 검증용)


@contextlib.contextmanager
def _silence():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ── DICOM 폴더/시리즈 매칭 (3_extract_z_bounds_liver.py 와 동일한 규칙) ─────────

def _norm_series(s: str) -> str:
    s = s.lower()
    s = " ".join(s.split())
    s = re.sub(r"[_/]", "_", s)
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    return s


def _get_folder_dicom_meta(folder: str) -> tuple[str | None, str | None]:
    try:
        for f in os.listdir(folder):
            fp = os.path.join(folder, f)
            if os.path.isfile(fp):
                try:
                    hdr = pydicom.dcmread(fp, stop_before_pixels=True)
                    series_desc  = str(hdr.SeriesDescription) if hasattr(hdr, "SeriesDescription") else None
                    manufacturer = str(hdr.ManufacturerModelName) if hasattr(hdr, "ManufacturerModelName") else None
                    return series_desc, manufacturer
                except Exception:
                    continue
    except Exception:
        pass
    return None, None


def build_folder_map(dicom_base: str) -> dict[str, str]:
    if not os.path.isdir(dicom_base):
        raise FileNotFoundError(f"DICOM 경로 없음: {dicom_base}")
    folder_map: dict[str, str] = {}
    for folder_name in os.listdir(dicom_base):
        full_path = os.path.join(dicom_base, folder_name)
        if not os.path.isdir(full_path):
            continue
        parts = folder_name.split("_")
        folder_map[folder_name if len(parts) == 1 else parts[1]] = full_path
    return folder_map


def find_series_dir(pid_str: str, folder_map: dict, expected_series: str | None,
                    expected_manufacturer: str | None, warnings: list) -> str | None:
    if pid_str not in folder_map:
        return None
    base = folder_map[pid_str]
    subfolders = [s for s in os.listdir(base) if os.path.isdir(os.path.join(base, s))]
    if not subfolders:
        return None
    if len(subfolders) == 1:
        return os.path.join(base, subfolders[0])

    matched     = None
    found_meta: list[tuple[str | None, str | None]] = []
    for sf in subfolders:
        sd, mf = _get_folder_dicom_meta(os.path.join(base, sf))
        found_meta.append((sd, mf))
        sd_ok = expected_series       is None or _norm_series(sd or "") == _norm_series(expected_series)
        mf_ok = expected_manufacturer is None or " ".join((mf or "").split()) == " ".join(str(expected_manufacturer).split())
        if sd_ok and mf_ok:
            matched = sf
            break
    if matched is None and expected_series is not None:
        norm_exp = _norm_series(expected_series)
        best_sf, best_ratio = None, 0.0
        for sf, (sd, _) in zip(subfolders, found_meta):
            ratio = max(
                SequenceMatcher(None, _norm_series(sd or ""), norm_exp).ratio(),
                SequenceMatcher(None, _norm_series(sf),       norm_exp).ratio(),
            )
            if ratio > best_ratio:
                best_ratio, best_sf = ratio, sf
        if best_sf:
            matched = best_sf
            warnings.append(f"series: 유사도 매칭 (ratio={best_ratio:.2f}) → '{best_sf}'")
    if matched is None:
        warnings.append(f"series: '{expected_series}' 매칭 실패 → 첫 번째 시리즈 사용 (읽힌 값: {found_meta})")
        matched = subfolders[0]
    return os.path.join(base, matched)


# ── 핵심 추출 (환자 1명) ──────────────────────────────────────────────────────

def _empty_result(pid: int) -> dict:
    result = {
        "PatientID": pid, "seg_status": "failed",
        "n_slices_range": np.nan, "l3_k_local": np.nan,
    }
    result.update({col: np.nan for col in L3_COLS})
    return result


def process_patient(zrow: pd.Series, folder_map: dict, tmp_dir: str) -> dict:
    pid     = int(zrow["PatientID"])
    pid_str = str(pid)
    result  = _empty_result(pid)
    warnings: list[str] = []

    dcm_dir = find_series_dir(pid_str, folder_map, zrow.get("series_description"),
                              zrow.get("manufacturer_model"), warnings)
    if dcm_dir is None:
        result["seg_status"] = "no_dicom"
        return result

    nii_path         = os.path.join(tmp_dir, f"{pid}.nii.gz")
    cropped_nii_path = os.path.join(tmp_dir, f"{pid}_cropped.nii.gz")
    seg_dir          = os.path.join(tmp_dir, f"{pid}_seg")
    vert_seg_dir     = os.path.join(tmp_dir, f"{pid}_vert_seg")

    try:
        reader     = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dcm_dir)
        if not series_ids:
            result["seg_status"] = "no_series_uid"
            return result
        dicom_names = reader.GetGDCMSeriesFileNames(dcm_dir, series_ids[0])
        reader.SetFileNames(dicom_names)
        img = reader.Execute()

        n_k = img.GetSize()[2]
        if len(dicom_names) != n_k:
            result["seg_status"] = "invalid_volume_match"
            return result

        lo_k = int(min(zrow["pubis_k"], zrow["liver_upper_k"]))
        hi_k = int(max(zrow["pubis_k"], zrow["liver_upper_k"]))
        lo_k = max(0, min(lo_k, n_k - 1))
        hi_k = max(0, min(hi_k, n_k - 1))
        if hi_k < lo_k:
            result["seg_status"] = "invalid_bounds"
            return result

        size  = [img.GetSize()[0], img.GetSize()[1], hi_k - lo_k + 1]
        index = [0, 0, lo_k]
        cropped_img = sitk.RegionOfInterest(img, size, index)
        sitk.WriteImage(cropped_img, cropped_nii_path)

        result["n_slices_range"] = hi_k - lo_k + 1

        os.makedirs(vert_seg_dir, exist_ok=True)
        with _silence():
            totalsegmentator(input=cropped_nii_path, output=vert_seg_dir, task=VERT_TASK,
                             roi_subset=["vertebrae_L3"], fast=False, quiet=True, device=DEVICE)

        l3_mask_path = os.path.join(vert_seg_dir, "vertebrae_L3.nii.gz")
        if not os.path.exists(l3_mask_path):
            result["seg_status"] = "l3_seg_missing"
            return result
        l3_mask = cast(Nifti1Image, nib.load(l3_mask_path)).get_fdata() > 0.5
        area_per_slice = l3_mask.sum(axis=(0, 1))
        if not np.any(area_per_slice > 0):
            result["seg_status"] = "l3_not_found"
            return result
        l3_idx = int(np.argmax(area_per_slice))
        result["l3_k_local"] = l3_idx

        os.makedirs(seg_dir, exist_ok=True)
        with _silence():
            totalsegmentator(input=cropped_nii_path, output=seg_dir, task=TASK,
                             fast=False, quiet=True, device=DEVICE)

        class_files = {
            "subcutaneous_fat":  "subcutaneous_fat.nii.gz",
            "torso_fat":         "torso_fat.nii.gz",
            "skeletal_muscle":   "skeletal_muscle.nii.gz",
            "intermuscular_fat": "intermuscular_fat.nii.gz",
        }
        masks = {}
        for cls, fname in class_files.items():
            p = os.path.join(seg_dir, fname)
            if not os.path.exists(p):
                warnings.append(f"{cls}: 세그멘테이션 결과 없음")
                result["seg_status"] = "tissue_seg_missing"
                return result
            masks[cls] = cast(Nifti1Image, nib.load(p)).get_fdata() > 0.5

        cropped_nii = cast(Nifti1Image, nib.load(cropped_nii_path))
        hu_data     = cropped_nii.get_fdata()
        zooms = cast(tuple[float, ...], cropped_nii.header.get_zooms())
        sx, sy = zooms[0], zooms[1]
        pixel_area_cm2 = float(sx) * float(sy) / 100.0

        hu_l3 = hu_data[:, :, l3_idx]

        fat_hu_mask = (hu_l3 >= FAT_HU[0]) & (hu_l3 <= FAT_HU[1])
        imata_mask  = masks["intermuscular_fat"][:, :, l3_idx] & fat_hu_mask
        vat_mask    = masks["torso_fat"][:, :, l3_idx]         & fat_hu_mask
        sat_mask    = masks["subcutaneous_fat"][:, :, l3_idx]  & fat_hu_mask

        muscle_mask = masks["skeletal_muscle"][:, :, l3_idx]
        nama_mask   = muscle_mask & (hu_l3 >= NAMA_HU[0]) & (hu_l3 <= NAMA_HU[1])
        lama_mask   = muscle_mask & (hu_l3 >= LAMA_HU[0]) & (hu_l3 <= LAMA_HU[1])

        result["IMATA_L3"]    = round(float(imata_mask.sum())  * pixel_area_cm2, 2)
        result["내장지방_L3"] = round(float(vat_mask.sum())    * pixel_area_cm2, 2)
        result["피하지방_L3"] = round(float(sat_mask.sum())    * pixel_area_cm2, 2)
        result["NAMA_L3"]     = round(float(nama_mask.sum())   * pixel_area_cm2, 2)
        result["LAMA_L3"]     = round(float(lama_mask.sum())   * pixel_area_cm2, 2)
        result["seg_status"]  = "ok"

    except Exception as e:
        result["seg_status"] = f"error:{type(e).__name__}:{e}"

    finally:
        for path in (nii_path, cropped_nii_path, seg_dir, vert_seg_dir):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    if warnings:
        tqdm.write(f"  PID {pid}: " + " | ".join(warnings))

    return result


# ── I/O / 체크포인트 ──────────────────────────────────────────────────────────

def _write_output(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["PatientID"], keep="last")
    df = df.sort_values("PatientID").reset_index(drop=True)
    col_order = ["PatientID", "seg_status", "n_slices_range", "l3_k_local"] + L3_COLS
    cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df[cols].to_excel(OUT_PATH, sheet_name="body_composition", index=False)


def _load_checkpoint() -> tuple[set[int], list[dict]]:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            data = pickle.load(f)
        processed, results = data["processed"], data["results"]

        error_pids = {r["PatientID"] for r in results if str(r.get("seg_status", "")) != "ok"}
        if error_pids:
            processed = processed - error_pids
            results   = [r for r in results if r["PatientID"] not in error_pids]
            print(f"[체크포인트] error/미완료 상태 {len(error_pids)}명 재시도 대상으로 포함")

        print(f"[체크포인트] {len(processed)}명 완료 → 이어서 시작")
        return processed, results
    return set(), []


def _save_checkpoint(processed: set[int], results: list[dict]) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({"processed": processed, "results": results}, f)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    merged_pids = set(pd.read_excel(MERGED_PATH, sheet_name="metadata")["PatientID"].astype(int))

    zdf = pd.read_excel(ZBOUNDS_PATH)
    zdf = zdf[zdf["seg_status"].astype(str) == "ok"]
    zdf = zdf[zdf["PatientID"].astype(int).isin(merged_pids)].reset_index(drop=True)
    if TEST_N is not None:
        zdf = zdf.head(TEST_N)
    print(f"대상: {len(zdf)}명 ({os.path.basename(MERGED_PATH)} metadata PatientID 기준, {len(merged_pids)}명 중 z_bounds ok 매칭)")

    folder_map = build_folder_map(DICOM_BASE)

    processed, results = _load_checkpoint()
    pending = zdf[~zdf["PatientID"].isin(processed)]
    print(f"완료: {len(processed)}명  |  처리 예정: {len(pending)}명")

    n_processed = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tqdm(total=len(zdf), initial=len(processed), desc="체성분 구간 합산") as pbar:
            for _, zrow in pending.iterrows():
                pid = int(zrow["PatientID"])
                try:
                    result = process_patient(zrow, folder_map, tmp_dir)
                except Exception as e:
                    tqdm.write(f"  [ERROR] PID {pid}: {type(e).__name__}: {e}")
                    result = _empty_result(pid)
                    result["seg_status"] = f"error:{type(e).__name__}:{e}"

                results.append(result)
                processed.add(pid)
                n_processed += 1
                pbar.update(1)
                tqdm.write(f"  PID {pid}: {result['seg_status']}")

                if n_processed % BATCH_SIZE == 0 or n_processed == len(pending):
                    _save_checkpoint(processed, results)
                    _write_output(results)

                if len(processed) % MERGE_EVERY == 0 or n_processed == len(pending):
                    try:
                        merge_mod.merge(SITE, backup=True, suffix="_L3", new_cols=L3_COLS)
                    except Exception as e:
                        tqdm.write(f"  [merge 경고] {type(e).__name__}: {e}")

    _write_output(results)
    counts = pd.Series([r["seg_status"] for r in results]).value_counts().to_dict()
    print(f"\n[완료] 총 {len(results)}명  |  {counts}  |  {OUT_PATH}")


if __name__ == "__main__":
    main()
