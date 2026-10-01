"""
aec/z_bounds.xlsx 에 seg_status='ok'로 기록된 각 시리즈(환자당 여러 시리즈 가능:
Pre Contrast/With Contrast/Arterial/Portal/Delay 등)에 대해, aec_cropped와 동일한
pubis_k~liver_k 구간을 D:\\데이터서비스팀 요청(이홍선) 원본 DICOM에서 다시 크롭하여
volumetric VAT(내장지방)/SAT(피하지방)를 TotalSegmentator로 측정한다.

[방법]
  z_bounds.xlsx의 (PatientID, series_desc, manufacturer, n_slices)로 원래 시리즈를
  total_crop_aec.find_axial_series() 결과 중에서 재매칭 → z 오름차순 정렬 후
  pubis_k~liver_k 구간으로 볼륨을 크롭 → TotalSegmentator "tissue_4_types" 실행
    (torso_fat=내장지방(VAT), subcutaneous_fat=피하지방(SAT))
  → 지방 HU(-190~-30)로 추가 필터링 → 복셀 수 × 복셀부피(cm³) = 구간 전체 volumetric 합

  body_composition_sum.py와 같은 HU 기준을 쓰지만, 그쪽은 슬라이스 면적 합(cm²)이고
  여기는 3차원 볼륨 합(cm³)이라는 점이 다르다.

[출력]
  aec/vat_sat_series.xlsx : (PatientID, series_desc, manufacturer)별 vat_sum/sat_sum(cm³) + seg_status
  → metadata_cleaned/metadata_raw 시트 병합은 merge_vat_sat_to_metadata.py에서 별도 수행 (환자당 여러
    시리즈의 평균을 낸다).
"""

import os
import io
import sys
import shutil
import pickle
import logging
import tempfile
import contextlib
from typing import cast

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import SimpleITK as sitk
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from totalsegmentator.python_api import totalsegmentator
import torch

import total_crop_aec as tca

torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

for _lg in ("nnunetv2", "totalsegmentator", "batchgeneratorsv2",
            "acvl_utils", "dynamic_network_architectures"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

# ── 설정 ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AEC_DIR      = os.path.join(PROJECT_ROOT, "aec")

DICOM_BASE   = tca.DICOM_BASE
ZBOUNDS_PATH = os.path.join(AEC_DIR, "z_bounds.xlsx")
OUT_PATH     = os.path.join(AEC_DIR, "vat_sat_series.xlsx")

CHECKPOINT_PATH = os.path.join(AEC_DIR, "vat_sat_checkpoint.pkl")

TASK       = "tissue_4_types"
FAT_HU     = (-190, -30)   # 지방 조직 (subcutaneous_fat/torso_fat 공통 적용)

BATCH_SIZE = 5      # 몇 시리즈 처리마다 체크포인트/중간 저장을 할지
TEST_N     = None   # 정수 지정 시 앞 N개 시리즈만 처리 (파일럿 검증용)


@contextlib.contextmanager
def _silence():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ── 대상 시리즈 목록 구성 ────────────────────────────────────────────────────

def load_targets() -> pd.DataFrame:
    zdf = pd.read_excel(ZBOUNDS_PATH)
    zdf = zdf[zdf["seg_status"].astype(str) == "ok"].reset_index(drop=True)
    zdf["PatientID"] = zdf["PatientID"].astype(int)
    if TEST_N is not None:
        zdf = zdf.head(TEST_N)
    return zdf


def build_folder_map(dicom_base: str) -> dict[str, str]:
    if not os.path.isdir(dicom_base):
        raise FileNotFoundError(f"DICOM 경로 없음: {dicom_base}")
    folder_map: dict[str, str] = {}
    for folder_name in os.listdir(dicom_base):
        full_path = os.path.join(dicom_base, folder_name)
        if not os.path.isdir(full_path):
            continue
        pid = tca._patient_id_from_folder(folder_name)
        folder_map[pid] = full_path
    return folder_map


def match_series(rows_for_patient: pd.DataFrame, found_series: list[dict]) -> dict[int, dict]:
    """z_bounds row index -> found_series 항목. (series_desc, manufacturer, n_slices)로 매칭하고
    한 번 매칭된 series는 다시 쓰지 않는다 (중복 desc/manufacturer/n_slices 대비)."""
    used = set()
    matched: dict[int, dict] = {}
    for idx, zrow in rows_for_patient.iterrows():
        for j, s in enumerate(found_series):
            if j in used:
                continue
            if s["series_desc"] != str(zrow["series_desc"]):
                continue
            if s["manufacturer"] != str(zrow["manufacturer"]):
                continue
            rows = tca.read_series_slices(s["files"])
            if rows is None or len(rows) != int(zrow["n_slices"]):
                continue
            matched[idx] = {**s, "_sorted_rows": rows}
            used.add(j)
            break
    return matched


# ── 핵심 추출 (시리즈 1개) ────────────────────────────────────────────────────

def _empty_result(zrow: pd.Series) -> dict:
    return {
        "PatientID": int(zrow["PatientID"]), "series_desc": zrow["series_desc"],
        "manufacturer": zrow["manufacturer"], "seg_status": "failed",
        "vat_sum": np.nan, "sat_sum": np.nan,
    }


def process_series(zrow: pd.Series, sorted_rows: list[dict], tmp_dir: str) -> dict:
    pid    = int(zrow["PatientID"])
    result = _empty_result(zrow)

    cropped_nii_path = os.path.join(tmp_dir, f"{pid}_{abs(hash((zrow['series_desc'], zrow['manufacturer'])))}.nii.gz")
    seg_dir           = cropped_nii_path[:-7] + "_seg"

    try:
        n_k = len(sorted_rows)
        pubis_k = int(zrow["pubis_k"])
        liver_k = int(zrow["liver_k"])
        lo_k = max(0, min(pubis_k, liver_k))
        hi_k = min(n_k - 1, max(pubis_k, liver_k))
        if hi_k < lo_k:
            result["seg_status"] = "invalid_bounds"
            return result

        files_in_range = [r["file"] for r in sorted_rows[lo_k: hi_k + 1]]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files_in_range)
        cropped_img = reader.Execute()
        sitk.WriteImage(cropped_img, cropped_nii_path)

        os.makedirs(seg_dir, exist_ok=True)
        with _silence():
            totalsegmentator(input=cropped_nii_path, output=seg_dir, task=TASK,
                             fast=False, quiet=True, device=DEVICE)

        class_files = {"subcutaneous_fat": "subcutaneous_fat.nii.gz", "torso_fat": "torso_fat.nii.gz"}
        masks = {}
        for cls, fname in class_files.items():
            p = os.path.join(seg_dir, fname)
            if not os.path.exists(p):
                result["seg_status"] = "tissue_seg_missing"
                return result
            masks[cls] = cast(Nifti1Image, nib.load(p)).get_fdata() > 0.5

        cropped_nii = cast(Nifti1Image, nib.load(cropped_nii_path))
        hu_data     = cropped_nii.get_fdata()
        zooms = cast(tuple[float, ...], cropped_nii.header.get_zooms())
        voxel_vol_cm3 = float(zooms[0]) * float(zooms[1]) * float(zooms[2]) / 1000.0

        fat_hu_mask = (hu_data >= FAT_HU[0]) & (hu_data <= FAT_HU[1])
        vat_mask = masks["torso_fat"]        & fat_hu_mask
        sat_mask = masks["subcutaneous_fat"] & fat_hu_mask

        result["vat_sum"]     = round(float(vat_mask.sum()) * voxel_vol_cm3, 2)
        result["sat_sum"]     = round(float(sat_mask.sum()) * voxel_vol_cm3, 2)
        result["seg_status"]  = "ok"

    except Exception as e:
        result["seg_status"] = f"error:{type(e).__name__}:{e}"

    finally:
        for path in (cropped_nii_path, seg_dir):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    return result


# ── I/O / 체크포인트 ──────────────────────────────────────────────────────────

def _write_output(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(["PatientID", "series_desc"]).reset_index(drop=True)
    os.makedirs(AEC_DIR, exist_ok=True)
    df.to_excel(OUT_PATH, sheet_name="vat_sat_series", index=False)


def _load_checkpoint() -> tuple[set[tuple], list[dict]]:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            data = pickle.load(f)
        processed, results = data["processed"], data["results"]

        errors = [r for r in results if str(r.get("seg_status", "")).startswith("error")]
        if errors:
            err_keys = {(r["PatientID"], r["series_desc"], r["manufacturer"]) for r in errors}
            processed = processed - err_keys
            results   = [r for r in results
                        if (r["PatientID"], r["series_desc"], r["manufacturer"]) not in err_keys]
            print(f"[체크포인트] error 상태 {len(err_keys)}건 재시도 대상으로 포함")

        print(f"[체크포인트] {len(processed)}건 완료 → 이어서 시작")
        return processed, results
    return set(), []


def _save_checkpoint(processed: set[tuple], results: list[dict]) -> None:
    os.makedirs(AEC_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({"processed": processed, "results": results}, f)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    zdf = load_targets()
    print(f"대상: {len(zdf)}개 시리즈 ({zdf['PatientID'].nunique()}명, {ZBOUNDS_PATH})")

    folder_map = build_folder_map(DICOM_BASE)

    processed, results = _load_checkpoint()

    n_processed = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tqdm(total=len(zdf), initial=len(processed), desc="VAT/SAT volumetric") as pbar:
            for pid, group in zdf.groupby("PatientID"):
                pid = int(pid)
                keys = [(pid, r["series_desc"], r["manufacturer"]) for _, r in group.iterrows()]
                if all(k in processed for k in keys):
                    continue

                pid_str = str(pid)
                if pid_str not in folder_map:
                    for _, zrow in group.iterrows():
                        key = (pid, zrow["series_desc"], zrow["manufacturer"])
                        if key in processed:
                            continue
                        result = _empty_result(zrow)
                        result["seg_status"] = "no_dicom_folder"
                        results.append(result)
                        processed.add(key)
                        n_processed += 1
                        pbar.update(1)
                    continue

                try:
                    found_series = tca.find_axial_series(folder_map[pid_str])
                except Exception as e:
                    found_series = []
                    tqdm.write(f"  [ERROR] PID {pid}: find_axial_series 실패 {type(e).__name__}: {e}")

                matched = match_series(group, found_series)

                for idx, zrow in group.iterrows():
                    key = (pid, zrow["series_desc"], zrow["manufacturer"])
                    if key in processed:
                        continue

                    if idx not in matched:
                        result = _empty_result(zrow)
                        result["seg_status"] = "series_match_failed"
                    else:
                        m = matched[idx]
                        try:
                            result = process_series(zrow, m["_sorted_rows"], tmp_dir)
                        except Exception as e:
                            tqdm.write(f"  [ERROR] PID {pid}: {type(e).__name__}: {e}")
                            result = _empty_result(zrow)
                            result["seg_status"] = f"error:{type(e).__name__}:{e}"

                    results.append(result)
                    processed.add(key)
                    n_processed += 1
                    pbar.update(1)
                    tqdm.write(f"  PID {pid} [{zrow['series_desc']}]: {result['seg_status']}")

                    if n_processed % BATCH_SIZE == 0:
                        _save_checkpoint(processed, results)
                        _write_output(results)

    _save_checkpoint(processed, results)
    _write_output(results)
    counts = pd.Series([r["seg_status"] for r in results]).value_counts().to_dict()
    print(f"\n[완료] 총 {len(results)}건  |  {counts}  |  {OUT_PATH}")


if __name__ == "__main__":
    main()
