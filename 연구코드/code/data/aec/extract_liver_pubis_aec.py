"""
E:/영상제공/{site}/{site}_axial 아래 DICOM에서 사이트별로
  1) 전체 슬라이스 AEC(XRayTubeCurrent) 배열      → {site}_aec_total.xlsx
  2) TotalSegmentator(liver, hip_left, hip_right)로 구한 liver~pubis 경계 → {site}_z_bounds.xlsx
  3) 위 경계로 크롭한 AEC(+128포인트 보간)         → {site}_aec_cropped.xlsx
  4) 체성분(IMATA/NAMA/LAMA/VAT/SAT) 합계, 단일 body_composition 시트 + kVp 포함
                                                   → {site}_body_composition.xlsx
  5) 체성분 슬라이스별 면적(SFA/VFA/TAMA/LAMA/NAMA/IMATA, +128포인트 보간),
     조직×raw/interp 조합별 시트(SFA_raw, SFA_interp, ... 총 12개) → {site}_body_comp_cropped.xlsx
를 추출한다. 대상 환자는 {site}_DLO_Results.xlsx의 PatientID 전체이며, 시리즈 폴더는
그 파일의 Series_Desc(=DICOM SeriesDescription 태그 원본값)를 힌트로 찾는다.

기존 xlsx에 이미 있는 PatientID는 절대 건드리지 않고 그대로 보존하며, 빠진 PatientID만
추가로 처리해 append한다 (재실행해도 안전 — 이미 끝난 환자는 다시 계산하지 않음).

[AEC 배열 방향]  aec_total, aec_cropped 모두 aec_1 = 간 쪽(가장 낮은 k, 정렬상 InstanceNumber가
  작은 쪽), aec_n = 두덩뼈 쪽으로 방향을 통일한다.
[InstanceNumber 주의] 일부 시리즈는 InstanceNumber가 1로 시작하지 않는다. 실제 값(inst)이
  아니라 정렬 후의 위치(index)로만 aec 배열/경계를 다루므로 시작값과 무관하게 안전하다.
[제외 기준] AEC 값이 모두 NaN / CV < 0.05 / range < 10mA / R² ≥ 0.95(거의 직선)
  인 환자는 aec_total에 포함하지 않는다 (기존 데이터와 동일 기준).
"""

import contextlib
import difflib
import io
import logging
import os
import pickle
import re
import shutil
import sys
import tempfile
from typing import cast

import numpy as np
import openpyxl
import pandas as pd
import pydicom
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from totalsegmentator.python_api import totalsegmentator
import torch

torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

for _lg in ("nnunetv2", "totalsegmentator", "batchgeneratorsv2",
            "acvl_utils", "dynamic_network_architectures"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

# ── 설정 ──────────────────────────────────────────────────────────────────────

SITES   = ["신촌", "강남"]
DATA_DIR = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data"

ROI_SUBSET       = ["liver", "hip_left", "hip_right"]
MIN_LIVER_VOXELS = 3000
N_INTERP         = 128
BATCH_SIZE       = 5

AEC_CV_MIN   = 0.05
RANGE_MIN    = 10
R2_THRESHOLD = 0.95

# 체성분(VAT/SAT 등) 계산 — pubis~liver 구간 볼륨에 tissue_4_types 실행
TISSUE_TASK = "tissue_4_types"
NAMA_HU     = (30, 150)     # 정상감쇠근육 (Normal Attenuation Muscle Area)
LAMA_HU     = (-30, 30)     # 저감쇠근육   (Low Attenuation Muscle Area)
FAT_HU      = (-190, -30)   # 지방 조직 (subcutaneous_fat/torso_fat/intermuscular_fat 공통 적용)

# 슬라이스별 체성분 곡선(aec_cropped와 동일 형식) 칼럼 접두사 — SFA=SAT, VFA=VAT, TAMA=NAMA+LAMA
TISSUE_KEYS = ["SFA", "VFA", "TAMA", "LAMA", "NAMA", "IMATA"]


def site_paths(site: str, shard: int | None = None) -> dict:
    suffix = f".shard{shard}" if shard is not None else ""
    return {
        "dicom_base":  rf"E:\영상제공\{site}\{site}_axial",
        "dlo_path":    rf"{DATA_DIR}\{site}\metadata\{site}_DLO_Results.xlsx",
        "aec_total":   rf"{DATA_DIR}\{site}\aec\{site}_aec_total{suffix}.xlsx",
        "z_bounds":    rf"{DATA_DIR}\{site}\aec\{site}_z_bounds{suffix}.xlsx",
        "aec_cropped": rf"{DATA_DIR}\{site}\aec\{site}_aec_cropped{suffix}.xlsx",
        "body_comp":   rf"{DATA_DIR}\{site}\aec\{site}_body_composition{suffix}.xlsx",
        "body_comp_cropped": rf"{DATA_DIR}\{site}\aec\{site}_body_comp_cropped{suffix}.xlsx",
        "checkpoint":  rf"{DATA_DIR}\{site}\aec\{site}_incremental_checkpoint{suffix}.pkl",
        # 병렬 실행(shard)일 때도 최종 결과 존재 여부는 항상 메인 파일 기준으로 확인한다.
        "main_z_bounds":  rf"{DATA_DIR}\{site}\aec\{site}_z_bounds.xlsx",
        "main_body_comp": rf"{DATA_DIR}\{site}\aec\{site}_body_composition.xlsx",
    }


@contextlib.contextmanager
def _silence():
    old_out, old_err = sys.stdout, sys.stderr
    buf = sys.stdout = sys.stderr = io.StringIO()
    try:
        yield
    except SystemExit:
        # 라이브러리가 sys.exit()로 끝낼 때(예: 라이선스 오류) 숨긴 출력을 보여줘야 원인을 알 수 있다.
        sys.stdout, sys.stderr = old_out, old_err
        print(buf.getvalue())
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ── 시리즈 폴더 매칭 (DLO_Results.xlsx의 ground-truth Series_Desc를 힌트로 사용) ──

def _norm_series_name(s) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\.\.\.$", "", s)
    s = re.sub(r"[^0-9a-z가-힣]", "", s)
    return s


def find_series_folder(patient_dir: str, seed_series_desc) -> str | None:
    subs = [s for s in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, s))]
    if not subs:
        return None
    if pd.isna(seed_series_desc):
        return subs[0] if len(subs) == 1 else None

    target = _norm_series_name(seed_series_desc)
    for s in subs:
        if _norm_series_name(s) == target:
            return s
    for s in subs:
        ns = _norm_series_name(s)
        if ns.startswith(target) or target.startswith(ns):
            return s
    if len(subs) == 1:
        return subs[0]

    norm_subs = {_norm_series_name(s): s for s in subs}
    best = difflib.get_close_matches(target, list(norm_subs.keys()), n=1, cutoff=0.75)
    return norm_subs[best[0]] if best else None


# ── DICOM 슬라이스 읽기 ──────────────────────────────────────────────────────

def read_slices(series_dir: str) -> list[dict] | None:
    """InstanceNumber 오름차순 정렬된 슬라이스 정보(간→두덩뼈 방향)."""
    rows = []
    for fname in os.listdir(series_dir):
        fp = os.path.join(series_dir, fname)
        if not os.path.isfile(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
        except Exception:
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        aec = getattr(ds, "XRayTubeCurrent", None) or getattr(ds, "TubeCurrent", None)
        rows.append({
            "file": fp,
            "inst": int(getattr(ds, "InstanceNumber", 0)),
            "z":    float(ipp[2]) if ipp is not None else float("nan"),
            "aec":  float(aec) if aec is not None else float("nan"),
        })
    if not rows:
        return None
    rows.sort(key=lambda r: r["inst"])
    return rows


def _aec_values(rows: list[dict]) -> list[float]:
    return [r["aec"] for r in rows if not np.isnan(r["aec"])]


def _is_excluded_signal(vals: list[float]) -> bool:
    """AEC 신호가 모두 NaN / CV 낮음 / 거의 직선이면 제외 대상."""
    if not vals:
        return True
    mean = float(np.mean(vals))
    cv = float(np.std(vals) / mean) if mean != 0 else 0.0
    if cv < AEC_CV_MIN:
        return True
    if max(vals) - min(vals) < RANGE_MIN:
        return True
    arr = np.array(vals)
    ss_tot = float(np.sum((arr - arr.mean()) ** 2))
    if ss_tot < 1e-10:
        return True
    r = float(np.corrcoef(np.arange(len(arr), dtype=float), arr)[0, 1])
    return r ** 2 >= R2_THRESHOLD


# ── Segmentation ──────────────────────────────────────────────────────────────

def _load_liver_mask(seg_dir: str) -> np.ndarray | None:
    p = os.path.join(seg_dir, "liver.nii.gz")
    if not os.path.exists(p):
        return None
    try:
        vol = cast(Nifti1Image, nib.load(p)).get_fdata()
        return vol if int((vol > 0.5).sum()) >= MIN_LIVER_VOXELS else None
    except Exception:
        return None


def _empty_bodycomp(pid: int) -> dict:
    return {
        "PatientID": pid, "kVp": np.nan, "seg_status": "failed", "n_slices_range": np.nan,
        "IMATA_sum_cm2": np.nan, "NAMA_sum_cm2": np.nan, "LAMA_sum_cm2": np.nan,
        "VAT_sum_cm2": np.nan, "SAT_sum_cm2": np.nan,
        "SFA_slices": None, "VFA_slices": None, "TAMA_slices": None,
        "LAMA_slices": None, "NAMA_slices": None, "IMATA_slices": None,
    }


def compute_body_composition(img: sitk.Image, lo_k: int, hi_k: int, tmp_dir: str, pid: int,
                              reverse: bool = False, keep_arrays: bool = False) -> dict:
    """pubis~liver 구간(k-index)만 크롭해 tissue_4_types로 VAT/SAT 등 체적(면적×슬라이스) 합과,
    슬라이스별 면적(SFA/VFA/TAMA/LAMA/NAMA/IMATA, cm2) 배열을 계산한다.
    reverse=True면 슬라이스 배열을 뒤집어 aec_cropped와 동일하게 간→두덩뼈 방향으로 맞춘다
    (crop은 항상 k 오름차순으로 이뤄지는데, apex_k(간)가 bottom_k(두덩뼈)보다 크면 오름차순이
    두덩뼈→간 방향이 되므로 뒤집어야 한다).
    keep_arrays=True면 QC 시각화용으로 hu_data/각 조직 mask를 result에 그대로 담는다
    (단일 슬라이스 호출 용도 — lo_k==hi_k가 아니면 3D 배열째로 담긴다)."""
    result = _empty_bodycomp(pid)
    cropped_nii_path = os.path.join(tmp_dir, f"{pid}_bc.nii.gz")
    seg_dir = os.path.join(tmp_dir, f"{pid}_bc_seg")
    try:
        size  = [img.GetSize()[0], img.GetSize()[1], hi_k - lo_k + 1]
        index = [0, 0, lo_k]
        cropped_img = sitk.RegionOfInterest(img, size, index)
        sitk.WriteImage(cropped_img, cropped_nii_path)
        result["n_slices_range"] = hi_k - lo_k + 1

        os.makedirs(seg_dir, exist_ok=True)
        with _silence():
            totalsegmentator(input=cropped_nii_path, output=seg_dir, task=TISSUE_TASK,
                             fast=False, quiet=True, device=DEVICE)

        class_files = {
            "subcutaneous_fat": "subcutaneous_fat.nii.gz", "torso_fat": "torso_fat.nii.gz",
            "skeletal_muscle": "skeletal_muscle.nii.gz", "intermuscular_fat": "intermuscular_fat.nii.gz",
        }
        masks = {}
        for cls, fname in class_files.items():
            p = os.path.join(seg_dir, fname)
            if not os.path.exists(p):
                result["seg_status"] = "tissue_seg_missing"
                return result
            masks[cls] = cast(Nifti1Image, nib.load(p)).get_fdata() > 0.5

        cropped_nii    = cast(Nifti1Image, nib.load(cropped_nii_path))
        hu_data        = cropped_nii.get_fdata()
        zooms          = cast(tuple[float, ...], cropped_nii.header.get_zooms())
        pixel_area_cm2 = float(zooms[0]) * float(zooms[1]) / 100.0

        fat_hu_mask = (hu_data >= FAT_HU[0]) & (hu_data <= FAT_HU[1])
        imata_mask  = masks["intermuscular_fat"] & fat_hu_mask
        vat_mask    = masks["torso_fat"]         & fat_hu_mask
        sat_mask    = masks["subcutaneous_fat"]  & fat_hu_mask

        muscle_mask = masks["skeletal_muscle"]
        nama_mask   = muscle_mask & (hu_data >= NAMA_HU[0]) & (hu_data <= NAMA_HU[1])
        lama_mask   = muscle_mask & (hu_data >= LAMA_HU[0]) & (hu_data <= LAMA_HU[1])

        result["IMATA_sum_cm2"] = round(float(imata_mask.sum())  * pixel_area_cm2, 2)
        result["VAT_sum_cm2"]   = round(float(vat_mask.sum())    * pixel_area_cm2, 2)
        result["SAT_sum_cm2"]   = round(float(sat_mask.sum())    * pixel_area_cm2, 2)
        result["NAMA_sum_cm2"]  = round(float(nama_mask.sum())   * pixel_area_cm2, 2)
        result["LAMA_sum_cm2"]  = round(float(lama_mask.sum())   * pixel_area_cm2, 2)

        # 슬라이스별 면적(cm2) — hu_data 축 순서는 (X, Y, Z)이므로 axis=(0,1) 합이 슬라이스당 픽셀 수
        sat_slices   = sat_mask.sum(axis=(0, 1))   * pixel_area_cm2
        vat_slices   = vat_mask.sum(axis=(0, 1))   * pixel_area_cm2
        imata_slices = imata_mask.sum(axis=(0, 1)) * pixel_area_cm2
        nama_slices  = nama_mask.sum(axis=(0, 1))  * pixel_area_cm2
        lama_slices  = lama_mask.sum(axis=(0, 1))  * pixel_area_cm2
        tama_slices  = nama_slices + lama_slices
        if reverse:
            sat_slices, vat_slices, imata_slices = sat_slices[::-1], vat_slices[::-1], imata_slices[::-1]
            nama_slices, lama_slices, tama_slices = nama_slices[::-1], lama_slices[::-1], tama_slices[::-1]

        result["SFA_slices"]   = sat_slices.tolist()
        result["VFA_slices"]   = vat_slices.tolist()
        result["TAMA_slices"]  = tama_slices.tolist()
        result["LAMA_slices"]  = lama_slices.tolist()
        result["NAMA_slices"]  = nama_slices.tolist()
        result["IMATA_slices"] = imata_slices.tolist()
        result["seg_status"]    = "ok"

        if keep_arrays:
            result["hu_slice"]    = hu_data
            result["sat_mask"]    = sat_mask
            result["vat_mask"]    = vat_mask
            result["nama_mask"]   = nama_mask
            result["lama_mask"]   = lama_mask
            result["imata_mask"]  = imata_mask
    except Exception as e:
        result["seg_status"] = f"error:{type(e).__name__}:{e}"
    finally:
        if os.path.isfile(cropped_nii_path):
            os.remove(cropped_nii_path)
        if os.path.isdir(seg_dir):
            shutil.rmtree(seg_dir, ignore_errors=True)
    return result


def _empty_zbounds(pid: int, series_desc, manufacturer, status: str) -> dict:
    return {
        "PatientID": pid, "series_description": series_desc, "manufacturer_model": manufacturer,
        "n_slices": np.nan, "inst_min": np.nan, "slices": np.nan,
        "liver_upper_slice": np.nan, "pubis_slice": np.nan,
        "liver_upper_k": np.nan, "pubis_k": np.nan,
        "z_range": np.nan, "z_liver_upper": np.nan, "z_pubis": np.nan,
        "seg_status": status,
    }


def process_patient(pid: int, series_dir: str, tmp_dir: str) -> tuple[dict | None, dict, list[float] | None, dict | None]:
    """반환: (aec_total 행 or None(제외), z_bounds 행, aec_cropped 값(pubis→liver) or None, body_composition 행 or None)"""
    try:
        hdr = pydicom.dcmread(next(os.path.join(series_dir, f) for f in os.listdir(series_dir)
                                    if os.path.isfile(os.path.join(series_dir, f))),
                               stop_before_pixels=True)
        series_desc  = re.sub(r"\s+", " ", str(getattr(hdr, "SeriesDescription", "")).replace("/", "_")).strip()
        manufacturer = str(getattr(hdr, "ManufacturerModelName", ""))
    except StopIteration:
        return None, _empty_zbounds(pid, None, None, "no_series"), None, None
    except Exception:
        series_desc = manufacturer = None

    by_inst = read_slices(series_dir)
    if by_inst is None:
        return None, _empty_zbounds(pid, series_desc, manufacturer, "no_position_data"), None, None

    n = len(by_inst)
    aec_full = [r["aec"] for r in by_inst]  # aec_1=간 방향, aec_n=두덩뼈 방향
    aec_row = {"PatientID": pid, "n_slices": n, "series_desc": series_desc, "manufacturer": manufacturer,
               "aec_full": aec_full}
    if _is_excluded_signal(_aec_values(by_inst)):
        aec_row = None  # AEC 신호 기준 미달 → aec_total 미포함

    z_row = _empty_zbounds(pid, series_desc, manufacturer, "failed")
    z_row["n_slices"] = n
    z_row["inst_min"] = int(min(r["inst"] for r in by_inst))
    cropped_liver_to_pubis: list[float] | None = None
    bc_row: dict | None = None

    nii_path = os.path.join(tmp_dir, f"{pid}.nii.gz")
    seg_dir  = os.path.join(tmp_dir, f"{pid}_seg")
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(series_dir)
        if not series_ids:
            z_row["seg_status"] = "nifti_fail"
            return aec_row, z_row, None, None
        k_files = list(reader.GetGDCMSeriesFileNames(series_dir, series_ids[0]))
        if len(k_files) != n:
            z_row["seg_status"] = "invalid_volume_match"
            return aec_row, z_row, None, None

        reader.SetFileNames(k_files)
        img = reader.Execute()
        sitk.WriteImage(img, nii_path)

        os.makedirs(seg_dir, exist_ok=True)
        with _silence():
            totalsegmentator(input=nii_path, output=seg_dir, task="total",
                             roi_subset=ROI_SUBSET, fast=False, quiet=True, device=DEVICE)

        n_k = int(cast(Nifti1Image, nib.load(nii_path)).shape[2])
        if n_k != n:
            z_row["seg_status"] = "invalid_volume_match"
            return aec_row, z_row, None, None

        hip_k: list[int] = []
        for side in ("hip_left", "hip_right"):
            p = os.path.join(seg_dir, f"{side}.nii.gz")
            if os.path.exists(p):
                idx = np.argwhere(cast(Nifti1Image, nib.load(p)).get_fdata() > 0.5)
                if len(idx) > 0:
                    hip_k.extend(idx[:, 2].astype(int).tolist())
        hip_found = len(hip_k) > 0

        liver_vol   = _load_liver_mask(seg_dir)
        liver_found = False
        liver_k     = np.array([], dtype=int)
        if liver_vol is not None:
            binary = liver_vol > 0.5
            labeled, n_comp = cast(tuple[np.ndarray, int], ndimage.label(binary))
            if n_comp > 1:
                sizes  = ndimage.sum(binary, labeled, list(range(1, n_comp + 1)))
                binary = labeled == int(np.argmax(sizes)) + 1
            liver_k     = np.argwhere(binary)[:, 2].astype(int)
            liver_found = (len(liver_k) >= MIN_LIVER_VOXELS
                           and float(np.median(liver_k)) > n_k * 0.3)
        del liver_vol

        if liver_found and hip_found:
            z_row["seg_status"] = "ok"
        elif liver_found:
            z_row["seg_status"] = "pubis_missing"
        elif hip_found:
            z_row["seg_status"] = "liver_missing"
        else:
            z_row["seg_status"] = "both_missing"

        apex_k   = int(liver_k.max())    if liver_found else None
        bottom_k = int(min(hip_k))       if hip_found   else None

        if apex_k is None or bottom_k is None or apex_k <= bottom_k:
            if z_row["seg_status"] == "ok":
                z_row["seg_status"] = "invalid_bounds"
            return aec_row, z_row, None, None

        # k(볼륨 인덱스, 오름차순 z=간쪽이 높은 k) → by_inst 내 위치로 환산 (파일명으로 매칭)
        # GetGDCMSeriesFileNames는 경로 구분자가 섞여 나올 수 있어(예: dir\...\file 대신 dir/file)
        # 원본 경로 문자열은 신뢰할 수 없다 — 파일명(fullpath 내 유일)으로 매칭한다.
        apex_file, bottom_file = k_files[apex_k], k_files[bottom_k]
        inst_lookup = {os.path.basename(r["file"]): i for i, r in enumerate(by_inst)}
        apex_pos    = inst_lookup.get(os.path.basename(apex_file))
        bottom_pos  = inst_lookup.get(os.path.basename(bottom_file))
        if apex_pos is None or bottom_pos is None:
            z_row["seg_status"] = "invalid_bounds"
            return aec_row, z_row, None, None

        z_row["liver_upper_k"]     = apex_k
        z_row["pubis_k"]           = bottom_k
        z_row["liver_upper_slice"] = by_inst[apex_pos]["inst"]
        z_row["pubis_slice"]       = by_inst[bottom_pos]["inst"]
        z_row["z_liver_upper"]     = by_inst[apex_pos]["z"]
        z_row["z_pubis"]           = by_inst[bottom_pos]["z"]
        if pd.notna(z_row["z_liver_upper"]) and pd.notna(z_row["z_pubis"]):
            z_row["z_range"] = round(abs(z_row["z_liver_upper"] - z_row["z_pubis"]), 2)

        lo, hi = min(apex_pos, bottom_pos), max(apex_pos, bottom_pos)
        z_row["slices"] = hi - lo + 1
        # by_inst는 인스턴스 오름차순(=간→두덩뼈)이므로 그대로 슬라이싱하면 aec_1=간쪽이 된다
        # (aec_total과 동일한 방향). AEC 신호가 aec_total 제외 기준(CV/range/R²)에 걸린
        # 환자는 aec_cropped에도 넣지 않는다 — 두 파일의 AEC 대상 환자 집합을 항상 일치시킨다.
        if aec_row is not None:
            cropped_liver_to_pubis = [by_inst[i]["aec"] for i in range(lo, hi + 1)]

        # VAT/SAT 등 체성분 — pubis~liver 구간(k-index)만 크롭해 계산
        lo_k, hi_k = min(apex_k, bottom_k), max(apex_k, bottom_k)
        bc = compute_body_composition(img, lo_k, hi_k, tmp_dir, pid, reverse=apex_k > bottom_k)
        bc_row = {"PatientID": pid, "series_description": series_desc, "manufacturer_model": manufacturer, **bc}

    except Exception as e:
        z_row["seg_status"] = f"error:{type(e).__name__}:{e}"
    finally:
        if os.path.isfile(nii_path):
            os.remove(nii_path)
        if os.path.isdir(seg_dir):
            shutil.rmtree(seg_dir, ignore_errors=True)

    return aec_row, z_row, cropped_liver_to_pubis, bc_row


def process_bodycomp_only(pid: int, series_dir: str, zrow: pd.Series, tmp_dir: str) -> dict:
    """이미 z_bounds가 'ok'인 환자용 — liver/hip 재분할 없이 저장된 k-index로 바로 크롭해 체성분만 계산."""
    series_desc, manufacturer = zrow.get("series_description"), zrow.get("manufacturer_model")
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(series_dir)
        if not series_ids:
            r = _empty_bodycomp(pid); r["seg_status"] = "nifti_fail"
            return {"PatientID": pid, "series_description": series_desc, "manufacturer_model": manufacturer, **r}
        k_files = list(reader.GetGDCMSeriesFileNames(series_dir, series_ids[0]))
        reader.SetFileNames(k_files)
        img = reader.Execute()

        n_k = img.GetSize()[2]
        if len(k_files) != n_k:
            r = _empty_bodycomp(pid); r["seg_status"] = "invalid_volume_match"
            return {"PatientID": pid, "series_description": series_desc, "manufacturer_model": manufacturer, **r}

        lo_k = max(0, min(int(zrow["pubis_k"]), int(zrow["liver_upper_k"])))
        hi_k = min(n_k - 1, max(int(zrow["pubis_k"]), int(zrow["liver_upper_k"])))
        if hi_k < lo_k:
            r = _empty_bodycomp(pid); r["seg_status"] = "invalid_bounds"
            return {"PatientID": pid, "series_description": series_desc, "manufacturer_model": manufacturer, **r}

        reverse = int(zrow["liver_upper_k"]) > int(zrow["pubis_k"])
        bc = compute_body_composition(img, lo_k, hi_k, tmp_dir, pid, reverse=reverse)
    except Exception as e:
        bc = _empty_bodycomp(pid)
        bc["seg_status"] = f"error:{type(e).__name__}:{e}"
    return {"PatientID": pid, "series_description": series_desc, "manufacturer_model": manufacturer, **bc}


# ── 출력 (기존 파일 보존 + 신규 행만 append) ────────────────────────────────────

def _interp_aec(vals: np.ndarray, n: int) -> np.ndarray:
    x_orig = np.linspace(0, 1, len(vals))
    x_new  = np.linspace(0, 1, n)
    return np.interp(x_new, x_orig, vals)


def _clean_ws(s):
    return re.sub(r"\s+", " ", str(s)).strip() if pd.notna(s) else s


def _tmp_xlsx_path(path: str) -> str:
    """path와 같은 폴더에 .xlsx 확장자를 유지한 임시 경로를 만든다.
    ".tmp"를 확장자 뒤에 붙이면(예: foo.xlsx.tmp) pandas 2.3.3의 to_excel 확장자 검사가
    깨져 엔진 문자열 대신 property object를 출력하며 실패한다(확장자가 .xlsx로 끝나야 함)."""
    root, ext = os.path.splitext(path)
    return f"{root}.tmp{ext}"


def _write_aec_total(paths: dict, new_rows: list[dict]) -> None:
    if not new_rows:
        return
    has_site_col = False
    old_df = None
    if os.path.exists(paths["aec_total"]):
        old_df = pd.read_excel(paths["aec_total"], sheet_name="aec_total")
        has_site_col = "site" in old_df.columns

    meta_cols = (["site"] if has_site_col else []) + ["PatientID", "n_slices", "series_desc", "manufacturer"]
    max_len = max(len(r["aec_full"]) for r in new_rows)
    if old_df is not None:
        max_len = max(max_len, sum(1 for c in old_df.columns if c.startswith("aec_")))

    out_rows = []
    for r in new_rows:
        row = {c: r.get(c) for c in meta_cols if c != "site"}
        if has_site_col:
            row["site"] = paths.get("_site")
        for i, v in enumerate(r["aec_full"]):
            row[f"aec_{i + 1}"] = v
        out_rows.append(row)

    new_df = pd.DataFrame(out_rows, columns=meta_cols + [f"aec_{i + 1}" for i in range(max_len)])
    combined = pd.concat([old_df, new_df], ignore_index=True) if old_df is not None else new_df
    combined = combined.drop_duplicates(subset=["PatientID"], keep="first")
    combined["n_slices"] = combined["n_slices"].astype("Int64")
    combined["series_desc"] = combined["series_desc"].map(_clean_ws)
    combined = combined.sort_values(["PatientID"]).reset_index(drop=True)
    full_cols = meta_cols + [f"aec_{i + 1}" for i in range(sum(1 for c in combined.columns if c.startswith("aec_")))]
    combined = combined[[c for c in full_cols if c in combined.columns]]
    os.makedirs(os.path.dirname(paths["aec_total"]), exist_ok=True)
    tmp_path = _tmp_xlsx_path(paths["aec_total"])
    if os.path.exists(paths["aec_total"]):
        # aec_cropped와 같은 파일을 공유하는 코호트(new10000)에서는 다른 시트를
        # 지우지 않도록 복사 후 이 시트만 교체한다.
        shutil.copy2(paths["aec_total"], tmp_path)
        with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            combined.to_excel(writer, sheet_name="aec_total", index=False)
    else:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            combined.to_excel(writer, sheet_name="aec_total", index=False)
    os.replace(tmp_path, paths["aec_total"])


def _write_zbounds(paths: dict, new_rows: list[dict]) -> None:
    if not new_rows:
        return
    cols = ["PatientID", "series_description", "manufacturer_model", "n_slices", "inst_min", "slices",
            "liver_upper_slice", "pubis_slice", "liver_upper_k", "pubis_k",
            "z_range", "z_liver_upper", "z_pubis", "seg_status"]
    new_df = pd.DataFrame(new_rows, columns=cols)

    old_df = pd.read_excel(paths["z_bounds"]) if os.path.exists(paths["z_bounds"]) else None
    combined = pd.concat([old_df, new_df], ignore_index=True) if old_df is not None else new_df
    combined = combined.drop_duplicates(subset=["PatientID"], keep="first")
    for c in ("n_slices", "inst_min", "slices", "liver_upper_slice", "pubis_slice", "liver_upper_k", "pubis_k"):
        combined[c] = combined[c].astype("Int64")
    combined["series_description"] = combined["series_description"].map(_clean_ws)
    combined = combined.sort_values(["PatientID"]).reset_index(drop=True)
    combined = combined[[c for c in cols if c in combined.columns]]
    os.makedirs(os.path.dirname(paths["z_bounds"]), exist_ok=True)
    tmp_path = _tmp_xlsx_path(paths["z_bounds"])
    combined.to_excel(tmp_path, index=False, engine="openpyxl")
    os.replace(tmp_path, paths["z_bounds"])


def _write_aec_cropped(paths: dict, new_entries: list[dict]) -> None:
    """new_entries: [{PatientID, series_description, manufacturer_model, kVp, z_range, aec_cropped(list, 간->두덩뼈)}]
    출력 칼럼명은 DLO_Results.xlsx와 맞춰 Series_Desc/Manufacturer로 쓴다."""
    rows = [e for e in new_entries if e.get("aec_cropped")]
    if not rows:
        return

    # (내부 키, 출력 칼럼명) — DLO_Results.xlsx의 Manufacturer/Series_Desc 표기와 통일
    field_map = [("PatientID", "PatientID"), ("series_description", "Series_Desc"),
                 ("manufacturer_model", "Manufacturer"), ("kVp", "kVp")]
    meta_cols = [out for _, out in field_map] + ["n_slices_cropped", "z_range"]

    cropped_rows, interp_rows = [], []
    for e in rows:
        vals = e["aec_cropped"]
        crow = {out: e.get(src) for src, out in field_map}
        crow["n_slices_cropped"] = len(vals)
        crow["z_range"] = e.get("z_range")
        for i, v in enumerate(vals):
            crow[f"aec_{i + 1}"] = v
        cropped_rows.append(crow)

        clean = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        interp = _interp_aec(clean, N_INTERP) if len(clean) >= 2 else np.full(N_INTERP, np.nan)
        interp_rows.append(
            {out: e.get(src) for src, out in field_map}
            | {"n_slices_cropped": len(vals), "z_range": e.get("z_range")}
            | {f"aec_{i + 1}": round(float(v), 2) for i, v in enumerate(interp)}
        )

    max_len = max(r["n_slices_cropped"] for r in cropped_rows)
    new_cropped = pd.DataFrame(cropped_rows, columns=meta_cols + [f"aec_{i + 1}" for i in range(max_len)])
    new_interp  = pd.DataFrame(interp_rows,  columns=meta_cols + [f"aec_{i + 1}" for i in range(N_INTERP)])

    if os.path.exists(paths["aec_cropped"]):
        old_cropped = pd.read_excel(paths["aec_cropped"], sheet_name="aec_cropped")
        old_interp  = pd.read_excel(paths["aec_cropped"], sheet_name=f"aec_{N_INTERP}")
        cropped_df = pd.concat([old_cropped, new_cropped], ignore_index=True)
        interp_df  = pd.concat([old_interp,  new_interp],  ignore_index=True)
    else:
        cropped_df, interp_df = new_cropped, new_interp

    cropped_df = cropped_df.drop_duplicates(subset=["PatientID"], keep="first").sort_values("PatientID").reset_index(drop=True)
    interp_df  = interp_df.drop_duplicates(subset=["PatientID"], keep="first").sort_values("PatientID").reset_index(drop=True)
    cropped_df["n_slices_cropped"] = cropped_df["n_slices_cropped"].astype("Int64")
    interp_df["n_slices_cropped"]  = interp_df["n_slices_cropped"].astype("Int64")
    cropped_df["Series_Desc"] = cropped_df["Series_Desc"].map(_clean_ws)
    interp_df["Series_Desc"]  = interp_df["Series_Desc"].map(_clean_ws)

    cropped_aec_cols = [f"aec_{i + 1}" for i in range(sum(1 for c in cropped_df.columns if c.startswith("aec_")))]
    cropped_df = cropped_df[[c for c in meta_cols if c in cropped_df.columns] + cropped_aec_cols]
    interp_df  = interp_df[[c for c in meta_cols if c in interp_df.columns] + [f"aec_{i + 1}" for i in range(N_INTERP)]]

    os.makedirs(os.path.dirname(paths["aec_cropped"]), exist_ok=True)
    tmp_path = _tmp_xlsx_path(paths["aec_cropped"])
    if os.path.exists(paths["aec_cropped"]):
        # aec_total과 같은 파일을 공유하는 코호트(new10000)에서는 다른 시트를
        # 지우지 않도록 복사 후 이 시트들만 교체한다.
        shutil.copy2(paths["aec_cropped"], tmp_path)
        with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            cropped_df.to_excel(writer, sheet_name="aec_cropped", index=False)
            interp_df.to_excel(writer, sheet_name=f"aec_{N_INTERP}", index=False)
    else:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            cropped_df.to_excel(writer, sheet_name="aec_cropped", index=False)
            interp_df.to_excel(writer, sheet_name=f"aec_{N_INTERP}", index=False)
    os.replace(tmp_path, paths["aec_cropped"])


BODY_COMP_METRIC_SHEETS = [("IMATA", "IMATA_sum_cm2"), ("NAMA", "NAMA_sum_cm2"), ("LAMA", "LAMA_sum_cm2"),
                           ("VAT", "VAT_sum_cm2"), ("SAT", "SAT_sum_cm2")]


def _read_body_comp(path: str) -> pd.DataFrame | None:
    """body_composition.xlsx를 단일 표로 읽는다. 파일이 없으면 None.
    현재 포맷(단일 body_composition 시트)이 없으면, 한때 사용했던 항목별 시트(IMATA/NAMA/LAMA/VAT/SAT)
    포맷으로 간주해 PatientID 기준으로 병합해 반환한다(과거 실행분 마이그레이션)."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_excel(path, sheet_name="body_composition")
    except ValueError:
        pass
    try:
        merged = pd.read_excel(path, sheet_name=BODY_COMP_METRIC_SHEETS[0][0])
        for name, value_col in BODY_COMP_METRIC_SHEETS[1:]:
            metric_df = pd.read_excel(path, sheet_name=name)
            merged = merged.merge(metric_df[["PatientID", value_col]], on="PatientID", how="left")
        return merged
    except ValueError:
        return None


def _write_body_composition(paths: dict, new_rows: list[dict]) -> None:
    """체성분(IMATA/NAMA/LAMA/VAT/SAT) 합계를 단일 body_composition 시트에 저장한다.
    칼럼: PatientID, series_description, manufacturer_model, kVp, seg_status, n_slices_range,
    IMATA_sum_cm2, NAMA_sum_cm2, LAMA_sum_cm2, VAT_sum_cm2, SAT_sum_cm2."""
    if not new_rows:
        return
    cols = (["PatientID", "series_description", "manufacturer_model", "kVp", "seg_status", "n_slices_range"]
            + [value_col for _, value_col in BODY_COMP_METRIC_SHEETS])
    new_df = pd.DataFrame(new_rows, columns=cols)

    old_df = _read_body_comp(paths["body_comp"])
    combined = pd.concat([old_df, new_df], ignore_index=True) if old_df is not None else new_df
    combined = combined.drop_duplicates(subset=["PatientID"], keep="first")
    combined["series_description"] = combined["series_description"].map(_clean_ws)
    combined = combined.sort_values("PatientID").reset_index(drop=True)
    combined = combined[[c for c in cols if c in combined.columns]]
    os.makedirs(os.path.dirname(paths["body_comp"]), exist_ok=True)
    tmp_path = _tmp_xlsx_path(paths["body_comp"])
    if os.path.exists(paths["body_comp"]):
        # body_comp_cropped와 같은 파일을 공유하는 코호트(new10000)에서는 다른 시트를
        # 지우지 않도록 복사 후 이 시트만 교체한다.
        shutil.copy2(paths["body_comp"], tmp_path)
        with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            combined.to_excel(writer, sheet_name="body_composition", index=False)
    else:
        combined.to_excel(tmp_path, sheet_name="body_composition", index=False, engine="openpyxl")
    os.replace(tmp_path, paths["body_comp"])


def _write_body_comp_cropped(paths: dict, new_rows: list[dict]) -> None:
    """new_rows: compute_body_composition() 결과가 담긴 body_composition 행들(seg_status=='ok'만 사용).
    aec_cropped와 동일한 방향(간→두덩뼈)·형식(raw + N_INTERP포인트 보간)으로 SFA/VFA/TAMA/LAMA/NAMA/IMATA
    6개 조직의 슬라이스별 면적(cm2) 곡선을, 조직×raw/interp 조합별 시트(예: SFA_raw, SFA_interp, ...
    총 12개)로 나눠 저장한다."""
    rows = [r for r in new_rows if r.get("seg_status") == "ok" and r.get("SFA_slices")]
    if not rows:
        return

    meta_cols = ["PatientID", "series_description", "manufacturer_model", "n_slices_range"]
    max_len = max(len(r["SFA_slices"]) for r in rows)

    os.makedirs(os.path.dirname(paths["body_comp_cropped"]), exist_ok=True)
    sheets: dict[str, pd.DataFrame] = {}
    for tk in TISSUE_KEYS:
        raw_rows, interp_rows = [], []
        for r in rows:
            vals = r[f"{tk}_slices"]
            raw_row = {c: r.get(c) for c in meta_cols}
            for i, v in enumerate(vals):
                raw_row[f"{tk}_{i + 1}"] = round(float(v), 2)
            raw_rows.append(raw_row)

            clean = np.asarray(vals, dtype=float)
            interp = _interp_aec(clean, N_INTERP) if len(clean) >= 2 else np.full(N_INTERP, np.nan)
            interp_row = {c: r.get(c) for c in meta_cols}
            for i, v in enumerate(interp):
                interp_row[f"{tk}_{i + 1}"] = round(float(v), 2)
            interp_rows.append(interp_row)

        for suffix, new_df in (("raw", pd.DataFrame(raw_rows, columns=meta_cols + [f"{tk}_{i + 1}" for i in range(max_len)])),
                               ("interp", pd.DataFrame(interp_rows, columns=meta_cols + [f"{tk}_{i + 1}" for i in range(N_INTERP)]))):
            sheet_name = f"{tk}_{suffix}"
            old_df = None
            if os.path.exists(paths["body_comp_cropped"]):
                try:
                    old_df = pd.read_excel(paths["body_comp_cropped"], sheet_name=sheet_name)
                except ValueError:
                    old_df = None  # 이전 포맷(raw/interp_128 통합 시트)에는 이 시트가 없음
            combined = pd.concat([old_df, new_df], ignore_index=True) if old_df is not None else new_df
            combined = combined.drop_duplicates(subset=["PatientID"], keep="first").sort_values("PatientID").reset_index(drop=True)
            combined["n_slices_range"] = combined["n_slices_range"].astype("Int64")
            combined["series_description"] = combined["series_description"].map(_clean_ws)

            if suffix == "raw":
                full_max_len = max(max_len, sum(1 for c in combined.columns if c.startswith(f"{tk}_")))
                full_cols = meta_cols + [f"{tk}_{i + 1}" for i in range(full_max_len)]
            else:
                full_cols = meta_cols + [f"{tk}_{i + 1}" for i in range(N_INTERP)]
            sheets[sheet_name] = combined[[c for c in full_cols if c in combined.columns]]

    tmp_path = _tmp_xlsx_path(paths["body_comp_cropped"])
    if os.path.exists(paths["body_comp_cropped"]):
        # body_comp와 같은 파일을 공유하는 코호트(new10000)에서는 다른 시트를
        # 지우지 않도록 복사 후 이 시트들만 교체한다.
        shutil.copy2(paths["body_comp_cropped"], tmp_path)
        with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    os.replace(tmp_path, paths["body_comp_cropped"])


def _merge_vat_sat_into_dlo(paths: dict) -> None:
    """{site}_body_composition.xlsx(seg_status=='ok')의 VAT/SAT를 DLO_Results.xlsx에 병합.
    DLO_Results.xlsx의 다른 행/열은 전혀 건드리지 않고, 비어 있는 두 칼럼만 채운다."""
    bc = _read_body_comp(paths["body_comp"])
    if bc is None:
        return
    bc_ok = bc[bc["seg_status"] == "ok"][["PatientID", "VAT_sum_cm2", "SAT_sum_cm2"]]
    lookup = {int(r["PatientID"]): (r["VAT_sum_cm2"], r["SAT_sum_cm2"]) for _, r in bc_ok.iterrows()}
    if not lookup:
        return

    wb = openpyxl.load_workbook(paths["dlo_path"])
    ws = wb.active
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    pid_col = header.index("PatientID") + 1
    vat_col = header.index("VAT(내장지방)_SUM") + 1
    sat_col = header.index("SAT(피하지방)_SUM") + 1

    filled = 0
    for row in range(2, ws.max_row + 1):
        pid = ws.cell(row=row, column=pid_col).value
        if pid is None:
            continue
        vals = lookup.get(int(pid))
        if vals is None:
            continue
        vat, sat = vals
        if pd.notna(vat):
            ws.cell(row=row, column=vat_col, value=float(vat))
        if pd.notna(sat):
            ws.cell(row=row, column=sat_col, value=float(sat))
        filled += 1

    wb.save(paths["dlo_path"])
    print(f"  [{paths['_site']}] DLO_Results VAT/SAT 병합: {filled}행")


# ── 체크포인트 (시도했으나 aec_total 필터에서 제외된 환자 등 재처리 방지) ─────────

def _load_checkpoint(path: str) -> set[int]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return set()


def _save_checkpoint(path: str, attempted: set[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(attempted, f)


# ── 병렬 실행(shard) 지원 ────────────────────────────────────────────────────

def _shard_filter(pid: int, shard_id: int, num_shards: int) -> bool:
    return num_shards <= 1 or (pid % num_shards) == shard_id


def run_site(site: str, shard_id: int = 0, num_shards: int = 1):
    """shard_id/num_shards>1이면 이 프로세스는 PatientID % num_shards == shard_id인 환자만
    처리해 자신만의 shard 파일에 저장한다 (다른 shard 프로세스와 파일 경합 없음).
    num_shards==1이면 기존처럼 메인 파일에 바로 append한다."""
    sharded = num_shards > 1
    paths = site_paths(site, shard=shard_id if sharded else None)
    paths["_site"] = site
    tag = f"{site}#{shard_id}" if sharded else site
    print("=" * 78)
    print(f"[{tag}] 시작  |  DICOM: {paths['dicom_base']}")

    dlo = pd.read_excel(paths["dlo_path"])[["PatientID", "Series_Desc", "kVp"]]
    dlo["PatientID"] = dlo["PatientID"].astype(int)

    # ── 완료 여부는 항상 '메인' z_bounds/checkpoint 기준으로 판단(shard 파일은 재시작용) ──
    check_paths = site_paths(site)  # 메인(비-shard) 경로
    done_pids = set()
    if os.path.exists(check_paths["z_bounds"]):
        done_pids |= set(pd.read_excel(check_paths["z_bounds"])["PatientID"].astype(int))
    if sharded and os.path.exists(paths["z_bounds"]):
        done_pids |= set(pd.read_excel(paths["z_bounds"])["PatientID"].astype(int))
    attempted = _load_checkpoint(paths["checkpoint"])
    done_pids |= attempted

    pending = dlo[~dlo["PatientID"].isin(done_pids)]
    if sharded:
        pending = pending[pending["PatientID"].apply(lambda p: _shard_filter(p, shard_id, num_shards))]
    print(f"[{tag}] 대상 {len(dlo)}명  |  완료/제외 {len(done_pids)}명  |  처리 예정(이 shard) {len(pending)}명")

    # ── 1단계: z_bounds/aec_total/aec_cropped/body_comp 전체 파이프라인 (신규 환자) ──
    aec_rows, z_rows, cropped_entries, bc_rows = [], [], [], []
    n_processed = 0

    if not pending.empty:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for _, r in tqdm(pending.iterrows(), total=len(pending), desc=f"{tag} 처리"):
                pid = int(r["PatientID"])
                patient_dir = os.path.join(paths["dicom_base"], str(pid))

                if not os.path.isdir(patient_dir):
                    z_rows.append(_empty_zbounds(pid, None, None, "no_dicom"))
                else:
                    folder = find_series_folder(patient_dir, r["Series_Desc"])
                    if folder is None:
                        z_rows.append(_empty_zbounds(pid, None, None, "no_series"))
                    else:
                        series_dir = os.path.join(patient_dir, folder)
                        try:
                            aec_row, z_row, cropped, bc_row = process_patient(pid, series_dir, tmp_dir)
                        except Exception as e:
                            aec_row, cropped, bc_row = None, None, None
                            z_row = _empty_zbounds(pid, None, None, f"error:{type(e).__name__}:{e}")
                        if aec_row is not None:
                            aec_rows.append(aec_row)
                        z_rows.append(z_row)
                        if cropped is not None:
                            cropped_entries.append({
                                "PatientID": pid,
                                "series_description": z_row["series_description"],
                                "manufacturer_model": z_row["manufacturer_model"],
                                "kVp": r["kVp"],
                                "z_range": z_row["z_range"],
                                "aec_cropped": cropped,
                            })
                        if bc_row is not None:
                            bc_row["kVp"] = r["kVp"]
                            bc_rows.append(bc_row)
                        tqdm.write(f"  PID {pid}: {z_row['seg_status']}")

                attempted.add(pid)
                n_processed += 1

                if n_processed % BATCH_SIZE == 0 or n_processed == len(pending):
                    _write_aec_total(paths, aec_rows)
                    _write_zbounds(paths, z_rows)
                    _write_aec_cropped(paths, cropped_entries)
                    _write_body_composition(paths, bc_rows)
                    _write_body_comp_cropped(paths, bc_rows)
                    _save_checkpoint(paths["checkpoint"], attempted)
                    aec_rows, z_rows, cropped_entries, bc_rows = [], [], [], []
                    tqdm.write(f"  [{tag} 체크포인트 저장 | {n_processed}/{len(pending)}]")

    if os.path.exists(paths["checkpoint"]):
        os.remove(paths["checkpoint"])

    # ── 2단계: z_bounds는 이미 'ok'이지만 body_composition이 아직 없는 환자(체성분만) ──
    run_bodycomp_only_stage(site, shard_id, num_shards, sharded, paths, check_paths)

    print(f"[{tag}] 완료")


def run_bodycomp_only_stage(site: str, shard_id: int, num_shards: int, sharded: bool,
                            paths: dict, check_paths: dict):
    if not os.path.exists(check_paths["z_bounds"]):
        return
    zdf = pd.read_excel(check_paths["z_bounds"])
    zdf = zdf[zdf["seg_status"].astype(str) == "ok"]

    # 완료 여부는 body_comp_cropped(슬라이스별 곡선)을 기준으로 판단한다 — body_comp.xlsx(합계)에는
    # 이미 있지만 body_comp_cropped.xlsx에는 없는 환자(과거 실행분)도 슬라이스 곡선을 새로 만들도록
    # 재처리 대상에 포함시켜야 하기 때문이다. seg_status가 'ok'가 아니었던 환자(세그멘테이션 실패)는
    # body_comp.xlsx만 보고 판단해 재시도하지 않는다(어차피 z_bounds=='ok'인 환자만 이 단계 대상이라
    # 이전에 실패했다면 대부분 재실행해도 다시 실패할 가능성이 높음).
    have_bc_pids = set()
    for p in (check_paths, paths if sharded else None):
        if p is None or not os.path.exists(p["body_comp_cropped"]):
            continue
        try:
            df = pd.read_excel(p["body_comp_cropped"], sheet_name=f"{TISSUE_KEYS[0]}_raw")
            have_bc_pids |= set(df["PatientID"].astype(int))
        except ValueError:
            pass  # 이전 포맷(raw/interp_128 통합 시트)만 있음 — 새 포맷으로 재생성 대상
    failed_bc_pids = set()
    bc_df = _read_body_comp(check_paths["body_comp"])
    if bc_df is not None:
        failed_bc_pids |= set(bc_df.loc[bc_df["seg_status"] != "ok", "PatientID"].astype(int))
    if sharded:
        bc_df = _read_body_comp(paths["body_comp"])
        if bc_df is not None:
            failed_bc_pids |= set(bc_df.loc[bc_df["seg_status"] != "ok", "PatientID"].astype(int))

    zdf = zdf[~zdf["PatientID"].astype(int).isin(have_bc_pids | failed_bc_pids)]
    if sharded:
        zdf = zdf[zdf["PatientID"].astype(int).apply(lambda p: _shard_filter(p, shard_id, num_shards))]
    if zdf.empty:
        return

    tag = f"{site}#{shard_id}" if sharded else site
    print(f"[{tag}] 체성분(VAT/SAT)만 필요한 환자 {len(zdf)}명 처리")

    dlo_kvp = pd.read_excel(check_paths["dlo_path"])[["PatientID", "kVp"]]
    dlo_kvp["PatientID"] = dlo_kvp["PatientID"].astype(int)
    kvp_by_pid = dict(zip(dlo_kvp["PatientID"], dlo_kvp["kVp"]))

    bc_rows: list[dict] = []
    n_processed = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        for _, zrow in tqdm(zdf.iterrows(), total=len(zdf), desc=f"{tag} 체성분"):
            pid = int(zrow["PatientID"])
            patient_dir = os.path.join(paths["dicom_base"], str(pid))
            if not os.path.isdir(patient_dir):
                bc_rows.append({"PatientID": pid, "series_description": None, "manufacturer_model": None,
                                **_empty_bodycomp(pid)} | {"seg_status": "no_dicom", "kVp": kvp_by_pid.get(pid)})
            else:
                folder = find_series_folder(patient_dir, zrow.get("series_description"))
                if folder is None:
                    bc_rows.append({"PatientID": pid, "series_description": None, "manufacturer_model": None,
                                    **_empty_bodycomp(pid)} | {"seg_status": "no_series", "kVp": kvp_by_pid.get(pid)})
                else:
                    series_dir = os.path.join(patient_dir, folder)
                    try:
                        bc_row = process_bodycomp_only(pid, series_dir, zrow, tmp_dir)
                    except Exception as e:
                        bc_row = {"PatientID": pid, "series_description": zrow.get("series_description"),
                                  "manufacturer_model": zrow.get("manufacturer_model"), **_empty_bodycomp(pid)}
                        bc_row["seg_status"] = f"error:{type(e).__name__}:{e}"
                    bc_row["kVp"] = kvp_by_pid.get(pid)
                    bc_rows.append(bc_row)
                    tqdm.write(f"  PID {pid}: {bc_row['seg_status']}")

            n_processed += 1
            if n_processed % BATCH_SIZE == 0 or n_processed == len(zdf):
                _write_body_composition(paths, bc_rows)
                _write_body_comp_cropped(paths, bc_rows)
                bc_rows = []
                tqdm.write(f"  [{tag} 체성분 체크포인트 저장 | {n_processed}/{len(zdf)}]")


# ── shard 결과 병합 ──────────────────────────────────────────────────────────

def merge_shards(site: str, num_shards: int, delete_shard_files: bool = False):
    """shard별 xlsx를 메인 xlsx로 합친다 (메인 파일 보존 + shard 신규분 추가).
    아직 실행 중인 shard가 있을 수 있으므로(중간 동기화 목적) 기본값은 shard 파일을 지우지 않는다.
    delete_shard_files=True는 해당 shard 프로세스가 완전히 종료된 게 확인된 뒤에만 사용할 것
    (실행 중에 지우면 그 shard가 다음 배치 저장 때 자기 파일을 처음부터 새로 만들어
    지금까지 누적한 진행분을 잃는다)."""
    main_paths = site_paths(site)
    main_paths["_site"] = site
    for shard_id in range(num_shards):
        sp = site_paths(site, shard=shard_id)
        if os.path.exists(sp["aec_total"]):
            df = pd.read_excel(sp["aec_total"], sheet_name="aec_total")
            rows = [{**row, "aec_full": [row[c] for c in df.columns if c.startswith("aec_")]}
                    for row in df.to_dict("records")]
            _write_aec_total(main_paths, rows)
        if os.path.exists(sp["z_bounds"]):
            df = pd.read_excel(sp["z_bounds"])
            _write_zbounds(main_paths, df.to_dict("records"))
        if os.path.exists(sp["aec_cropped"]):
            cdf = pd.read_excel(sp["aec_cropped"], sheet_name="aec_cropped")
            # 구버전 shard 파일(리네임 전)과 신버전(Series_Desc/Manufacturer) 모두 지원
            series_col = "Series_Desc" if "Series_Desc" in cdf.columns else "series_description"
            mfr_col    = "Manufacturer" if "Manufacturer" in cdf.columns else "manufacturer_model"
            aec_cols = [c for c in cdf.columns if c.startswith("aec_")]
            entries = []
            for row in cdf.to_dict("records"):
                n = int(row["n_slices_cropped"])
                entries.append({
                    "PatientID": row["PatientID"],
                    "series_description": row.get(series_col),
                    "manufacturer_model": row.get(mfr_col),
                    "kVp": row.get("kVp"),
                    "z_range": row.get("z_range"),
                    "aec_cropped": [row[c] for c in aec_cols[:n]],
                })
            _write_aec_cropped(main_paths, entries)
        df = _read_body_comp(sp["body_comp"])
        if df is not None:
            _write_body_composition(main_paths, df.to_dict("records"))
        if os.path.exists(sp["body_comp_cropped"]):
            try:
                tissue_raw = {tk: pd.read_excel(sp["body_comp_cropped"], sheet_name=f"{tk}_raw").set_index("PatientID")
                              for tk in TISSUE_KEYS}
            except ValueError:
                print(f"[{site}] shard {shard_id} body_comp_cropped 이전 포맷(raw/interp_128 통합 시트) — 병합 건너뜀, 재처리 필요")
            else:
                base_df = tissue_raw[TISSUE_KEYS[0]].reset_index()
                rows = []
                for row in base_df.to_dict("records"):
                    pid = int(row["PatientID"])
                    n = int(row["n_slices_range"])
                    entry = {k: row[k] for k in ("PatientID", "series_description", "manufacturer_model", "n_slices_range")}
                    entry["seg_status"] = "ok"
                    for tk in TISSUE_KEYS:
                        trow = tissue_raw[tk].loc[pid]
                        entry[f"{tk}_slices"] = [trow[f"{tk}_{i + 1}"] for i in range(n)]
                    rows.append(entry)
                _write_body_comp_cropped(main_paths, rows)
        print(f"[{site}] shard {shard_id} 병합 완료")

        if delete_shard_files:
            for key in ("aec_total", "z_bounds", "aec_cropped", "body_comp", "body_comp_cropped", "checkpoint"):
                if os.path.exists(sp[key]):
                    os.remove(sp[key])
            print(f"[{site}] shard {shard_id} 임시 파일 삭제 완료")

    _merge_vat_sat_into_dlo(main_paths)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=SITES, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--delete-shards", action="store_true",
                        help="병합 후 shard 임시 파일 삭제 (해당 shard 프로세스가 모두 종료된 뒤에만 사용)")
    args = parser.parse_args()

    sites = [args.site] if args.site else SITES

    if args.merge_only:
        for site in sites:
            merge_shards(site, args.num_shards, delete_shard_files=args.delete_shards)
        return

    for site in sites:
        run_site(site, shard_id=args.shard, num_shards=args.num_shards)
        if args.num_shards <= 1:
            main_paths = site_paths(site)
            main_paths["_site"] = site
            _merge_vat_sat_into_dlo(main_paths)


if __name__ == "__main__":
    main()
