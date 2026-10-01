"""
D:/데이터서비스팀 요청(이홍선) 데이터셋 처리.

환자 폴더 하나에 여러 Series(실제 axial 촬영, CORONAL/SAGITTAL 재구성, SCOUT, DOSE report 등)의
dcm 파일이 섞여 있다. Patient·Series Description 별로 axial 시리즈만 골라 TotalSegmentator로
pubis(hip 최하단)~liver(상단) 경계를 구하고, 그 구간의 AEC(XRayTubeCurrent)를 pubis→liver 순으로 저장한다.
AEC 배열은 aec_total/aec_cropped 모두 z 오름차순(=pubis→liver 방향)으로 통일한다.

[시리즈 판별]
  - ImageOrientationPatient로 axial 여부 판정 (CORONAL/SAGITTAL 재구성 제외)
  - slice 수 < MIN_SLICES → SCOUT/DOSE report 등으로 간주해 제외
  - ImageType(DERIVED/REFORMATTED) 또는 SeriesDescription에 "MPR" 포함 → 재구성 시리즈로 간주해 제외

[출력]
  {OUT_DIR}/aec_total.xlsx    - 시리즈 전체 slice AEC (pubis→liver 순)
  {OUT_DIR}/z_bounds.xlsx     - TotalSegmentator로 구한 pubis/liver 경계
  {OUT_DIR}/aec_cropped.xlsx  - pubis~liver 구간 크롭 AEC (pubis→liver 순) + 128포인트 보간
"""

import os
import io
import re
import sys
import shutil
import pickle
import logging
import tempfile
import contextlib
from typing import cast

import numpy as np
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AEC_DIR      = os.path.join(PROJECT_ROOT, "aec")

DICOM_BASE      = r"D:\데이터서비스팀 요청(이홍선)"
CHECKPOINT_PATH = os.path.join(AEC_DIR, "checkpoint.pkl")

AEC_TOTAL_PATH   = os.path.join(AEC_DIR, "aec_total.xlsx")
ZBOUNDS_PATH     = os.path.join(AEC_DIR, "z_bounds.xlsx")
AEC_CROPPED_PATH = os.path.join(AEC_DIR, "aec_cropped.xlsx")

ROI_SUBSET       = ["liver", "hip_left", "hip_right"]
MIN_SLICES       = 20      # 이보다 slice 수가 적으면 scout/dose 등으로 간주해 제외
MIN_LIVER_VOXELS = 3000
N_INTERP         = 128
BATCH_SIZE       = 5      # 몇 명 처리마다 체크포인트/중간 저장을 할지

TEST_N = None   # 정수 지정 시 앞 N명 환자만 처리 (검증용)


@contextlib.contextmanager
def _silence():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ── 시리즈 탐색 ───────────────────────────────────────────────────────────────

def _patient_id_from_folder(folder_name: str) -> str:
    parts = folder_name.split("_")
    return parts[1] if len(parts) >= 2 else folder_name


def _is_axial(iop) -> bool:
    """ImageOrientationPatient의 row/col 벡터 외적(slice normal)이 z축에 가까우면 axial."""
    if iop is None or len(iop) != 6:
        return False
    row    = np.array(iop[0:3], dtype=float)
    col    = np.array(iop[3:6], dtype=float)
    normal = np.cross(row, col)
    return abs(normal[2]) > 0.9


_REFORMAT_DESC_KEYWORDS = ("MPR", "COR", "CORONAL", "SAG", "SAGITTAL", "SPO")


def _desc_has_reformat_keyword(desc: str) -> bool:
    """SeriesDescription에 재구성/비-axial 관련 키워드(MPR/COR/SAG/SPO 등)가 단어 단위로 포함되는지 확인."""
    tokens = re.split(r"[^A-Z0-9]+", str(desc).upper())
    return any(kw in tokens for kw in _REFORMAT_DESC_KEYWORDS)


def _is_mpr(hdr) -> bool:
    """ImageType에 DERIVED/REFORMATTED가 있거나 SeriesDescription에 재구성 관련 키워드(MPR/COR/SAG 등)가
    단어 단위로 포함되면 재구성 시리즈로 간주."""
    image_type = [str(v).upper() for v in getattr(hdr, "ImageType", [])]
    if "DERIVED" in image_type or "REFORMATTED" in image_type:
        return True
    return _desc_has_reformat_keyword(getattr(hdr, "SeriesDescription", ""))


def find_axial_series(patient_dir: str) -> list[dict]:
    """환자 폴더(여러 Series의 dcm이 섞인 상태)에서 axial 시리즈만 골라 반환."""
    reader = sitk.ImageSeriesReader()
    try:
        series_ids = reader.GetGDCMSeriesIDs(patient_dir)
    except Exception:
        series_ids = ()

    found = []
    for series_uid in series_ids or ():
        files = list(reader.GetGDCMSeriesFileNames(patient_dir, series_uid))
        if len(files) < MIN_SLICES:
            continue

        hdrs = []
        for fp in files:
            try:
                hdrs.append(pydicom.dcmread(fp, stop_before_pixels=True))
            except Exception:
                hdrs = []
                break
        if not hdrs:
            continue

        if _is_mpr(hdrs[0]):
            continue
        # 시리즈 내 파일 중 하나라도 axial이 아니면 시리즈 전체를 제외한다.
        if not all(_is_axial(getattr(h, "ImageOrientationPatient", None)) for h in hdrs):
            continue

        found.append({
            "series_uid":   series_uid,
            "series_desc":  str(getattr(hdrs[0], "SeriesDescription", "")).strip(),
            "manufacturer": str(getattr(hdrs[0], "ManufacturerModelName", "")),
            "files":        files,
        })
    return found


def read_series_slices(files: list[str]) -> list[dict] | None:
    """z 오름차순(=pubis→liver 방향)으로 정렬된 slice 정보를 반환.

    같은 SeriesInstanceUID인데도 Rows/Columns가 다른 파일이 섞여 있으면
    (스캐너가 다른 reformat을 같은 series UID로 저장하는 경우) ImageSeriesReader가
    size mismatch로 실패하므로, 다수(majority) 크기와 다른 파일은 제외한다.
    """
    rows = []
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
        except Exception:
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        z   = float(ipp[2]) if ipp is not None else float("nan")
        aec = getattr(ds, "XRayTubeCurrent", None) or getattr(ds, "TubeCurrent", None)
        rows.append({
            "file": fp,
            "inst": int(getattr(ds, "InstanceNumber", 0)),
            "z":    z,
            "aec":  float(aec) if aec is not None else float("nan"),
            "dim":  (int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))),
        })
    if not rows or all(np.isnan(r["z"]) for r in rows):
        return None

    dims = [r["dim"] for r in rows]
    majority_dim = max(set(dims), key=dims.count)
    rows = [r for r in rows if r["dim"] == majority_dim]

    rows.sort(key=lambda r: r["z"])
    return rows


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


def extract_series(pid: str, series_uid: str, series_desc: str, manufacturer: str,
                    files: list[str], tmp_dir: str) -> dict:
    """단일 axial 시리즈에서 pubis~liver 경계와 크롭 AEC를 추출 (pubis→liver 순)."""
    result = {
        "PatientID": pid, "series_uid": series_uid,
        "series_desc": series_desc, "manufacturer": manufacturer,
        "n_slices": np.nan, "seg_status": "failed",
        "pubis_k": np.nan, "liver_k": np.nan,
        "pubis_inst": np.nan, "liver_inst": np.nan,
        "z_pubis": np.nan, "z_liver": np.nan, "z_range": np.nan,
        "aec_full": None, "aec_cropped": None,
    }

    rows = read_series_slices(files)
    if rows is None:
        result["seg_status"] = "no_position_data"
        return result

    n_k = len(rows)
    result["n_slices"] = n_k
    result["aec_full"] = [r["aec"] for r in rows]  # k 오름차순 = pubis→liver 방향

    nii_path = os.path.join(tmp_dir, f"{pid}_{series_uid}.nii.gz")
    seg_dir  = os.path.join(tmp_dir, f"{pid}_{series_uid}_seg")
    try:
        # z 오름차순으로 정렬한 파일 순서 그대로 볼륨을 만들어 k축 방향을 고정한다 (k=0: pubis 방향, k=n-1: liver 방향).
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([r["file"] for r in rows])
        sitk.WriteImage(reader.Execute(), nii_path)

        os.makedirs(seg_dir, exist_ok=True)
        with _silence():
            totalsegmentator(input=nii_path, output=seg_dir, task="total",
                             roi_subset=ROI_SUBSET, fast=False, quiet=True, device=DEVICE)

        if int(cast(Nifti1Image, nib.load(nii_path)).shape[2]) != n_k:
            result["seg_status"] = "invalid_volume_match"
            return result

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
            result["seg_status"] = "ok"
        elif liver_found:
            result["seg_status"] = "pubis_missing"
        elif hip_found:
            result["seg_status"] = "liver_missing"
        else:
            result["seg_status"] = "both_missing"

        liver_k_val = int(liver_k.max()) if liver_found else None
        pubis_k_val = int(min(hip_k))    if hip_found   else None

        if liver_k_val is not None:
            result["liver_k"]    = liver_k_val
            result["liver_inst"] = rows[liver_k_val]["inst"]
            result["z_liver"]    = rows[liver_k_val]["z"]
        if pubis_k_val is not None:
            result["pubis_k"]    = pubis_k_val
            result["pubis_inst"] = rows[pubis_k_val]["inst"]
            result["z_pubis"]    = rows[pubis_k_val]["z"]

        if liver_k_val is not None and pubis_k_val is not None and pubis_k_val <= liver_k_val:
            result["z_range"]     = round(abs(result["z_liver"] - result["z_pubis"]), 2)
            result["aec_cropped"] = [r["aec"] for r in rows[pubis_k_val: liver_k_val + 1]]
        elif result["seg_status"] == "ok":
            result["seg_status"] = "invalid_bounds"

    except Exception as e:
        result["seg_status"] = f"error:{type(e).__name__}:{e}"
    finally:
        if os.path.isfile(nii_path):
            os.remove(nii_path)
        if os.path.isdir(seg_dir):
            shutil.rmtree(seg_dir, ignore_errors=True)

    return result


# ── 출력 ──────────────────────────────────────────────────────────────────────

def _interp_aec(vals: np.ndarray, n: int) -> np.ndarray:
    x_orig = np.linspace(0, 1, len(vals))
    x_new  = np.linspace(0, 1, n)
    return np.interp(x_new, x_orig, vals)


def _write_aec_total(results: list[dict]) -> None:
    rows = [r for r in results if r.get("aec_full")]
    if not rows:
        return
    meta_cols = ["PatientID", "series_desc", "manufacturer", "n_slices", "seg_status"]
    max_len   = max(len(r["aec_full"]) for r in rows)
    out = []
    for r in rows:
        row = {c: r.get(c) for c in meta_cols}
        for i, v in enumerate(r["aec_full"]):
            row[f"aec_{i + 1}"] = v
        out.append(row)
    df = pd.DataFrame(out, columns=meta_cols + [f"aec_{i + 1}" for i in range(max_len)])
    df["n_slices"] = df["n_slices"].astype("Int64")
    df = df.sort_values(["PatientID", "series_desc"]).reset_index(drop=True)
    os.makedirs("aec", exist_ok=True)
    with pd.ExcelWriter(AEC_TOTAL_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="aec_total", index=False)


def _write_zbounds(results: list[dict]) -> None:
    if not results:
        return
    cols = ["PatientID", "series_desc", "manufacturer", "n_slices", "seg_status",
            "pubis_inst", "liver_inst", "pubis_k", "liver_k",
            "z_pubis", "z_liver", "z_range"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in results], columns=cols)
    for c in ("n_slices", "pubis_inst", "liver_inst", "pubis_k", "liver_k"):
        df[c] = df[c].astype("Int64")
    df = df.sort_values(["PatientID", "series_desc"]).reset_index(drop=True)
    os.makedirs("aec", exist_ok=True)
    df.to_excel(ZBOUNDS_PATH, index=False)


def _write_aec_cropped(results: list[dict]) -> None:
    rows = [r for r in results if r.get("seg_status") == "ok" and r.get("aec_cropped")]
    if not rows:
        return

    meta_cols = ["PatientID", "series_desc", "manufacturer", "z_range"]
    max_len   = max(len(r["aec_cropped"]) for r in rows)
    cropped_out, interp_out = [], []

    for r in rows:
        vals = r["aec_cropped"]  # 이미 pubis→liver 순
        crow = {c: r.get(c) for c in meta_cols}
        crow["n_slices_cropped"] = len(vals)
        for i, v in enumerate(vals):
            crow[f"aec_{i + 1}"] = v
        cropped_out.append(crow)

        clean  = np.array([v for v in vals if not np.isnan(v)], dtype=float)
        interp = _interp_aec(clean, N_INTERP) if len(clean) >= 2 else np.full(N_INTERP, np.nan)
        interp_out.append(
            {c: r.get(c) for c in meta_cols}
            | {"n_slices_cropped": len(vals)}
            | {f"aec_{i + 1}": round(float(v), 2) for i, v in enumerate(interp)}
        )

    cropped_cols = meta_cols + ["n_slices_cropped"] + [f"aec_{i + 1}" for i in range(max_len)]
    interp_cols  = meta_cols + ["n_slices_cropped"] + [f"aec_{i + 1}" for i in range(N_INTERP)]

    cropped_df = pd.DataFrame(cropped_out, columns=cropped_cols).sort_values(["PatientID", "series_desc"]).reset_index(drop=True)
    interp_df  = pd.DataFrame(interp_out,  columns=interp_cols).sort_values(["PatientID", "series_desc"]).reset_index(drop=True)
    cropped_df["n_slices_cropped"] = cropped_df["n_slices_cropped"].astype("Int64")
    interp_df["n_slices_cropped"]  = interp_df["n_slices_cropped"].astype("Int64")

    os.makedirs("aec", exist_ok=True)
    with pd.ExcelWriter(AEC_CROPPED_PATH, engine="openpyxl") as writer:
        cropped_df.to_excel(writer, sheet_name="aec_cropped", index=False)
        interp_df.to_excel(writer, sheet_name=f"aec_{N_INTERP}", index=False)


def _write_outputs(results: list[dict]) -> None:
    _write_aec_total(results)
    _write_zbounds(results)
    _write_aec_cropped(results)


# ── 체크포인트 ────────────────────────────────────────────────────────────────

def _load_checkpoint() -> tuple[set[str], list[dict]]:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            data = pickle.load(f)
        results = data["results"]
        n_before = len(results)
        results = [r for r in results if not _desc_has_reformat_keyword(r.get("series_desc", ""))]
        n_removed = n_before - len(results)
        print(f"[체크포인트] {len(data['processed'])}명 완료 → 이어서 시작")
        if n_removed:
            print(f"[체크포인트] SeriesDescription에 재구성 키워드({', '.join(_REFORMAT_DESC_KEYWORDS)}) 포함된 "
                  f"{n_removed}건 제거")
        return data["processed"], results
    return set(), []


def _save_checkpoint(processed: set[str], results: list[dict]) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({"processed": processed, "results": results}, f)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(DICOM_BASE):
        raise FileNotFoundError(f"DICOM 경로 없음: {DICOM_BASE}")

    all_folders = [f for f in os.listdir(DICOM_BASE)
                   if os.path.isdir(os.path.join(DICOM_BASE, f))]
    all_folders.sort(key=lambda f: int(f.split("_")[0]) if f.split("_")[0].isdigit() else 0)
    if TEST_N is not None:
        all_folders = all_folders[:TEST_N]

    processed, results = _load_checkpoint()
    remaining = [f for f in all_folders if f not in processed]
    print(f"대상: {len(all_folders)}명  |  미처리: {len(remaining)}명")

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tqdm(total=len(all_folders), initial=len(processed), desc="Pubis~Liver AEC 추출") as pbar:
            for folder_name in remaining:
                patient_dir = os.path.join(DICOM_BASE, folder_name)
                pid = _patient_id_from_folder(folder_name)

                try:
                    series_list = find_axial_series(patient_dir)
                    if not series_list:
                        tqdm.write(f"  [SKIP] PID {pid}: axial 시리즈 없음")
                    for s in series_list:
                        r = extract_series(pid, s["series_uid"], s["series_desc"],
                                           s["manufacturer"], s["files"], tmp_dir)
                        results.append(r)
                        tqdm.write(f"  PID {pid} [{s['series_desc']}]: {r['seg_status']}")
                except Exception as e:
                    tqdm.write(f"  [ERROR] PID {pid}: {type(e).__name__}: {e}")

                processed.add(folder_name)
                pbar.update(1)

                if len(processed) % BATCH_SIZE == 0:
                    _save_checkpoint(processed, results)
                    _write_outputs(results)
                    tqdm.write(f"  [체크포인트 저장 | 수집 시리즈 {len(results)}건]")

    _write_outputs(results)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    n_ok = sum(1 for r in results if r.get("seg_status") == "ok")
    print(f"\n[완료] 총 {len(results)}개 시리즈 (ok: {n_ok})  |  aec")


if __name__ == "__main__":
    main()
