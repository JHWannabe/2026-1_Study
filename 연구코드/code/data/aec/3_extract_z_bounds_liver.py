import os, io, sys, re
import json, shutil, tempfile
from difflib import SequenceMatcher
import logging, contextlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import SimpleITK as sitk
import pydicom
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

SITE = "강남"

DICOM_BASE       = rf"D:/영상제공/{SITE}/{SITE}_axial"
OUT_PATH         = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\aec\{SITE}_z_bounds.xlsx"

AEC_TOTAL_PATH   = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\aec\{SITE}_aec_total.xlsx"
AEC_CROPPED_PATH = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\aec\{SITE}_aec_cropped.xlsx"

META_FILTER_PATH  = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\metadata\{SITE}_DLO_Results_SMI_kVp100.xlsx"
META_FILTER_SHEET = "kVp_100"

UPPER_DIR         = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\aec\liver_pubis\ok\upper"
BOTTOM_DIR        = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\aec\liver_pubis\ok\bottom"
MISSING_UPPER_DIR = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\aec\liver_pubis\missing\upper"
MISSING_BOTTOM_DIR= rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\aec\liver_pubis\missing\bottom"

ROI_SUBSET = ["liver", "hip_left", "hip_right"]

BATCH_SIZE       = 1
NUM_WORKERS      = 1
MIN_LIVER_VOXELS = 3000
N_INTERP         = 128

class _ThreadLocalStream:
    def __init__(self, real_stream):
        self._real  = real_stream
        self._local = threading.local()

    def _target(self):
        sink = getattr(self._local, "sink", None)
        return sink if sink is not None else self._real

    def write(self, data):       return self._target().write(data)
    def flush(self):             return self._target().flush()
    def __getattr__(self, name): return getattr(self._real, name)


_ts_stdout = _ThreadLocalStream(sys.stdout)
_ts_stderr = _ThreadLocalStream(sys.stderr)
sys.stdout = _ts_stdout  # type: ignore[assignment]
sys.stderr = _ts_stderr  # type: ignore[assignment]


@contextlib.contextmanager
def _silence():
    sink = io.StringIO()
    _ts_stdout._local.sink = sink
    _ts_stderr._local.sink = sink
    try:
        yield
    finally:
        _ts_stdout._local.sink = None
        _ts_stderr._local.sink = None


# ── DICOM 유틸 ────────────────────────────────────────────────────────────────

def _norm_series(s: str) -> str:
    """
    Series description 정규화 규칙:
      1. 소문자 통일          : Pre → pre, PRE → pre
      2. 공백 정규화          : 연속 공백 → 단일 공백
      3. '_' '/' → '_'        : 3/3mm → 3_3mm
      4. 괄호 주변 공백 제거  : iDose (3) → iDose(3)
    """
    s = s.lower()
    s = " ".join(s.split())
    s = re.sub(r"[_/]", "_", s)
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    return s


def _get_folder_dicom_meta(folder: str) -> tuple[str | None, str | None]:
    """Returns (SeriesDescription, Manufacturer) from the first readable DICOM in folder."""
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
    """
    폴더명 형식:
      ① {PatientID}              (예: 5253702)
      ② {순번}_{PatientID}_...   (예: 0001_5253702_20180201_CT)
    """
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


def get_instance_number(dcm_path: str) -> int | None:
    try:
        return int(pydicom.dcmread(dcm_path, stop_before_pixels=True).InstanceNumber)
    except Exception:
        return None


def read_z(dcm_path: str) -> float:
    try:
        return float(pydicom.dcmread(dcm_path, stop_before_pixels=True).ImagePositionPatient[2])
    except Exception:
        return float(np.nan)


# ── Segmentation ──────────────────────────────────────────────────────────────

def _load_liver_mask(seg_dir: str, warnings: list) -> np.ndarray | None:
    p = os.path.join(seg_dir, "liver.nii.gz")
    if not os.path.exists(p):
        warnings.append("liver: 분할 결과 없음 → liver_missing")
        return None
    try:
        vol = cast(Nifti1Image, nib.load(p)).get_fdata()
        if int((vol > 0.5).sum()) >= MIN_LIVER_VOXELS:
            return vol
        warnings.append("liver: 분할 결과 부족 → liver_missing")
    except Exception as e:
        warnings.append(f"liver: 로드 실패({type(e).__name__}: {e}) → liver_missing")
    return None


# ── PNG 유틸 ──────────────────────────────────────────────────────────────────

_SEG_COLORS = {
    "liver":     (1.0, 0.25, 0.25),
    "hip_left":  (0.2,  0.45, 1.0),
    "hip_right": (0.2,  0.75, 1.0),
}


def _read_dicom_pixels(dcm_path: str) -> np.ndarray:
    dcm    = pydicom.dcmread(dcm_path)
    pixels = dcm.pixel_array.astype(np.float32)
    pixels = pixels * float(getattr(dcm, "RescaleSlope",     1) or 1) \
                    + float(getattr(dcm, "RescaleIntercept", 0) or 0)
    wc = getattr(dcm, "WindowCenter", None)
    ww = getattr(dcm, "WindowWidth",  None)
    if wc is not None and ww is not None:
        wc = float(wc[0]) if hasattr(wc, "__iter__") and not isinstance(wc, str) else float(wc)
        ww = float(ww[0]) if hasattr(ww, "__iter__") and not isinstance(ww, str) else float(ww)
    else:
        lo, hi = np.percentile(pixels, [1, 99])
        wc, ww = (lo + hi) / 2, hi - lo
    low, high = wc - ww / 2, wc + ww / 2
    return ((np.clip(pixels, low, high) - low) / max(high - low, 1e-6) * 255).astype(np.uint8)


def save_slice_png(dcm_path: str, out_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(_read_dicom_pixels(dcm_path)).save(out_path)
        return True
    except Exception:
        return False


def save_slice_png_with_seg(dcm_path: str, seg_masks: dict, out_path: str) -> bool:
    try:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(_read_dicom_pixels(dcm_path), cmap="gray")
        patches = []
        for name, mask in seg_masks.items():
            if mask is None or mask.sum() == 0:
                continue
            color = _SEG_COLORS.get(name, (0.0, 1.0, 0.0))
            rgba  = np.zeros((*mask.shape, 4), dtype=np.float32)
            rgba[mask > 0.5] = [*color, 0.45]
            ax.imshow(rgba)
            patches.append(mpatches.Patch(color=color, label=name, alpha=0.8))
        if patches:
            ax.legend(handles=patches, loc="lower right", fontsize=7,
                      framealpha=0.6, edgecolor="white")
        ax.axis("off")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", dpi=150, pad_inches=0)
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False


def save_boundary_previews(pid_str: str, upper_dcm: str | None, bottom_dcm: str | None,
                           upper_masks: dict | None = None, bottom_masks: dict | None = None,
                           upper_dir: str = UPPER_DIR, bottom_dir: str = BOTTOM_DIR) -> None:
    def _save(dcm_path: str, masks: dict | None, out_dir: str):
        inst  = get_instance_number(dcm_path)
        fname = f"{pid_str}_{inst}.png" if inst is not None else f"{pid_str}.png"
        out   = os.path.join(out_dir, fname)
        if masks:
            if not save_slice_png_with_seg(dcm_path, masks, out):
                tqdm.write(f"    [PNG] overlay 실패 → 단순 저장: {fname}")
                save_slice_png(dcm_path, out)
        else:
            save_slice_png(dcm_path, out)

    if upper_dcm:  _save(upper_dcm,  upper_masks,  upper_dir)
    if bottom_dcm: _save(bottom_dcm, bottom_masks, bottom_dir)


# ── 핵심 추출 (환자 1명) ──────────────────────────────────────────────────────

def _empty_result() -> dict:
    return {
        "liver_upper_slice":  np.nan, "pubis_slice":        np.nan,
        "liver_upper_k":      np.nan, "pubis_k":            np.nan,
        "slices":             np.nan, "n_slices":           np.nan,
        "inst_min":           np.nan,
        "z_liver_upper":      np.nan, "z_pubis":            np.nan,
        "z_range":            np.nan, "seg_status":         "failed",
        "series_description": None,   "manufacturer_model": None,
        "aec_cropped":        None,
    }


def extract_bounds(pid_str: str, tmp_dir: str, folder_map: dict,
                   aec_df: pd.DataFrame | None = None,
                   expected_series: str | None = None,
                   expected_manufacturer: str | None = None) -> dict:
    result = _empty_result()
    warnings: list[str] = []

    if pid_str not in folder_map:
        result["seg_status"] = "no_dicom"
        return result

    subfolders = [s for s in os.listdir(folder_map[pid_str])
                  if os.path.isdir(os.path.join(folder_map[pid_str], s))]
    if not subfolders:
        result["seg_status"] = "no_series"
        return result

    if (expected_series is not None or expected_manufacturer is not None) and len(subfolders) > 1:
        matched   = None
        found_meta: list[tuple[str | None, str | None]] = []
        for sf in subfolders:
            sd, mf = _get_folder_dicom_meta(os.path.join(folder_map[pid_str], sf))
            found_meta.append((sd, mf))
            sd_ok = expected_series       is None or _norm_series(sd or "") == _norm_series(expected_series)
            mf_ok = expected_manufacturer is None or " ".join((mf or "").split()) == " ".join(expected_manufacturer.split())
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
                warnings.append(
                    f"series: 유사도 매칭 (ratio={best_ratio:.2f}) → '{best_sf}' "
                    f"(DICOM에서 읽힌 값: {found_meta})"
                )
        if matched is None:
            warnings.append(
                f"series: '{expected_series}' 매칭 실패 → 첫 번째 시리즈 사용 "
                f"(DICOM에서 읽힌 값: {found_meta})"
            )
        dcm_dir = os.path.join(folder_map[pid_str], matched if matched else subfolders[0])
    else:
        dcm_dir = os.path.join(folder_map[pid_str], subfolders[0])
    nii_path      = os.path.join(tmp_dir, f"{pid_str}.nii.gz")
    seg_dir = os.path.join(tmp_dir, f"{pid_str}_seg")

    try:
        # DICOM → NIfTI
        dcm_files = None
        try:
            reader     = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(dcm_dir)
            if series_ids:
                dicom_names = reader.GetGDCMSeriesFileNames(dcm_dir, series_ids[0])
                reader.SetFileNames(dicom_names)
                sitk.WriteImage(reader.Execute(), nii_path)
                dcm_files = list(dicom_names)
        except Exception:
            pass
        if not dcm_files:
            result["seg_status"] = "nifti_fail"
            return result

        try:
            hdr = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            result["series_description"] = str(hdr.SeriesDescription).replace("/", "_")
            result["manufacturer_model"] = str(hdr.ManufacturerModelName)
        except Exception:
            pass

        tqdm.write(f"  PID {pid_str}: series='{result['series_description'] or 'unknown'}'")

        # Segmentation (단일 호출)
        os.makedirs(seg_dir, exist_ok=True)
        with _silence():
            totalsegmentator(input=nii_path, output=seg_dir,
                             task="total", roi_subset=ROI_SUBSET,
                             fast=False, quiet=True, device=DEVICE)
        liver_vol = _load_liver_mask(seg_dir, warnings)

        n_k = int(cast(Nifti1Image, nib.load(nii_path)).shape[2])
        result["n_slices"] = n_k
        if n_k < 2 or len(dcm_files) != n_k:
            result["seg_status"] = "invalid_volume_match"
            return result

        # 엉덩뼈 k-index 수집
        hip_k: list[int] = []
        for side in ("hip_left", "hip_right"):
            p = os.path.join(seg_dir, f"{side}.nii.gz")
            if os.path.exists(p):
                data = cast(Nifti1Image, nib.load(p)).get_fdata()
                idx  = np.argwhere(data > 0.5)
                if len(idx) > 0:
                    hip_k.extend(idx[:, 2].astype(int).tolist())
        hip_k_arr = np.array(hip_k, dtype=int)
        hip_found = len(hip_k_arr) > 0

        # 간 마스크 정제 및 유효성 검증
        if liver_vol is not None:
            binary = liver_vol > 0.5
            labeled, n_comp = cast(tuple[np.ndarray, int], ndimage.label(binary))
            if n_comp > 1:
                sizes     = ndimage.sum(binary, labeled, list(range(1, n_comp + 1)))
                liver_vol = (labeled == int(np.argmax(sizes)) + 1).astype(np.float32)
            liver_k     = np.argwhere(liver_vol > 0.5)[:, 2].astype(int)
            # 높은 k = 머리 방향(고정 가정). 간 중앙이 전체 범위의 30% 이상(cranial)이어야 유효.
            liver_found = (len(liver_k) >= MIN_LIVER_VOXELS
                           and float(np.median(liver_k)) > n_k * 0.3)
            if not liver_found:
                warnings.append(f"liver: 해부학적 검증 실패 (voxels={len(liver_k)}) → liver_missing")
        else:
            liver_k     = np.array([], dtype=int)
            liver_found = False

        # 경계 슬라이스 결정 (높은 k = 머리 → liver max = cranial, hip min = 두덩뼈 하단)
        apex_k   = max(0, min(int(liver_k.max())   if liver_found else n_k - 1, n_k - 1))
        bottom_k = max(0, min(int(np.min(hip_k_arr)) if hip_found  else 0,       n_k - 1))
        upper_dcm  = dcm_files[apex_k]
        bottom_dcm = dcm_files[bottom_k]

        inst_nums = [n for f in dcm_files if (n := get_instance_number(f)) is not None]
        result["inst_min"] = int(min(inst_nums)) if inst_nums else np.nan

        result["z_liver_upper"]     = read_z(upper_dcm)
        result["liver_upper_slice"] = get_instance_number(upper_dcm) or np.nan
        result["z_pubis"]           = read_z(bottom_dcm)
        result["pubis_slice"]       = get_instance_number(bottom_dcm) or np.nan
        result["liver_upper_k"]     = apex_k
        result["pubis_k"]           = bottom_k
        result["slices"]            = abs(apex_k - bottom_k) + 1
        if pd.notna(result["z_liver_upper"]) and pd.notna(result["z_pubis"]):
            result["z_range"] = round(abs(result["z_liver_upper"] - result["z_pubis"]), 2)

        # AEC 크롭 (high k = cranial, aec_1 = cranial → aec_position = n_aec - k)
        if aec_df is not None:
            row = aec_df[aec_df["PatientID"] == int(pid_str)]
            if not row.empty:
                n_aec = int(row["n_slices"].iloc[0])
                vals  = []
                for i in range(n_aec):
                    col = f"aec_{i + 1}"
                    v   = row[col].iloc[0] if col in row.columns else float("nan")
                    vals.append(float(v) if pd.notna(v) else float("nan"))
                # k → aec 0-based 인덱스: n_aec - k - 1
                lo = max(0,         n_aec - max(apex_k, bottom_k) - 1)
                hi = min(n_aec - 1, n_aec - min(apex_k, bottom_k) - 1)
                result["aec_cropped"] = vals[lo : hi + 1]

        # seg_status 결정
        if liver_found and hip_found:
            result["seg_status"] = (
                "instance_number_missing"
                if pd.isna(result["liver_upper_slice"]) or pd.isna(result["pubis_slice"])
                else "ok"
            )
        elif liver_found:
            result["seg_status"] = "pubis_missing"
        elif hip_found:
            result["seg_status"] = "liver_missing"
        else:
            result["seg_status"] = "both_missing"

        # ok 전환 시 missing 폴더 PNG 정리
        if result["seg_status"] == "ok":
            for d in (MISSING_UPPER_DIR, MISSING_BOTTOM_DIR):
                if os.path.isdir(d):
                    for fname in os.listdir(d):
                        if fname.startswith(pid_str) and fname.endswith(".png"):
                            os.remove(os.path.join(d, fname))
                            tqdm.write(f"    [cleanup] 삭제: {os.path.join(d, fname)}")

        # PNG용 2D 슬라이스 추출 후 3D 볼륨 해제
        # liver_slice_2d = liver_vol[:, :, apex_k].T if liver_vol is not None and len(liver_k) > 0 else None
        del liver_vol
        # bottom_masks = {
        #     side: np.asarray(cast(Nifti1Image, nib.load(p)).dataobj[:, :, bottom_k]).T
        #     for side in ("hip_left", "hip_right")
        #     if os.path.exists(p := os.path.join(seg_dir, f"{side}.nii.gz"))
        # }
        # is_ok = result["seg_status"] == "ok"
        # save_boundary_previews(
        #     pid_str, upper_dcm, bottom_dcm,
        #     upper_masks  = {"liver": liver_slice_2d} if liver_slice_2d is not None else {},
        #     bottom_masks = bottom_masks,
        #     upper_dir    = UPPER_DIR  if is_ok else MISSING_UPPER_DIR,
        #     bottom_dir   = BOTTOM_DIR if is_ok else MISSING_BOTTOM_DIR,
        # )

    except Exception as e:
        warnings.append(f"[ERROR] {type(e).__name__}: {e}")
        result["seg_status"] = f"error:{type(e).__name__}:{e}"

    finally:
        for path in (nii_path, seg_dir):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    if warnings:
        tqdm.write(f"  PID {pid_str}: " + " | ".join(warnings))

    return result


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load_meta_filter() -> dict[str, tuple[str, str]] | None:
    """Returns {pid_str: (Series_Desc, Manufacturer)}"""
    if not os.path.exists(META_FILTER_PATH):
        print(f"[경고] 메타 필터 파일 없음: {META_FILTER_PATH}")
        return None
    df = pd.read_excel(META_FILTER_PATH, sheet_name=META_FILTER_SHEET)
    df["PatientID"] = df["PatientID"].astype(int).astype(str)
    result = {
        row["PatientID"]: (str(row["Series_Desc"]), str(row["Manufacturer"]))
        for _, row in df.iterrows()
    }
    print(f"메타 필터 로드: {len(result)}명  |  [{META_FILTER_SHEET}]  |  {META_FILTER_PATH}")
    return result


def _load_aec_df() -> pd.DataFrame | None:
    if not os.path.exists(AEC_TOTAL_PATH):
        print(f"[경고] AEC xlsx 없음 → 크롭 생략: {AEC_TOTAL_PATH}")
        return None
    df = pd.read_excel(AEC_TOTAL_PATH, sheet_name="aec_total")
    df["PatientID"] = df["PatientID"].astype(int)
    print(f"AEC xlsx 로드: {len(df)}명  |  {AEC_TOTAL_PATH}")
    return df


def _load_completed_from_xlsx(path: str = OUT_PATH) -> tuple[set, list]:
    if not os.path.exists(path):
        return set(), []
    try:
        df = pd.read_excel(path)
        if "seg_status" not in df.columns or "PatientID" not in df.columns:
            return set(), []
        # 이전에 처리된 환자(성공/실패 무관)는 모두 스킵하여 기존 데이터 보존
        return set(df["PatientID"].astype(int).astype(str)), df.to_dict("records")
    except Exception as e:
        print(f"[경고] xlsx 읽기 실패 ({e}). 처음부터 시작합니다.")
        return set(), []


def _write_output(rows: list):
    clean_rows = [{k: v for k, v in r.items() if k != "aec_cropped"} for r in rows]
    df = pd.DataFrame(clean_rows).drop_duplicates(subset=["PatientID"], keep="last")
    df = df.sort_values("PatientID").reset_index(drop=True)
    for col in ["liver_upper_slice", "pubis_slice", "slices", "n_slices", "inst_min"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    col_order = ["PatientID", "series_description", "manufacturer_model", "n_slices", "inst_min", "slices",
                 "liver_upper_slice", "pubis_slice", "liver_upper_k", "pubis_k",
                 "z_range", "z_liver_upper", "z_pubis", "seg_status"]
    cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df[cols].to_excel(OUT_PATH, index=False)


def _interp_aec(vals: np.ndarray, n: int) -> np.ndarray:
    x_orig = np.linspace(0, 1, len(vals))
    x_new  = np.linspace(0, 1, n)
    return np.interp(x_new, x_orig, vals)


def _write_cropped_aec(rows: list):
    cropped_rows: list[dict] = []
    for r in rows:
        aec_vals = r.get("aec_cropped")
        if not aec_vals:
            continue
        crow: dict = {
            "PatientID":        r["PatientID"],
            "series_description": r.get("series_description"),
            "manufacturer_model": r.get("manufacturer_model"),
            "n_slices_cropped": len(aec_vals),
            "z_range":          r.get("z_range", np.nan),
        }
        for i, v in enumerate(reversed(aec_vals)):
            crow[f"aec_{i + 1}"] = v
        cropped_rows.append(crow)
    if not cropped_rows:
        return

    max_len   = max(r["n_slices_cropped"] for r in cropped_rows)
    meta_cols = ["PatientID", "series_description", "manufacturer_model", "n_slices_cropped", "z_range"]
    aec_cols  = [f"aec_{i + 1}" for i in range(max_len)]

    new_cropped = pd.DataFrame(cropped_rows, columns=meta_cols + aec_cols)
    new_cropped["n_slices_cropped"] = new_cropped["n_slices_cropped"].astype("Int64")

    # 환자별 aec_128 보간
    interp_rows = []
    for _, row in new_cropped.iterrows():
        n    = int(row["n_slices_cropped"])
        raw  = row[aec_cols[:n]].values.astype(float)
        vals = raw[~np.isnan(raw)]
        interped = _interp_aec(vals, N_INTERP) if len(vals) >= 2 else np.full(N_INTERP, np.nan)
        interp_rows.append(
            {k: row[k] for k in meta_cols}
            | {f"aec_{i + 1}": round(float(v), 2) for i, v in enumerate(interped)}
        )
    new_interp = pd.DataFrame(interp_rows,
                              columns=meta_cols + [f"aec_{i + 1}" for i in range(N_INTERP)])
    new_interp["n_slices_cropped"] = new_interp["n_slices_cropped"].astype("Int64")

    # 기존 데이터 로드 후 신규 환자만 추가
    if os.path.exists(AEC_CROPPED_PATH):
        try:
            old_cropped = pd.read_excel(AEC_CROPPED_PATH, sheet_name="aec_cropped")
            old_interp  = pd.read_excel(AEC_CROPPED_PATH, sheet_name=f"aec_{N_INTERP}")
            existing_pids = set(old_cropped["PatientID"])
            add_cropped = new_cropped[~new_cropped["PatientID"].isin(existing_pids)]
            add_interp  = new_interp[~new_interp["PatientID"].isin(existing_pids)]
            cropped_df = pd.concat([old_cropped, add_cropped], ignore_index=True)
            interp_df  = pd.concat([old_interp,  add_interp],  ignore_index=True)
        except Exception:
            cropped_df, interp_df = new_cropped, new_interp
    else:
        cropped_df, interp_df = new_cropped, new_interp

    cropped_df = cropped_df.sort_values("PatientID").reset_index(drop=True)
    interp_df  = interp_df.sort_values("PatientID").reset_index(drop=True)

    os.makedirs(os.path.dirname(AEC_CROPPED_PATH), exist_ok=True)
    with pd.ExcelWriter(AEC_CROPPED_PATH, engine="openpyxl") as writer:
        cropped_df.to_excel(writer, sheet_name="aec_cropped", index=False)
        interp_df.to_excel(writer, sheet_name=f"aec_{N_INTERP}", index=False)


def crop_aec_from_zbounds(
    zbounds_path:   str = OUT_PATH,
    aec_total_path: str = AEC_TOTAL_PATH,
    out_path:       str = AEC_CROPPED_PATH,
    n_interp:       int = N_INTERP,
) -> None:
    """
    기존 z_bounds.xlsx(TotalSegmentator 결과)와 aec_total.xlsx를 사용해
    aec 데이터를 [liver upper ~ pubis] 구간으로 크롭하고 aec_cropped.xlsx를 저장한다.

    k-index 기반 크롭 (high k = cranial, aec_1 = cranial):
      aec 0-based index = n_aec - k - 1
      liver_upper_k (apex_k, 높은 k) → 낮은 aec 위치
      pubis_k       (bottom_k, 낮은 k) → 높은 aec 위치
    """
    if not os.path.exists(zbounds_path):
        print(f"[경고] z_bounds 파일 없음: {zbounds_path}")
        return
    if not os.path.exists(aec_total_path):
        print(f"[경고] aec_total 파일 없음: {aec_total_path}")
        return

    zdf = pd.read_excel(zbounds_path)
    adf = pd.read_excel(aec_total_path, sheet_name="aec_total")
    adf["PatientID"] = adf["PatientID"].astype(int)

    ok_df   = zdf[zdf["seg_status"].astype(str) == "ok"].copy()
    has_k   = "liver_upper_k" in ok_df.columns and "pubis_k" in ok_df.columns
    print(f"z_bounds 로드: {len(ok_df)}명(ok) / {len(zdf)}명  |  k-index 컬럼: {has_k}")

    cropped_rows: list[dict] = []
    interp_rows:  list[dict] = []

    for _, zrow in ok_df.iterrows():
        pid   = int(zrow["PatientID"])
        arow  = adf[adf["PatientID"] == pid]
        if arow.empty:
            continue

        n_aec = int(arow["n_slices"].iloc[0])
        vals  = [
            float(arow[f"aec_{i+1}"].iloc[0]) if f"aec_{i+1}" in arow.columns else float("nan")
            for i in range(n_aec)
        ]

        if has_k and pd.notna(zrow["liver_upper_k"]) and pd.notna(zrow["pubis_k"]):
            lk  = int(zrow["liver_upper_k"])   # high k = cranial
            pk  = int(zrow["pubis_k"])          # low  k = caudal
            lo  = max(0,         n_aec - max(lk, pk) - 1)
            hi  = min(n_aec - 1, n_aec - min(lk, pk) - 1)
        else:
            # fallback: k-index 없는 구버전 z_bounds — 슬라이스 위치를 n_slices에서 역산
            n_z = int(zrow["n_slices"]) if pd.notna(zrow.get("n_slices")) else n_aec
            lu  = int(zrow["liver_upper_slice"]) if pd.notna(zrow["liver_upper_slice"]) else 1
            pb  = int(zrow["pubis_slice"])        if pd.notna(zrow["pubis_slice"])       else n_z
            # aec_position(1-based from cranial) → 0-based index = aec_position - 1
            lo  = max(0,         min(lu, pb) - 1)
            hi  = min(n_aec - 1, max(lu, pb) - 1)

        cropped = vals[lo : hi + 1]
        z_range      = zrow.get("z_range", np.nan)
        series_desc  = zrow.get("series_description")
        manufacturer = zrow.get("manufacturer_model")

        crow: dict = {
            "PatientID":          pid,
            "series_description": series_desc,
            "manufacturer_model": manufacturer,
            "n_slices_cropped":   len(cropped),
            "z_range":            z_range,
        }
        for i, v in enumerate(reversed(cropped)):
            crow[f"aec_{i+1}"] = v
        cropped_rows.append(crow)

        raw      = np.array([v for v in reversed(cropped) if not np.isnan(v)], dtype=float)
        interped = _interp_aec(raw, n_interp) if len(raw) >= 2 else np.full(n_interp, np.nan)
        interp_rows.append(
            {"PatientID": pid, "series_description": series_desc,
             "manufacturer_model": manufacturer, "n_slices_cropped": len(cropped), "z_range": z_range}
            | {f"aec_{i+1}": round(float(v), 2) for i, v in enumerate(interped)}
        )

    if not cropped_rows:
        print("[경고] 크롭된 데이터 없음")
        return

    max_len   = max(r["n_slices_cropped"] for r in cropped_rows)
    meta_cols = ["PatientID", "series_description", "manufacturer_model", "n_slices_cropped", "z_range"]

    cropped_df = pd.DataFrame(cropped_rows, columns=meta_cols + [f"aec_{i+1}" for i in range(max_len)])
    cropped_df = cropped_df.sort_values("PatientID").reset_index(drop=True)
    cropped_df["n_slices_cropped"] = cropped_df["n_slices_cropped"].astype("Int64")

    interp_df = pd.DataFrame(interp_rows, columns=meta_cols + [f"aec_{i+1}" for i in range(n_interp)])
    interp_df = interp_df.sort_values("PatientID").reset_index(drop=True)
    interp_df["n_slices_cropped"] = interp_df["n_slices_cropped"].astype("Int64")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        cropped_df.to_excel(writer, sheet_name="aec_cropped", index=False)
        interp_df.to_excel(writer, sheet_name=f"aec_{n_interp}", index=False)
    print(f"[완료] aec 크롭 완료: {len(cropped_rows)}명  →  {out_path}")


def _print_summary(label: str):
    df = pd.read_excel(OUT_PATH)
    print(f"\n[{label}] {OUT_PATH}")
    print(f"  shape : {df.shape}")
    print(f"  상태  :\n{df['seg_status'].value_counts().to_string()}")
    for col in ("liver_upper_slice", "pubis_slice"):
        if col in df.columns and not df[col].isna().all():
            print(f"  {col}: median={df[col].median():.0f}  std={df[col].std():.1f}")
    _cnt = lambda d: len([f for f in os.listdir(d) if f.endswith(".png")]) if os.path.isdir(d) else 0
    print(f"  PNG   : ok/upper={_cnt(UPPER_DIR)}, ok/bottom={_cnt(BOTTOM_DIR)}, "
          f"missing/upper={_cnt(MISSING_UPPER_DIR)}, missing/bottom={_cnt(MISSING_BOTTOM_DIR)}")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    # for d in (UPPER_DIR, BOTTOM_DIR, MISSING_UPPER_DIR, MISSING_BOTTOM_DIR):
    #     os.makedirs(d, exist_ok=True)

    folder_map = build_folder_map(DICOM_BASE)
    aec_df     = _load_aec_df()

    if aec_df is not None:
        aec_pids = set(aec_df["PatientID"].astype(str))
        all_pids = sorted([pid for pid in folder_map if pid in aec_pids], key=int)
        print(f"AEC 기준 환자: {len(all_pids)}명 (DICOM 존재)")
    else:
        all_pids = sorted(folder_map.keys(), key=int)

    if os.path.exists(AEC_TOTAL_PATH):
        filter_ids = set(pd.read_excel(AEC_TOTAL_PATH)["PatientID"].astype(int).astype(str))
        all_pids   = [pid for pid in all_pids if pid in filter_ids]
        print(f"AEC 필터 적용: {len(filter_ids)}명 중 DICOM 존재 {len(all_pids)}명")

    meta_filter = _load_meta_filter()
    if meta_filter is not None:
        all_pids = [pid for pid in all_pids if pid in meta_filter]
        print(f"메타 필터 적용: {len(all_pids)}명")

    total          = len(all_pids)
    completed_pids, rows = _load_completed_from_xlsx()
    n_done         = len(completed_pids)
    pending_pids   = [pid for pid in all_pids if pid not in completed_pids]
    rows           = rows or []
    print(f"대상: {total}명  |  완료: {n_done}명  |  처리 예정: {len(pending_pids)}명  |  스레드: {NUM_WORKERS}")

    lock        = threading.Lock()
    n_processed = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tqdm(total=total, desc="경계 추출", initial=n_done) as pbar:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                future_to_pid = {
                    executor.submit(
                        extract_bounds, pid, tmp_dir, folder_map, aec_df,
                        *(meta_filter[pid] if meta_filter and pid in meta_filter else (None, None))
                    ): pid
                    for pid in pending_pids
                }
                for future in as_completed(future_to_pid):
                    pid = future_to_pid[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        tqdm.write(f"  [ERROR] PID {pid}: {type(e).__name__}: {e}")
                        result = _empty_result()
                        result["seg_status"] = f"error:{type(e).__name__}:{e}"
                    result["PatientID"] = int(pid)

                    with lock:
                        rows.append(result)
                        if result.get("seg_status") == "ok":
                            completed_pids.add(pid)
                        pbar.update(1)
                        n_processed += 1

                        if n_processed % BATCH_SIZE == 0 or n_processed == len(pending_pids):
                            _write_output(rows)
                            _write_cropped_aec(rows)
                            counts = pd.Series([r["seg_status"] for r in rows]).value_counts().to_dict()
                            tqdm.write(f"  [{n_done + n_processed}/{total}] {counts}")

    _print_summary("완료")

    # z_bounds.xlsx 기반으로 전체 환자 aec 크롭 (재실행 시에도 완전히 동작)
    crop_aec_from_zbounds()


if __name__ == "__main__":
    import time
    # print("강남세브란스 병원 DICOM 경계 추출 및 AEC 크롭 시작")
    # print("시작시간:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # print("2시간 뒤 실행")
    # time.sleep(7200) # 2시간 대기
    
    main()
