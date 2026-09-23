"""
AEC crop(liver~pubis) 구간이 아니라 스캔 전체 range에서 TotalSegmentator로
11개 해부학적 anchor(아래→위)의 존재 여부와 좌표를 확인하는 QC 스크립트.

Core anchor (아래→위, 11개):
  1. Inferior pubic margin (hip_left/right 최하단, caudal)
  2. S1 center
  3. L5 center
  4. L4 center
  5. L3 center
  6. L2 center
  7. L1 center
  8. T12 center
  9. T11 center
  10. T10 center
  11. Liver dome (liver 최상단, cranial)

추가 후보 anchor:
  - Femoral head center (좌우 femur 상단부 centroid 평균, pubis~S1 사이 위치)
  - T9 center, T10 위
  - T8 center

각 anchor는 center(site)×scanner(manufacturer_model)별로 ≥95% 완전 포함될 때만
core 채택 후보로 표시한다(seg_summarize_qc). 고관절 인공관절 의심(HU 임계값 초과) /
척추 numbering anomaly(순서 역전)는 QC 대상으로 별도 플래그된다.

k축은 extract_liver_pubis_aec.py와 동일 관례를 따른다: 값이 클수록 cranial(머리 쪽),
값이 작을수록 caudal(발 쪽) — 기존 파이프라인에서 liver_upper_k=max, pubis_k=min으로
이미 검증된 방향과 동일하다.

출력:
  - {site}_landmarks.xlsx : 환자별 각 anchor의 k-index / Instance Number / z좌표 / status
  - landmark_qc_summary.xlsx : site×scanner×anchor 완전 포함률 + core 채택 여부(2개 시트)
  - {site}/aec/landmark_preview/{PatientID}/{anchor}_inst{N}.png : 랜드마크 육안 검증용 무작위 샘플 PNG
"""

import os
import pickle
import re
import shutil
import tempfile
from typing import cast

import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
from nibabel.nifti1 import Nifti1Image
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from extract_liver_pubis_aec import (
    DATA_DIR, SITES, DEVICE, MIN_LIVER_VOXELS,
    _silence, find_series_folder, read_slices,
)
from totalsegmentator.python_api import totalsegmentator

# ── 설정 ──────────────────────────────────────────────────────────────────────

BATCH_SIZE      = 5
MIN_VOXELS      = 50     # 척추/femur 라벨의 최소 유효 복셀 수(너무 작으면 미검출 취급)
HIP_HARDWARE_HU = 2500   # 고관절 인공관절(금속) 의심 HU 임계값
MIN_GROUP_N     = 10     # QC 완전포함률 계산 시 최소 표본 수(그 미만 그룹은 core 판정에서 제외)

# 무작위 샘플 PNG 미리보기 설정 (--preview-only 대신 여기서 직접 값을 바꿔서 사용)
PREVIEW_ONLY  = False    # True면 세그멘테이션 없이 기존 {site}_landmarks.xlsx로 미리보기만 저장하고 종료
PREVIEW_N     = 5        # 사이트당 무작위 샘플 환자 수
PREVIEW_SEED  = None     # 샘플 재현용 랜덤 시드 (None이면 매번 랜덤)

VERTEBRAE = ["vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
             "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
             "vertebrae_S1"]
ROI_SUBSET = VERTEBRAE + ["liver", "hip_left", "hip_right", "femur_left", "femur_right"]

# 아래(caudal, k 작음) → 위(cranial, k 큼) 순서. vertebra_order_anomaly 판정에도 사용.
ORDERED_VERTEBRAE = ["vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
                     "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
                     "vertebrae_S1"][::-1]  # T8(가장 cranial) ... S1(가장 caudal)

CORE_ANCHORS = ["inferior_pubic_margin", "S1_center", "L5_center", "L4_center", "L3_center",
                "L2_center", "L1_center", "T12_center", "T11_center", "T10_center", "liver_dome"]
OPTIONAL_ANCHORS = ["femoral_head_center", "T9_center", "T8_center"]
# 아래(caudal)→위(cranial) 해부학적 순서로 컬럼 배열(엑셀 가독성). z값 기준 liver_dome은 T9~T8 사이에 위치.
ALL_ANCHORS = ["inferior_pubic_margin", "femoral_head_center", "S1_center", "L5_center", "L4_center",
               "L3_center", "L2_center", "L1_center", "T12_center", "T11_center", "T10_center",
               "T9_center", "liver_dome", "T8_center"]


def site_paths(site: str, shard: int | None = None) -> dict:
    suffix = f".shard{shard}" if shard is not None else ""
    return {
        "dicom_base": rf"E:\영상제공\{site}\{site}_axial",
        "dlo_path":   rf"{DATA_DIR}\{site}\metadata\{site}_DLO_Results.xlsx",
        "landmarks":  rf"{DATA_DIR}\{site}\aec\{site}_landmarks{suffix}.xlsx",
        "checkpoint": rf"{DATA_DIR}\{site}\aec\{site}_landmarks_checkpoint{suffix}.pkl",
        "main_landmarks": rf"{DATA_DIR}\{site}\aec\{site}_landmarks.xlsx",
    }


# ── 라벨 마스크 → k-index ────────────────────────────────────────────────────

def _largest_component_k(mask_path: str, min_voxels: int = MIN_VOXELS) -> tuple[np.ndarray | None, int, int]:
    """라벨 마스크에서 가장 큰 연결 성분의 k-index 배열, 복셀 수, 연결 성분 개수를 반환.
    없거나 너무 작으면 (None, 0, 0)."""
    if not os.path.exists(mask_path):
        return None, 0, 0
    try:
        vol = cast(Nifti1Image, nib.load(mask_path)).get_fdata()
    except Exception:
        return None, 0, 0
    binary = vol > 0.5
    if not binary.any():
        return None, 0, 0
    labeled, n_comp = cast(tuple[np.ndarray, int], ndimage.label(binary))
    if n_comp > 1:
        sizes = ndimage.sum(binary, labeled, list(range(1, n_comp + 1)))
        binary = labeled == int(np.argmax(sizes)) + 1
    k_idx = np.argwhere(binary)[:, 2].astype(int)
    if len(k_idx) < min_voxels:
        return None, len(k_idx), n_comp
    return k_idx, len(k_idx), n_comp


def _femoral_head_k(seg_dir: str) -> tuple[int | None, int]:
    """femur_left/right 마스크 중 가장 cranial(k 큰) 쪽 상위 25% 구간을 femoral head로 근사,
    좌우 centroid k의 평균을 반환."""
    centers, total_vox = [], 0
    for side in ("femur_left", "femur_right"):
        k_arr, n_vox, _ = _largest_component_k(os.path.join(seg_dir, f"{side}.nii.gz"))
        if k_arr is None:
            continue
        k_min, k_max = int(k_arr.min()), int(k_arr.max())
        head_thresh = k_max - 0.25 * (k_max - k_min)
        head_k = k_arr[k_arr >= head_thresh]
        if len(head_k) == 0:
            continue
        centers.append(float(np.mean(head_k)))
        total_vox += n_vox
    if not centers:
        return None, 0
    return int(round(sum(centers) / len(centers))), total_vox


# ── 환자 1명 처리 ─────────────────────────────────────────────────────────────

def _empty_landmark_row(pid: int, series_desc, manufacturer, status: str) -> dict:
    row = {"PatientID": pid, "series_description": series_desc, "manufacturer_model": manufacturer,
           "n_slices": np.nan, "seg_status": status,
           "vertebra_order_anomaly": np.nan, "hip_hardware_suspected": np.nan}
    for anchor in ALL_ANCHORS:
        row[f"{anchor}_status"] = "missing"
        row[f"{anchor}_k"] = np.nan
        row[f"{anchor}_instance"] = np.nan
        row[f"{anchor}_z"] = np.nan
    return row


def compute_landmarks(row: dict, seg_dir: str, nii_path: str, n_k: int, k_to_slice: list) -> None:
    """seg_dir(TotalSegmentator 출력)에서 anchor들을 찾아 row에 채운다.
    k_to_slice[k]는 {"inst": ..., "z": ...}를 가진 dict(또는 None)여야 한다.
    사이트 기반(check_landmarks.py)/new10000 코호트 양쪽에서 공용으로 쓴다."""

    def _set_anchor(name: str, k_val: int | None):
        if k_val is None:
            return
        k_val = max(0, min(k_val, n_k - 1))
        slice_row = k_to_slice[k_val]
        row[f"{name}_status"] = "ok"
        row[f"{name}_k"] = k_val
        row[f"{name}_instance"] = slice_row["inst"] if slice_row else np.nan
        row[f"{name}_z"] = slice_row["z"] if slice_row else np.nan

    # 척추 anchor (S1/L1~L5/T10~T12/T8~T9)
    vertebra_centers: dict[str, int] = {}
    for label in VERTEBRAE:
        name = label.replace("vertebrae_", "") + "_center"
        k_arr, _, _ = _largest_component_k(os.path.join(seg_dir, f"{label}.nii.gz"))
        if k_arr is not None:
            centroid_k = int(round(float(np.mean(k_arr))))
            vertebra_centers[label] = max(0, min(centroid_k, n_k - 1))
            _set_anchor(name, vertebra_centers[label])

    # 척추 numbering anomaly: T8(가장 cranial) -> S1(가장 caudal) 순으로 k가 단조 감소해야 함
    present = [(lab, vertebra_centers[lab]) for lab in ORDERED_VERTEBRAE if lab in vertebra_centers]
    row["vertebra_order_anomaly"] = any(
        present[i][1] <= present[i + 1][1] for i in range(len(present) - 1)
    ) if len(present) >= 2 else False

    # Liver dome: liver 최상단(cranial 끝, k 최대)
    liver_k_arr, _, _ = _largest_component_k(os.path.join(seg_dir, "liver.nii.gz"),
                                              min_voxels=MIN_LIVER_VOXELS)
    _set_anchor("liver_dome", int(liver_k_arr.max()) if liver_k_arr is not None else None)

    # Inferior pubic margin: hip_left/right 최하단(caudal 끝, k 최소)
    hip_k_all = []
    for side in ("hip_left", "hip_right"):
        k_arr, _, _ = _largest_component_k(os.path.join(seg_dir, f"{side}.nii.gz"))
        if k_arr is not None:
            hip_k_all.append(k_arr)
    _set_anchor("inferior_pubic_margin",
                int(np.concatenate(hip_k_all).min()) if hip_k_all else None)

    # Femoral head center (좌우 평균, 선택 anchor)
    head_k, _ = _femoral_head_k(seg_dir)
    _set_anchor("femoral_head_center", head_k)

    # 고관절 인공관절 의심: hip 마스크 내 최대 HU가 임계값 초과
    if hip_k_all:
        hu_vol = cast(Nifti1Image, nib.load(nii_path)).get_fdata()
        hip_mask = np.zeros(hu_vol.shape, dtype=bool)
        for side in ("hip_left", "hip_right"):
            p = os.path.join(seg_dir, f"{side}.nii.gz")
            if os.path.exists(p):
                hip_mask |= cast(Nifti1Image, nib.load(p)).get_fdata() > 0.5
        if hip_mask.any():
            row["hip_hardware_suspected"] = bool(np.nanmax(hu_vol[hip_mask]) > HIP_HARDWARE_HU)
        else:
            row["hip_hardware_suspected"] = False
    else:
        row["hip_hardware_suspected"] = False

    found_core = sum(1 for a in CORE_ANCHORS if row[f"{a}_status"] == "ok")
    row["seg_status"] = "ok" if found_core == len(CORE_ANCHORS) else (
        "partial" if found_core > 0 else "both_missing")


def process_patient(pid: int, series_dir: str, tmp_dir: str) -> dict:
    try:
        hdr_file = next(os.path.join(series_dir, f) for f in os.listdir(series_dir)
                         if os.path.isfile(os.path.join(series_dir, f)))
        hdr = pydicom.dcmread(hdr_file, stop_before_pixels=True)
        series_desc  = str(getattr(hdr, "SeriesDescription", ""))
        manufacturer = str(getattr(hdr, "ManufacturerModelName", ""))
    except StopIteration:
        return _empty_landmark_row(pid, None, None, "no_series")
    except Exception:
        series_desc = manufacturer = None

    by_inst = read_slices(series_dir)
    if by_inst is None:
        return _empty_landmark_row(pid, series_desc, manufacturer, "no_position_data")

    n = len(by_inst)
    row = _empty_landmark_row(pid, series_desc, manufacturer, "failed")
    row["n_slices"] = n

    nii_path = os.path.join(tmp_dir, f"{pid}.nii.gz")
    seg_dir  = os.path.join(tmp_dir, f"{pid}_seg")
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(series_dir)
        if not series_ids:
            row["seg_status"] = "nifti_fail"
            return row
        k_files = list(reader.GetGDCMSeriesFileNames(series_dir, series_ids[0]))
        if len(k_files) != n:
            row["seg_status"] = "invalid_volume_match"
            return row

        reader.SetFileNames(k_files)
        img = reader.Execute()
        sitk.WriteImage(img, nii_path)

        os.makedirs(seg_dir, exist_ok=True)
        with _silence():
            totalsegmentator(input=nii_path, output=seg_dir, task="total",
                             roi_subset=ROI_SUBSET, fast=False, quiet=True, device=DEVICE)

        n_k = int(cast(Nifti1Image, nib.load(nii_path)).shape[2])
        if n_k != n:
            row["seg_status"] = "invalid_volume_match"
            return row

        basename_to_slice = {os.path.basename(r["file"]): r for r in by_inst}
        k_to_slice = [basename_to_slice.get(os.path.basename(f)) for f in k_files]

        compute_landmarks(row, seg_dir, nii_path, n_k, k_to_slice)

    except Exception as e:
        row["seg_status"] = f"error:{type(e).__name__}:{e}"
    finally:
        if os.path.isfile(nii_path):
            os.remove(nii_path)
        if os.path.isdir(seg_dir):
            shutil.rmtree(seg_dir, ignore_errors=True)

    return row


# ── 출력 (기존 파일 보존 + 신규 행만 append) ────────────────────────────────────

def _clean_ws(s):
    return re.sub(r"\s+", " ", str(s)).strip() if pd.notna(s) else s


def _write_landmarks(paths: dict, new_rows: list[dict]) -> None:
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    old_df = pd.read_excel(paths["landmarks"]) if os.path.exists(paths["landmarks"]) else None
    combined = pd.concat([old_df, new_df], ignore_index=True) if old_df is not None else new_df
    combined = combined.drop_duplicates(subset=["PatientID"], keep="first")
    combined["series_description"] = combined["series_description"].map(_clean_ws)
    combined = combined.sort_values("PatientID").reset_index(drop=True)
    os.makedirs(os.path.dirname(paths["landmarks"]), exist_ok=True)
    combined.to_excel(paths["landmarks"], index=False)


# ── 체크포인트 ────────────────────────────────────────────────────────────────

def _load_checkpoint(path: str) -> set[int]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return set()


def _save_checkpoint(path: str, attempted: set[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(attempted, f)


def _shard_filter(pid: int, shard_id: int, num_shards: int) -> bool:
    return num_shards <= 1 or (pid % num_shards) == shard_id


# ── 사이트 실행 ───────────────────────────────────────────────────────────────

def run_site(site: str, shard_id: int = 0, num_shards: int = 1):
    sharded = num_shards > 1
    paths = site_paths(site, shard=shard_id if sharded else None)
    tag = f"{site}#{shard_id}" if sharded else site
    print("=" * 78)
    print(f"[{tag}] 시작 (전체 range 랜드마크 QC)  |  DICOM: {paths['dicom_base']}")

    dlo = pd.read_excel(paths["dlo_path"])[["PatientID", "Series_Desc"]]
    dlo["PatientID"] = dlo["PatientID"].astype(int)

    check_paths = site_paths(site)
    done_pids = set()
    if os.path.exists(check_paths["landmarks"]):
        done_pids |= set(pd.read_excel(check_paths["landmarks"])["PatientID"].astype(int))
    if sharded and os.path.exists(paths["landmarks"]):
        done_pids |= set(pd.read_excel(paths["landmarks"])["PatientID"].astype(int))
    attempted = _load_checkpoint(paths["checkpoint"])
    done_pids |= attempted

    pending = dlo[~dlo["PatientID"].isin(done_pids)]
    if sharded:
        pending = pending[pending["PatientID"].apply(lambda p: _shard_filter(p, shard_id, num_shards))]
    print(f"[{tag}] 대상 {len(dlo)}명  |  완료 {len(done_pids)}명  |  처리 예정(이 shard) {len(pending)}명")

    rows: list[dict] = []
    n_processed = 0
    if not pending.empty:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for _, r in tqdm(pending.iterrows(), total=len(pending), desc=f"{tag} 처리"):
                pid = int(r["PatientID"])
                patient_dir = os.path.join(paths["dicom_base"], str(pid))
                if not os.path.isdir(patient_dir):
                    rows.append(_empty_landmark_row(pid, None, None, "no_dicom"))
                else:
                    folder = find_series_folder(patient_dir, r["Series_Desc"])
                    if folder is None:
                        rows.append(_empty_landmark_row(pid, None, None, "no_series"))
                    else:
                        series_dir = os.path.join(patient_dir, folder)
                        try:
                            row = process_patient(pid, series_dir, tmp_dir)
                        except Exception as e:
                            row = _empty_landmark_row(pid, None, None, f"error:{type(e).__name__}:{e}")
                        rows.append(row)
                        tqdm.write(f"  PID {pid}: {row['seg_status']}")

                attempted.add(pid)
                n_processed += 1
                if n_processed % BATCH_SIZE == 0 or n_processed == len(pending):
                    _write_landmarks(paths, rows)
                    _save_checkpoint(paths["checkpoint"], attempted)
                    rows = []
                    tqdm.write(f"  [{tag} 체크포인트 저장 | {n_processed}/{len(pending)}]")

    if os.path.exists(paths["checkpoint"]):
        os.remove(paths["checkpoint"])
    print(f"[{tag}] 완료")


def merge_shards(site: str, num_shards: int, delete_shard_files: bool = False):
    main_paths = site_paths(site)
    for shard_id in range(num_shards):
        sp = site_paths(site, shard=shard_id)
        if os.path.exists(sp["landmarks"]):
            df = pd.read_excel(sp["landmarks"])
            _write_landmarks(main_paths, df.to_dict("records"))
        print(f"[{site}] shard {shard_id} 병합 완료")
        if delete_shard_files:
            for key in ("landmarks", "checkpoint"):
                if os.path.exists(sp[key]):
                    os.remove(sp[key])
            print(f"[{site}] shard {shard_id} 임시 파일 삭제 완료")


# ── QC 요약: site x scanner x anchor 완전 포함률 ────────────────────────────────

def summarize_qc(sites: list[str], out_path: str) -> None:
    frames = []
    for site in sites:
        p = site_paths(site)
        if not os.path.exists(p["landmarks"]):
            print(f"[{site}] {p['landmarks']} 없음 — QC 요약에서 제외")
            continue
        df = pd.read_excel(p["landmarks"])
        df["site"] = site
        frames.append(df)
    if not frames:
        print("QC 요약 대상 없음")
        return
    all_df = pd.concat(frames, ignore_index=True)

    summary_rows = []
    for (site, mfr), g in all_df.groupby(["site", "manufacturer_model"], dropna=False):
        n_total = len(g)
        qc_mask = (g["vertebra_order_anomaly"].fillna(False).astype(bool)
                   | g["hip_hardware_suspected"].fillna(False).astype(bool))
        g_clean = g[~qc_mask]
        for anchor in ALL_ANCHORS:
            status_col = f"{anchor}_status"
            if status_col not in g.columns:
                continue
            n_found = int((g[status_col] == "ok").sum())
            n_total_clean = len(g_clean)
            n_found_clean = int((g_clean[status_col] == "ok").sum()) if n_total_clean else 0
            summary_rows.append({
                "site": site, "manufacturer_model": mfr, "anchor": anchor,
                "n_total": n_total, "n_found": n_found,
                "pct_found": round(100 * n_found / n_total, 2) if n_total else np.nan,
                "n_total_excl_qc": n_total_clean, "n_found_excl_qc": n_found_clean,
                "pct_found_excl_qc": round(100 * n_found_clean / n_total_clean, 2) if n_total_clean else np.nan,
            })
    summary = pd.DataFrame(summary_rows)

    core_rows = []
    for anchor in ALL_ANCHORS:
        g = summary[summary["anchor"] == anchor]
        eligible = g[g["n_total_excl_qc"] >= MIN_GROUP_N]
        core = bool(len(eligible) > 0 and (eligible["pct_found_excl_qc"] >= 95).all())
        core_rows.append({
            "anchor": anchor,
            "is_core_recommended": anchor in CORE_ANCHORS,
            "core_candidate_by_qc": core,
            "n_groups_checked": len(eligible),
            "n_groups_total": len(g),
            "min_pct_found_excl_qc": round(float(eligible["pct_found_excl_qc"].min()), 2) if len(eligible) else np.nan,
        })
    core_df = pd.DataFrame(core_rows)

    n_qc_flagged = int((all_df["vertebra_order_anomaly"].fillna(False).astype(bool)
                        | all_df["hip_hardware_suspected"].fillna(False).astype(bool)).sum())
    print(f"QC 대상(척추 순서 이상/고관절 인공관절 의심) 환자: {n_qc_flagged}/{len(all_df)}명")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="site_scanner_anchor", index=False)
        core_df.to_excel(writer, sheet_name="core_candidate", index=False)
    print(f"QC 요약 저장: {out_path}")


# ── 랜드마크 육안 검증용 무작위 샘플 PNG ────────────────────────────────────────

def _window_to_uint8(pixels: np.ndarray, center: float, width: float) -> np.ndarray:
    low, high = center - width / 2, center + width / 2
    clipped = np.clip(pixels, low, high)
    return ((clipped - low) / max(high - low, 1e-6) * 255).astype(np.uint8)


def save_slice_png(dcm_path: str, out_path: str) -> bool:
    """DICOM 슬라이스 1장을 8-bit PNG로 저장(window center/width 기준, 없으면 1~99 percentile)."""
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


def save_landmark_previews(site: str, n_sample: int = 5, seed: int | None = None) -> None:
    """{site}_landmarks.xlsx에서 seg_status가 ok/partial인 환자 중 무작위 n_sample명을 뽑아
    찾아낸(status=ok) anchor의 Instance Number 슬라이스를 그대로 PNG로 저장한다.
    저장된 이미지를 열어 anchor 이름과 실제 해부학적 위치(예: L3 라벨 슬라이스가 정말
    L3 높이인지)가 맞는지 육안으로 확인하는 용도 — 세그멘테이션 재실행 없이 동작한다."""
    paths = site_paths(site)
    if not os.path.exists(paths["landmarks"]):
        print(f"[{site}] {paths['landmarks']} 없음 — 먼저 랜드마크를 추출하세요")
        return

    df = pd.read_excel(paths["landmarks"])
    candidates = df[df["seg_status"].isin(["ok", "partial"])]
    if candidates.empty:
        print(f"[{site}] 미리보기를 만들 수 있는 환자가 없습니다")
        return

    n_sample = min(n_sample, len(candidates))
    sample = candidates.sample(n=n_sample, random_state=seed)

    dlo = pd.read_excel(paths["dlo_path"])[["PatientID", "Series_Desc"]]
    dlo["PatientID"] = dlo["PatientID"].astype(int)
    series_desc_by_pid = dict(zip(dlo["PatientID"], dlo["Series_Desc"]))

    preview_dir = rf"{DATA_DIR}\{site}\aec\landmark_preview"
    saved_total = 0
    for _, row in sample.iterrows():
        pid = int(row["PatientID"])
        patient_dir = os.path.join(paths["dicom_base"], str(pid))
        if not os.path.isdir(patient_dir):
            print(f"  PID {pid}: DICOM 폴더 없음")
            continue
        folder = find_series_folder(patient_dir, series_desc_by_pid.get(pid))
        if folder is None:
            print(f"  PID {pid}: 시리즈 폴더 매칭 실패")
            continue
        by_inst = read_slices(os.path.join(patient_dir, folder))
        if by_inst is None:
            print(f"  PID {pid}: 슬라이스 읽기 실패")
            continue
        file_by_inst = {r["inst"]: r["file"] for r in by_inst}

        pid_saved = 0
        for anchor in ALL_ANCHORS:
            if row.get(f"{anchor}_status") != "ok":
                continue
            inst = row.get(f"{anchor}_instance")
            if pd.isna(inst):
                continue
            dcm_path = file_by_inst.get(int(inst))
            if dcm_path is None:
                continue
            out_path = os.path.join(preview_dir, str(pid), f"{anchor}_inst{int(inst)}.png")
            if save_slice_png(dcm_path, out_path):
                pid_saved += 1
        print(f"  PID {pid}: {pid_saved}개 anchor PNG 저장")
        saved_total += pid_saved

    print(f"[{site}] 샘플 {n_sample}명, 총 {saved_total}장 저장 → {preview_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=SITES, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--delete-shards", action="store_true")
    parser.add_argument("--qc-summary-only", action="store_true",
                        help="세그멘테이션 재실행 없이 기존 {site}_landmarks.xlsx로 QC 요약만 생성")
    args = parser.parse_args()

    sites = [args.site] if args.site else SITES
    qc_out = rf"{DATA_DIR}\landmark_qc_summary.xlsx"

    if PREVIEW_ONLY:
        for site in sites:
            save_landmark_previews(site, n_sample=PREVIEW_N, seed=PREVIEW_SEED)
        return

    if args.qc_summary_only:
        summarize_qc(sites, qc_out)
        return

    if args.merge_only:
        for site in sites:
            merge_shards(site, args.num_shards, delete_shard_files=args.delete_shards)
        summarize_qc(sites, qc_out)
        return

    for site in sites:
        run_site(site, shard_id=args.shard, num_shards=args.num_shards)

    if args.num_shards <= 1:
        summarize_qc(sites, qc_out)


if __name__ == "__main__":
    main()
