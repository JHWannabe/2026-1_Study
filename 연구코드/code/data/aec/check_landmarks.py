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

랜드마크를 찾은 김에(DICOM을 한 번만 읽어) core anchor 슬라이스마다 체성분(VAT/SAT/NAMA/
LAMA/IMATA, extract_liver_pubis_aec.compute_body_composition 재사용)도 함께 계산한다
(원래 extract_landmark_body_composition.py로 분리돼 있던 2차 처리를 통합).

출력:
  - {site}_landmarks.xlsx : 환자별 각 anchor의 k-index / Instance Number / z좌표 / status
  - {site}_landmark_body_composition.xlsx : 환자×anchor(최대 11) 당 1행, VAT/SAT/NAMA/LAMA/IMATA
  - landmark_qc_summary.xlsx : site×scanner×anchor 완전 포함률 + core 채택 여부(2개 시트)
  - {site}/landmark/landmark_preview/{PatientID}/{anchor}_inst{N}.png : 랜드마크 육안 검증용 무작위 샘플 PNG
  - {site}/landmark/landmark_report/{PatientID}.png : anchor별 세그멘테이션 오버레이 + 체성분 수치를
    한 장(그리드)에 모은 환자별 QC 리포트
"""

import importlib.util
import os
import pickle
import re
import shutil
import tempfile
from pathlib import Path
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
    _silence, compute_body_composition, find_series_folder, read_slices,
)
from totalsegmentator.python_api import totalsegmentator

# 7_select_best_series.py의 phase 우선순위(Portal > Contrast > Delay > Arterial >
# Pre/NonContrast > Unknown) 선정 로직을 재사용 — DLO_Results.xlsx에 없는(=Series_Desc
# 힌트가 없는) 환자의 series를 고를 때 쓴다. 파일명이 숫자로 시작해 import 불가라 경로로 로드.
_sb_spec = importlib.util.spec_from_file_location(
    "select_best_series", Path(__file__).resolve().parent.parent / "7_select_best_series.py")
sb = importlib.util.module_from_spec(_sb_spec)
_sb_spec.loader.exec_module(sb)

# ── 설정 ──────────────────────────────────────────────────────────────────────

BATCH_SIZE      = 1  # 환자당 처리 시간이 길어 매 환자마다 저장(중단 시 손실 최소화)
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
ROI_SUBSET = VERTEBRAE + ["liver", "hip_left", "hip_right", "femur_left", "femur_right"]  # check_landmarks_new10000.py가 참조

CORE_ANCHORS = ["inferior_pubic_margin", "S1_center", "L5_center", "L4_center", "L3_center",
                "L2_center", "L1_center", "T12_center", "T11_center", "T10_center", "liver_dome"]
# 체성분 저장 순서: liver_dome(cranial) → ... → inferior_pubic_margin(caudal). CORE_ANCHORS는 반대 순.
ANCHOR_ORDER = list(reversed(CORE_ANCHORS))
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
        "landmarks":  rf"{DATA_DIR}\{site}\landmark\{site}_landmarks{suffix}.xlsx",
        "checkpoint": rf"{DATA_DIR}\{site}\landmark\{site}_landmarks_checkpoint{suffix}.pkl",
        "main_landmarks": rf"{DATA_DIR}\{site}\landmark\{site}_landmarks.xlsx",
        "bodycomp":   rf"{DATA_DIR}\{site}\landmark\{site}_landmark_body_composition{suffix}.xlsx",
        "report_dir": rf"{DATA_DIR}\{site}\landmark\landmark_report",
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


def _empty_bc_row(pid: int, row: dict, anchor: str, status: str) -> dict:
    return {
        "PatientID": pid, "anchor": anchor,
        "anchor_k": row.get(f"{anchor}_k"),
        "anchor_instance": row.get(f"{anchor}_instance"),
        "anchor_z": row.get(f"{anchor}_z"),
        "series_description": row.get("series_description"),
        "manufacturer_model": row.get("manufacturer_model"),
        "VAT_sum_cm2": None, "SAT_sum_cm2": None,
        "NAMA_sum_cm2": None, "LAMA_sum_cm2": None, "IMATA_sum_cm2": None,
        "seg_status": status,
    }


def _empty_bc_rows(pid: int, row: dict) -> list[dict]:
    """환자 전체가 실패(no_series/nifti_fail/error 등)한 경우 11개 anchor 모두에
    같은 실패 사유를 남긴다."""
    return [_empty_bc_row(pid, row, a, row["seg_status"]) for a in CORE_ANCHORS]


def _compute_bc_rows(pid: int, row: dict, img: sitk.Image | None, n_k: int, tmp_dir: str,
                      report_path: str | None = None) -> list[dict]:
    """랜드마크로 찾은 core anchor 슬라이스(최대 11개)마다 체성분(VAT/SAT/NAMA/LAMA/IMATA)을
    계산한다. compute_landmarks()가 이미 채운 row와, process_patient()에서 한 번만 읽은 img를
    재사용해 DICOM을 다시 읽지 않는다. report_path가 주어지면 anchor별 세그멘테이션 오버레이
    이미지를 한 장(그리드)으로 합쳐 저장한다(육안 QC용)."""
    bc_rows = []
    bc_by_anchor: dict[str, dict] = {}
    for anchor in CORE_ANCHORS:
        if row.get(f"{anchor}_status") != "ok":
            bc_rows.append(_empty_bc_row(pid, row, anchor, "anchor_not_found"))
            continue
        k = int(row[f"{anchor}_k"])
        if not (0 <= k < n_k):
            bc_rows.append(_empty_bc_row(pid, row, anchor, "k_out_of_range"))
            continue
        bc = compute_body_composition(img, k, k, tmp_dir, pid, keep_arrays=report_path is not None)
        bc["anchor_k"] = k
        bc["anchor_instance"] = row.get(f"{anchor}_instance")
        bc_by_anchor[anchor] = bc
        bc_rows.append({
            "PatientID": pid, "anchor": anchor,
            "anchor_k": k,
            "anchor_instance": row.get(f"{anchor}_instance"),
            "anchor_z": row.get(f"{anchor}_z"),
            "series_description": row.get("series_description"),
            "manufacturer_model": row.get("manufacturer_model"),
            "VAT_sum_cm2": bc["VAT_sum_cm2"], "SAT_sum_cm2": bc["SAT_sum_cm2"],
            "NAMA_sum_cm2": bc["NAMA_sum_cm2"], "LAMA_sum_cm2": bc["LAMA_sum_cm2"],
            "IMATA_sum_cm2": bc["IMATA_sum_cm2"],
            "seg_status": bc["seg_status"],
        })

    if report_path is not None and bc_by_anchor:
        try:
            _save_landmark_report(pid, bc_by_anchor, report_path)
        except Exception as e:
            tqdm.write(f"  PID {pid}: report 저장 실패 ({type(e).__name__}: {e})")

    return bc_rows


def _save_landmark_report(pid: int, bc_by_anchor: dict[str, dict], report_path: str) -> None:
    """anchor별 CT 슬라이스 + 조직 마스크 오버레이 + VAT/SAT/NAMA/LAMA/IMATA 수치를
    liver_dome→inferior_pubic_margin 순서로 그리드 배치해 환자당 PNG 한 장으로 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlay_colors = {
        "sat_mask":   (1.0, 1.0, 0.0),  # SAT: 노랑
        "vat_mask":   (1.0, 0.0, 0.0),  # VAT: 빨강
        "nama_mask":  (0.0, 0.4, 1.0),  # NAMA: 파랑
        "lama_mask":  (0.0, 1.0, 1.0),  # LAMA: 청록
        "imata_mask": (0.0, 1.0, 0.0),  # IMATA: 초록
    }
    wl, ww = 40, 400  # soft tissue window
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    ncols = 4
    nrows = -(-len(ANCHOR_ORDER) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5.2 * nrows))
    axes = axes.flatten()

    for ax, anchor in zip(axes, ANCHOR_ORDER):
        bc = bc_by_anchor.get(anchor)
        ax.axis("off")
        if bc is None or bc.get("seg_status") != "ok" or "hu_slice" not in bc:
            ax.set_title(anchor, fontsize=12)
            status = bc.get("seg_status") if bc else "anchor_not_found"
            ax.text(0.5, 0.5, f"N/A\n({status})", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
            continue

        hu = np.squeeze(bc["hu_slice"])
        gray = np.clip((hu - (wl - ww / 2)) / ww, 0, 1)
        rgb = np.stack([gray] * 3, axis=-1)
        for key, color in overlay_colors.items():
            mask = np.squeeze(bc[key])
            if mask.any():
                for c in range(3):
                    rgb[..., c] = np.where(mask, 0.55 * rgb[..., c] + 0.45 * color[c], rgb[..., c])

        inst = bc.get("anchor_instance")
        inst_str = f"slice #{int(inst)}" if inst is not None and not pd.isna(inst) else "slice #?"
        ax.set_title(f"{anchor} / {inst_str}", fontsize=12)
        ax.imshow(np.rot90(rgb))
        txt = (f"VAT {bc['VAT_sum_cm2']:.1f}  SAT {bc['SAT_sum_cm2']:.1f}\n"
               f"NAMA {bc['NAMA_sum_cm2']:.1f}  LAMA {bc['LAMA_sum_cm2']:.1f}  IMATA {bc['IMATA_sum_cm2']:.1f}")
        ax.text(0.5, -0.14, txt, ha="center", va="top", fontsize=12, transform=ax.transAxes)

    for ax in axes[len(ANCHOR_ORDER):]:
        ax.axis("off")

    fig.suptitle(f"PatientID {pid}", fontsize=13)
    # 마지막 행 아래 여백(rect 하단)을 0으로 두면 ax.text(y=-0.14, 축 밖)로 그린 수치
    # 텍스트가 캔버스 밖으로 잘린다 — 하단에 여백을 남겨 마지막 줄도 다 보이게 한다.
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.subplots_adjust(hspace=0.45, wspace=0.15)
    fig.savefig(report_path, dpi=110)
    plt.close(fig)


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

    # anchor numbering/detection anomaly: liver_dome(가장 cranial) -> inferior_pubic_margin(가장
    # caudal) 전체 core anchor(척추뿐 아니라 liver_dome/pubic margin 포함) 순으로 k가 단조
    # 감소해야 함 — 척추끼리는 순서가 맞아도 liver_dome이 잘못 낮게 잡히는 경우(예: 248989)는
    # 척추만 보던 이전 로직으론 못 잡았음.
    present_k = [row[f"{a}_k"] for a in ANCHOR_ORDER if row[f"{a}_status"] == "ok"]
    row["vertebra_order_anomaly"] = any(
        present_k[i] <= present_k[i + 1] for i in range(len(present_k) - 1)
    ) if len(present_k) >= 2 else False

    found_core = sum(1 for a in CORE_ANCHORS if row[f"{a}_status"] == "ok")
    row["seg_status"] = "ok" if found_core == len(CORE_ANCHORS) else (
        "partial" if found_core > 0 else "both_missing")


def process_patient(pid: int, series_dir: str, tmp_dir: str,
                     report_path: str | None = None) -> tuple[dict, list[dict]]:
    """랜드마크 검출과 core anchor별 체성분 계산을 한 번에 수행한다(DICOM을 한 번만 읽어 재사용).
    (landmarks row, body-composition row 최대 11개) 튜플을 반환한다."""
    try:
        hdr_file = next(os.path.join(series_dir, f) for f in os.listdir(series_dir)
                         if os.path.isfile(os.path.join(series_dir, f)))
        hdr = pydicom.dcmread(hdr_file, stop_before_pixels=True)
        series_desc  = str(getattr(hdr, "SeriesDescription", ""))
        manufacturer = str(getattr(hdr, "ManufacturerModelName", ""))
    except StopIteration:
        row = _empty_landmark_row(pid, None, None, "no_series")
        return row, _empty_bc_rows(pid, row)
    except Exception:
        series_desc = manufacturer = None

    by_inst = read_slices(series_dir)
    if by_inst is None:
        row = _empty_landmark_row(pid, series_desc, manufacturer, "no_position_data")
        return row, _empty_bc_rows(pid, row)

    n = len(by_inst)
    row = _empty_landmark_row(pid, series_desc, manufacturer, "failed")
    row["n_slices"] = n

    nii_path = os.path.join(tmp_dir, f"{pid}.nii.gz")
    seg_dir  = os.path.join(tmp_dir, f"{pid}_seg")
    img, n_k = None, None
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(series_dir)
        if not series_ids:
            row["seg_status"] = "nifti_fail"
            return row, _empty_bc_rows(pid, row)
        k_files = list(reader.GetGDCMSeriesFileNames(series_dir, series_ids[0]))
        if len(k_files) != n:
            row["seg_status"] = "invalid_volume_match"
            return row, _empty_bc_rows(pid, row)

        reader.SetFileNames(k_files)
        img = reader.Execute()
        sitk.WriteImage(img, nii_path)

        os.makedirs(seg_dir, exist_ok=True)
        with _silence():
            # roi_subset을 주면 TotalSegmentator가 저해상도 예비 세그멘테이션으로 해당 클래스
            # 영역만 크롭한 뒤 본 모델을 돌린다 — 척추가 전체 척추(경추~천추) 맥락 없이 잘려서
            # 번호가 밀리는 원인이 됐다. roi_subset 없이 전신을 그대로 세그멘테이션한다.
            totalsegmentator(input=nii_path, output=seg_dir, task="total",
                             fast=False, quiet=True, device=DEVICE)

        n_k = int(cast(Nifti1Image, nib.load(nii_path)).shape[2])
        if n_k != n:
            row["seg_status"] = "invalid_volume_match"
            return row, _empty_bc_rows(pid, row)

        basename_to_slice = {os.path.basename(r["file"]): r for r in by_inst}
        k_to_slice = [basename_to_slice.get(os.path.basename(f)) for f in k_files]

        compute_landmarks(row, seg_dir, nii_path, n_k, k_to_slice)

    except Exception as e:
        row["seg_status"] = f"error:{type(e).__name__}:{e}"
        return row, _empty_bc_rows(pid, row)
    finally:
        if os.path.isfile(nii_path):
            os.remove(nii_path)
        if os.path.isdir(seg_dir):
            shutil.rmtree(seg_dir, ignore_errors=True)

    bc_rows = _compute_bc_rows(pid, row, img, n_k, tmp_dir, report_path=report_path)
    return row, bc_rows


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


def _write_bodycomp(out_path: str, new_rows: list[dict]) -> None:
    """anchor를 liver_dome(cranial)→inferior_pubic_margin(caudal) 고정 순서로 정렬해 저장.
    (PatientID, anchor) 중복은 기존 값을 유지한다."""
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    old_df = pd.read_excel(out_path) if os.path.exists(out_path) else None
    combined = pd.concat([old_df, new_df], ignore_index=True) if old_df is not None else new_df
    combined = combined.drop_duplicates(subset=["PatientID", "anchor"], keep="first")
    combined["anchor"] = pd.Categorical(combined["anchor"], categories=ANCHOR_ORDER, ordered=True)
    combined = combined.sort_values(["PatientID", "anchor"]).reset_index(drop=True)
    combined["anchor"] = combined["anchor"].astype(str)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_excel(out_path, index=False)


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


def pick_series_folder(pid: int, patient_dir: str, subs: list[str]) -> str | None:
    """DLO_Results.xlsx에 Series_Desc 힌트가 없는 환자(폴더에는 있지만 DLO 명단 밖인 환자)를 위한
    폴백 — 환자 폴더의 series 서브폴더들을 7_select_best_series.py와 동일한 phase 우선순위
    (Portal > Contrast > Delay > Arterial > Pre/NonContrast > Unknown)로 골라 폴더명을 반환한다."""
    rows = []
    for sub in subs:
        sub_dir = os.path.join(patient_dir, sub)
        files = [f for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
        if not files:
            continue
        try:
            hdr = pydicom.dcmread(os.path.join(sub_dir, files[0]), stop_before_pixels=True)
        except Exception:
            continue
        series_desc = str(getattr(hdr, "SeriesDescription", sub))
        thickness = getattr(hdr, "SliceThickness", None)
        by_inst = read_slices(sub_dir)
        z_vals = [r["z"] for r in by_inst] if by_inst else []
        z_range_mm = round(max(z_vals) - min(z_vals), 2) if len(z_vals) >= 2 else 0.0
        rows.append({"PatientID": pid, "series_desc": sub,
                     "phase_rank": sb.PHASE_RANK[sb.classify_phase(series_desc)],
                     "z_range_mm": z_range_mm,
                     "slice_thickness_mm": float(thickness) if thickness not in (None, "") else 99.0})
    if not rows:
        return None
    return str(sb.pick_best(pd.DataFrame(rows)).iloc[0]["series_desc"])


# ── 사이트 실행 ───────────────────────────────────────────────────────────────

def run_site(site: str, shard_id: int = 0, num_shards: int = 1):
    sharded = num_shards > 1
    paths = site_paths(site, shard=shard_id if sharded else None)
    tag = f"{site}#{shard_id}" if sharded else site
    print("=" * 78)
    print(f"[{tag}] 시작 (전체 range 랜드마크 QC)  |  DICOM: {paths['dicom_base']}")

    dlo = pd.read_excel(paths["dlo_path"])[["PatientID", "Series_Desc"]]
    dlo["PatientID"] = dlo["PatientID"].astype(int)
    series_hint = dict(zip(dlo["PatientID"], dlo["Series_Desc"]))

    check_paths = site_paths(site)
    done_pids = set()
    if os.path.exists(check_paths["landmarks"]):
        done_pids |= set(pd.read_excel(check_paths["landmarks"])["PatientID"].astype(int))
    if sharded and os.path.exists(paths["landmarks"]):
        done_pids |= set(pd.read_excel(paths["landmarks"])["PatientID"].astype(int))
    attempted = _load_checkpoint(paths["checkpoint"])
    done_pids |= attempted

    # DLO_Results.xlsx(연구 대상자 명단) 대신, 신촌 971명/강남 21명이 누락돼 있던 DICOM
    # 폴더 전체를 대상으로 한다 — DLO에 없는 환자는 series_hint가 없어 pick_series_folder로
    # phase 우선순위 폴백 선택을 쓴다.
    all_pids = sorted(int(f) for f in os.listdir(paths["dicom_base"])
                       if f.isdigit() and os.path.isdir(os.path.join(paths["dicom_base"], f)))
    pending_pids = [pid for pid in all_pids if pid not in done_pids]
    if sharded:
        pending_pids = [pid for pid in pending_pids if _shard_filter(pid, shard_id, num_shards)]
    print(f"[{tag}] 대상 {len(all_pids)}명(DICOM 폴더 전체)  |  완료 {len(done_pids)}명  |  "
          f"처리 예정(이 shard) {len(pending_pids)}명")

    rows: list[dict] = []
    bc_rows_all: list[dict] = []
    n_processed = 0
    if pending_pids:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for pid in tqdm(pending_pids, total=len(pending_pids), desc=f"{tag} 처리"):
                patient_dir = os.path.join(paths["dicom_base"], str(pid))
                subs = [s for s in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, s))]
                if not subs:
                    row = _empty_landmark_row(pid, None, None, "no_series")
                    bc_rows = _empty_bc_rows(pid, row)
                else:
                    hint = series_hint.get(pid)
                    folder = find_series_folder(patient_dir, hint) if hint is not None else None
                    if folder is None:
                        folder = pick_series_folder(pid, patient_dir, subs)
                    if folder is None:
                        row = _empty_landmark_row(pid, None, None, "no_series")
                        bc_rows = _empty_bc_rows(pid, row)
                    else:
                        series_dir = os.path.join(patient_dir, folder)
                        report_path = os.path.join(paths["report_dir"], f"{pid}.png")
                        try:
                            row, bc_rows = process_patient(pid, series_dir, tmp_dir, report_path=report_path)
                        except Exception as e:
                            row = _empty_landmark_row(pid, None, None, f"error:{type(e).__name__}:{e}")
                            bc_rows = _empty_bc_rows(pid, row)

                rows.append(row)
                bc_rows_all.extend(bc_rows)
                tqdm.write(f"  PID {pid}: {row['seg_status']}")

                attempted.add(pid)
                n_processed += 1
                if n_processed % BATCH_SIZE == 0 or n_processed == len(pending_pids):
                    _write_landmarks(paths, rows)
                    _write_bodycomp(paths["bodycomp"], bc_rows_all)
                    _save_checkpoint(paths["checkpoint"], attempted)
                    rows = []
                    bc_rows_all = []
                    tqdm.write(f"  [{tag} 체크포인트 저장 | {n_processed}/{len(pending_pids)}]")

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
        if os.path.exists(sp["bodycomp"]):
            df = pd.read_excel(sp["bodycomp"])
            _write_bodycomp(main_paths["bodycomp"], df.to_dict("records"))
        print(f"[{site}] shard {shard_id} 병합 완료")
        if delete_shard_files:
            for key in ("landmarks", "checkpoint", "bodycomp"):
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

    preview_dir = rf"{DATA_DIR}\{site}\landmark\landmark_preview"
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


def _self_test() -> None:
    """세그멘테이션 없이 순수 로직(anchor 못찾음 처리, bodycomp write dedup/정렬)만 검증."""
    assert len(CORE_ANCHORS) == 11
    row = {f"{a}_status": "missing" for a in CORE_ANCHORS}
    empty_rows = _compute_bc_rows(1, row, img=None, n_k=0, tmp_dir="")
    assert len(empty_rows) == 11
    assert all(r["seg_status"] == "anchor_not_found" for r in empty_rows)

    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        out_path = os.path.join(d, "test.xlsx")
        _write_bodycomp(out_path, [{"PatientID": 1, "anchor": "L3_center", "anchor_instance": 50, "VAT_sum_cm2": 10.0}])
        _write_bodycomp(out_path, [{"PatientID": 1, "anchor": "L3_center", "anchor_instance": 50, "VAT_sum_cm2": 999.0},
                                    {"PatientID": 1, "anchor": "L1_center", "anchor_instance": 20, "VAT_sum_cm2": 20.0}])
        result = pd.read_excel(out_path)
        assert len(result) == 2, "중복 (PatientID, anchor)는 기존 값을 유지해야 함"
        assert float(result.loc[result["anchor"] == "L3_center", "VAT_sum_cm2"].iloc[0]) == 10.0
        assert list(result["anchor"]) == ["L1_center", "L3_center"], "ANCHOR_ORDER(liver_dome→pubis) 순으로 정렬돼야 함"

        fake = np.zeros((20, 20, 1))
        mask = np.zeros((20, 20, 1), dtype=bool)
        mask[5:10, 5:10, :] = True
        bc_ok = {"seg_status": "ok", "hu_slice": fake, "sat_mask": mask, "vat_mask": mask,
                 "nama_mask": mask, "lama_mask": mask, "imata_mask": mask,
                 "VAT_sum_cm2": 1.0, "SAT_sum_cm2": 2.0, "NAMA_sum_cm2": 3.0,
                 "LAMA_sum_cm2": 4.0, "IMATA_sum_cm2": 5.0}
        report_path = os.path.join(d, "report.png")
        _save_landmark_report(1, {"liver_dome": bc_ok}, report_path)
        assert os.path.exists(report_path), "report PNG가 생성돼야 함"
    print("self-test OK")


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
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

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
