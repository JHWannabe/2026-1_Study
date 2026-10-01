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
  - Femoral head center (좌우 femur 상단부 centroid 평균, pubis~S1 사이 위치) — 체성분/
    리포트 대상에는 포함하되(BC_ANCHORS), seg_status 판정 기준인 CORE_ANCHORS에는 넣지
    않는다(검출 실패해도 환자 전체를 partial로 떨어뜨리지 않기 위함).
  - T9 center, T10 위
  - T8 center

각 anchor는 center(site)×scanner(manufacturer_model)별로 ≥95% 완전 포함될 때만
core 채택 후보로 표시한다(seg_summarize_qc). 고관절 인공관절 의심(HU 임계값 초과) /
척추 numbering anomaly(순서 역전)는 QC 대상으로 별도 플래그된다.

k축은 extract_liver_pubis_aec.py와 동일 관례를 따른다: 값이 클수록 cranial(머리 쪽),
값이 작을수록 caudal(발 쪽) — 기존 파이프라인에서 liver_upper_k=max, pubis_k=min으로
이미 검증된 방향과 동일하다.

랜드마크를 찾은 김에(DICOM을 한 번만 읽어) anchor 슬라이스마다 체성분(VAT/SAT/NAMA/
LAMA/IMATA, extract_liver_pubis_aec.compute_body_composition 재사용)과 AEC(XRayTubeCurrent)도
함께 계산한다 — anchor 구간 전체를 한 번에 세그멘테이션해 슬라이스별 값을 꺼내 쓰므로
(1슬라이스씩 11번 돌리면 z 문맥이 없어 마스크가 조각난다), 세그멘테이션 호출도 1번이면 된다.

[볼륨 방향 보정] 일부 시리즈(GE Revolution CT 등)는 ITK가 direction의 slice축을 좌수좌표계
(det=-1)로 읽어, NIfTI의 affine이 실제 DICOM z진행과 반대가 된다 — 그대로 두면
TotalSegmentator가 환자를 머리-발 뒤집힌 상태로 봐서 척추 번호가 뒤죽박죽된다. IOP 두 축의
외적으로 slice축을 되돌리고 실제 IPP 진행과 일치하는지 확인한다(불일치하면 direction_unresolved).

[절단 가드] liver_dome이 볼륨 최상단(k=n_k-1)에 닿거나, inferior_pubic_margin이 최하단(k=0)에
닿았는데 대퇴골두 아래 여유(PUBIS_MARGIN_MM)가 없으면 실제 해부학적 랜드마크가 아니라
스캔이 잘린 지점이므로 truncated로 표시한다(체성분·순서검사에서 제외).

[다중 시리즈 폴더] 한 폴더에 SeriesInstanceUID가 여러 개 섞인 경우(PACS 내보내기 중복),
슬라이스가 가장 많은(동률이면 z범위가 넓은) 하위 시리즈 하나만 골라 쓴다.

출력 (한 파일에 시트 4개로 모은다 — {site}/landmark/{slug}_landmark.xlsx):
  - landmarks        : 환자별 각 anchor의 k-index / Instance Number / z좌표 / status
  - body_composition : 환자×anchor(최대 12) 당 1행, VAT/SAT/NAMA/LAMA/IMATA
  - aec_total        : 환자별 전체 슬라이스 AEC(XRayTubeCurrent) 배열
  - aec_landmark     : 환자×anchor별 AEC 값(landmark 검출에 쓴 DICOM을 재사용, 추가 I/O 없음)
그 외:
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
import warnings
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
    _silence, _aec_values, _is_excluded_signal, compute_body_composition,
    find_series_folder, read_slices,
)
from totalsegmentator.python_api import totalsegmentator
import totalsegmentator.python_api as _tsa
# 호출마다 ~/.totalsegmentator/config.json을 읽고 다시 쓰는 사용 횟수 카운터는 프로세스를 여러 개 띄우면
# 서로 덮어쓰다 파일이 비어 JSONDecodeError(전 환자 실패)를 낸다 — 통계용이라 끈다.
_tsa.increase_prediction_counter = lambda: None

# 환자 1명씩 append할 때 신규 행의 NaN 컬럼(anchor 미검출 등)과 기존 시트를 concat하면
# 뜨는 경고. 현재 pandas에서는 dtype/값 모두 의도대로 나옴(재현 테스트로 확인) — 미래
# pandas 버전에서 동작이 바뀔 수 있다는 예고일 뿐이라, 며칠짜리 실행 콘솔을 도배하지
# 않도록 이 경고만 꺼둔다.
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*empty or all-NA entries.*")

# 7_select_best_series.py의 phase 우선순위(Portal > Contrast > Delay > Arterial >
# Pre/NonContrast > Unknown > Scout/Tracking) 선정 로직을 재사용 — 시리즈가 여러 개인
# 환자의 series를 고를 때 쓴다. 파일명이 숫자로 시작해 import 불가라 경로로 로드.
_sb_spec = importlib.util.spec_from_file_location(
    "select_best_series", Path(__file__).resolve().parent.parent / "7_select_best_series.py")
sb = importlib.util.module_from_spec(_sb_spec)
_sb_spec.loader.exec_module(sb)

# ── 설정 ──────────────────────────────────────────────────────────────────────

BATCH_SIZE      = 5  # 저장 1회 = 엑셀 전체 읽기+쓰기(~10초)라 5명마다 저장(중단 시 최대 5명 재처리)
MIN_VOXELS      = 50     # 척추/femur 라벨의 최소 유효 복셀 수(너무 작으면 미검출 취급)
HIP_HARDWARE_HU = 2500   # 고관절 인공관절(금속) 의심 HU 임계값
MIN_GROUP_N     = 10     # QC 완전포함률 계산 시 최소 표본 수(그 미만 그룹은 core 판정에서 제외)
PUBIS_MARGIN_MM = 30.0   # 대퇴골두 아래 이만큼 남아 있으면 두덩뼈 하연이 범위 안이라고 본다

# 무작위 샘플 PNG 미리보기 설정 (--preview-only 대신 여기서 직접 값을 바꿔서 사용)
PREVIEW_ONLY  = False    # True면 세그멘테이션 없이 기존 결과로 미리보기만 저장하고 종료
PREVIEW_N     = 5        # 사이트당 무작위 샘플 환자 수
PREVIEW_SEED  = None     # 샘플 재현용 랜덤 시드 (None이면 매번 랜덤)

VERTEBRAE = ["vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
             "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
             "vertebrae_S1"]
ROI_SUBSET = VERTEBRAE + ["liver", "hip_left", "hip_right", "femur_left", "femur_right"]  # check_landmarks_new10000.py가 참조

SITE_SLUG = {"강남": "gangnam", "신촌": "sinchon"}   # 출력 파일명은 로마자로 쓴다

CORE_ANCHORS = ["inferior_pubic_margin", "S1_center", "L5_center", "L4_center", "L3_center",
                "L2_center", "L1_center", "T12_center", "T11_center", "T10_center", "liver_dome"]
# 체성분/리포트 대상 anchor. CORE_ANCHORS(seg_status 판정 기준)에 femoral_head_center를
# 해부학적 위치(두덩뼈와 S1 사이)에 끼워 넣은 것 — 체성분은 뽑되, 검출 실패해도 seg_status를
# partial로 떨어뜨리지 않도록 CORE_ANCHORS 자체는 11개로 유지한다.
BC_ANCHORS = ["inferior_pubic_margin", "femoral_head_center", "S1_center", "L5_center",
              "L4_center", "L3_center", "L2_center", "L1_center", "T12_center", "T11_center",
              "T10_center", "liver_dome"]
# 체성분 저장 순서: liver_dome(cranial) → ... → inferior_pubic_margin(caudal). BC_ANCHORS는 반대 순.
ANCHOR_ORDER = list(reversed(BC_ANCHORS))
OPTIONAL_ANCHORS = ["femoral_head_center", "T9_center", "T8_center"]
# 아래(caudal)→위(cranial) 해부학적 순서로 컬럼 배열(엑셀 가독성). z값 기준 liver_dome은 T9~T8 사이에 위치.
ALL_ANCHORS = ["inferior_pubic_margin", "femoral_head_center", "S1_center", "L5_center", "L4_center",
               "L3_center", "L2_center", "L1_center", "T12_center", "T11_center", "T10_center",
               "T9_center", "liver_dome", "T8_center"]


def site_paths(site: str, shard: int | None = None) -> dict:
    suffix = f".shard{shard}" if shard is not None else ""
    # DICOM 원본과 DLO 입력 파일은 기존 한글 이름을 그대로 쓰고, 우리가 만드는 출력만
    # 로마자로 쓴다 — 한글 파일명이 셸을 거칠 때 인코딩이 깨지는 문제가 반복돼서다.
    slug = SITE_SLUG.get(site, site)
    return {
        "dicom_base": rf"E:\영상제공\{site}\{site}_axial",
        "dlo_path":   rf"{DATA_DIR}\{site}\metadata\{site}_DLO_Results.xlsx",
        # landmarks / body_composition / aec_total / aec_landmark를 한 파일의 시트로 모은다.
        "out":        rf"{DATA_DIR}\{site}\landmark\{slug}_landmark{suffix}.xlsx",
        "main_out":   rf"{DATA_DIR}\{site}\landmark\{slug}_landmark.xlsx",
        "checkpoint": rf"{DATA_DIR}\{site}\landmark\{slug}_landmark_checkpoint{suffix}.pkl",
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
           "n_slices": np.nan, "seg_status": status, "direction_fixed": False,
           "multi_series_in_folder": np.nan,
           "vertebra_order_anomaly": np.nan, "liver_below_T10": np.nan,
           "hip_hardware_suspected": np.nan}
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
    """환자 전체가 실패(no_series/nifti_fail/error 등)한 경우 BC_ANCHORS 전부에
    같은 실패 사유를 남긴다."""
    return [_empty_bc_row(pid, row, a, row["seg_status"]) for a in BC_ANCHORS]


def _aec_rows(pid: int, row: dict, by_inst: list[dict] | None) -> tuple[dict, dict]:
    """landmark 검출에 이미 읽어둔 by_inst를 재사용해 AEC를 만든다(DICOM 재읽기 없음).
    (aec_total 행, aec_landmark 행) 튜플. by_inst가 없으면 빈 행을 돌려준다."""
    meta = {"PatientID": pid, "series_description": row.get("series_description"),
            "manufacturer_model": row.get("manufacturer_model"),
            "n_slices": row.get("n_slices"), "seg_status": row.get("seg_status")}
    if not by_inst:
        return dict(meta, aec_full=[], aec_flat=np.nan), dict(meta)

    aec = [r["aec"] for r in by_inst]                 # InstanceNumber 오름차순
    total = dict(meta, aec_full=aec,
                 aec_flat=bool(_is_excluded_signal(_aec_values(by_inst))))

    # InstanceNumber가 1부터 시작하지 않는 시리즈가 있어 배열 위치(idx)를 따로 준다.
    idx_by_inst = {r["inst"]: i for i, r in enumerate(by_inst)}
    mark = dict(meta)
    for anchor in ALL_ANCHORS:
        inst = row.get(f"{anchor}_instance")
        mark[f"{anchor}_status"] = row.get(f"{anchor}_status")
        if inst is None or pd.isna(inst):
            mark[f"{anchor}_instance"] = np.nan
            mark[f"{anchor}_idx"] = pd.NA
            mark[f"{anchor}_aec"] = np.nan
            continue
        i = idx_by_inst.get(int(inst))
        mark[f"{anchor}_instance"] = int(inst)
        mark[f"{anchor}_idx"] = (i + 1) if i is not None else pd.NA   # aec_{idx} 컬럼과 대응
        mark[f"{anchor}_aec"] = aec[i] if i is not None else np.nan
    return total, mark


def _compute_bc_rows(pid: int, row: dict, img: sitk.Image | None, n_k: int, tmp_dir: str,
                      report_path: str | None = None) -> list[dict]:
    """랜드마크로 찾은 anchor 슬라이스(최대 12개)마다 체성분(VAT/SAT/NAMA/LAMA/IMATA)을
    계산한다. compute_landmarks()가 이미 채운 row와, process_patient()에서 한 번만 읽은 img를
    재사용해 DICOM을 다시 읽지 않는다. report_path가 주어지면 anchor별 세그멘테이션 오버레이
    이미지를 한 장(그리드)으로 합쳐 저장한다(육안 QC용)."""
    bc_rows = []
    bc_by_anchor: dict[str, dict] = {}

    # anchor마다 1슬라이스 볼륨에 tissue_4_types(3D nnU-Net)를 따로 돌리면 z 컨텍스트가 없어
    # 마스크가 조각나고 VAT가 0에 가깝게 나온다. anchor 전체 구간을 한 번에 세그멘테이션한 뒤
    # 슬라이스를 꺼내 쓴다 — 결과도 맞고 세그멘테이션 호출도 anchor개수번에서 1번으로 준다.
    anchor_k = {}
    for anchor in BC_ANCHORS:
        if row.get(f"{anchor}_status") != "ok":
            continue
        k = int(row[f"{anchor}_k"])
        if 0 <= k < n_k:
            anchor_k[anchor] = k

    vol = None
    if anchor_k and img is not None:
        lo, hi = min(anchor_k.values()), max(anchor_k.values())
        vol = compute_body_composition(img, lo, hi, tmp_dir, pid,
                                       keep_arrays=report_path is not None)

    # 구간 전체 결과에서 anchor 슬라이스별 면적(cm2)을 꺼낸다(배열은 k 오름차순).
    area_keys = {"VAT_sum_cm2": "VFA_slices", "SAT_sum_cm2": "SFA_slices",
                 "NAMA_sum_cm2": "NAMA_slices", "LAMA_sum_cm2": "LAMA_slices",
                 "IMATA_sum_cm2": "IMATA_slices"}
    for anchor in BC_ANCHORS:
        if row.get(f"{anchor}_status") != "ok":
            bc_rows.append(_empty_bc_row(pid, row, anchor, "anchor_not_found"))
            continue
        if anchor not in anchor_k:
            bc_rows.append(_empty_bc_row(pid, row, anchor, "k_out_of_range"))
            continue
        if vol is None or vol["seg_status"] != "ok":
            bc_rows.append(_empty_bc_row(pid, row, anchor,
                                          vol["seg_status"] if vol else "no_volume"))
            continue

        i = anchor_k[anchor] - lo
        bc = {"seg_status": "ok", "anchor_k": anchor_k[anchor],
              "anchor_instance": row.get(f"{anchor}_instance")}
        for out_key, slice_key in area_keys.items():
            vals = vol.get(slice_key) or []
            bc[out_key] = round(float(vals[i]), 2) if i < len(vals) else None
        if report_path is not None and "hu_slice" in vol:
            for arr_key in ("hu_slice", "sat_mask", "vat_mask",
                            "nama_mask", "lama_mask", "imata_mask"):
                bc[arr_key] = vol[arr_key][:, :, i]
        bc_by_anchor[anchor] = bc
        bc_rows.append({
            "PatientID": pid, "anchor": anchor,
            "anchor_k": anchor_k[anchor],
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
        # nii 축 순서는 (X=환자 왼쪽, Y=환자 뒤쪽, Z). imshow는 axis0을 화면 세로로 쓰므로
        # 그대로/rot90으로 그리면 상하가 뒤집힌다(전면이 아래로 감). transpose하면 화면 세로=
        # 앞→뒤, 가로=환자 오른쪽→왼쪽이 되어 DICOM pixel_array와 정확히 일치한다(MAE=0 검증).
        ax.imshow(np.transpose(rgb, (1, 0, 2)))
        txt = (f"VAT {bc['VAT_sum_cm2']:.1f}  SAT {bc['SAT_sum_cm2']:.1f}\n"
               f"NAMA {bc['NAMA_sum_cm2']:.1f}  LAMA {bc['LAMA_sum_cm2']:.1f}  IMATA {bc['IMATA_sum_cm2']:.1f}")
        ax.text(0.5, -0.14, txt, ha="center", va="top", fontsize=12, transform=ax.transAxes)

    for ax in axes[len(ANCHOR_ORDER):]:
        ax.axis("off")

    fig.suptitle(f"PatientID {pid}", fontsize=13)
    # 조직 색 범례 — overlay_colors를 그대로 써서 색을 바꾸면 범례도 따라간다.
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, label=k.replace("_mask", "").upper())
                        for k, c in overlay_colors.items()],
               loc="upper center", bbox_to_anchor=(0.5, 0.972), ncol=5,
               frameon=False, fontsize=11, handlelength=1.4, columnspacing=1.8)
    # 마지막 행 아래 여백(rect 하단)을 0으로 두면 ax.text(y=-0.14, 축 밖)로 그린 수치
    # 텍스트가 캔버스 밖으로 잘린다 — 하단에 여백을 남겨 마지막 줄도 다 보이게 한다.
    fig.tight_layout(rect=(0, 0.05, 1, 0.945))   # 상단에 범례 자리를 남긴다
    fig.subplots_adjust(hspace=0.45, wspace=0.15)
    fig.savefig(report_path, dpi=110)
    plt.close(fig)


# 인접 슬라이스의 z 차이로 두께(mm)를 구한다. 구할 수 없으면 None.
def _slice_spacing_mm(k_to_slice: list) -> float | None:
    zs = [r["z"] for r in k_to_slice if r and not np.isnan(r["z"])]
    if len(zs) < 2:
        return None
    d = float(np.median(np.abs(np.diff(zs))))
    return d if d > 0 else None


def compute_landmarks(row: dict, seg_dir: str, nii_path: str, n_k: int, k_to_slice: list) -> None:
    """seg_dir(TotalSegmentator 출력)에서 anchor들을 찾아 row에 채운다.
    k_to_slice[k]는 {"inst": ..., "z": ...}를 가진 dict(또는 None)여야 한다.
    사이트 기반(check_landmarks.py)/new10000 코호트 양쪽에서 공용으로 쓴다."""

    def _set_anchor(name: str, k_val: int | None, status: str = "ok"):
        if k_val is None:
            return
        k_val = max(0, min(k_val, n_k - 1))
        slice_row = k_to_slice[k_val]
        # status="truncated"면 좌표는 참고용으로 남기되 ok가 아니므로 체성분/순서검사에서 빠진다.
        row[f"{name}_status"] = status
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
    # 간 마스크가 볼륨 최상단(k=n_k-1)에 닿으면 실제 간 돔은 촬영 범위 밖이다.
    # 그대로 두면 "스캔이 잘린 지점"을 간 돔으로 잡게 되므로 truncated로 표시한다.
    if liver_k_arr is not None:
        liver_max = int(liver_k_arr.max())
        _set_anchor("liver_dome", liver_max,
                    "truncated" if liver_max >= n_k - 1 else "ok")

    # Inferior pubic margin: hip_left/right 최하단(caudal 끝, k 최소)
    hip_k_all = []
    for side in ("hip_left", "hip_right"):
        k_arr, _, _ = _largest_component_k(os.path.join(seg_dir, f"{side}.nii.gz"))
        if k_arr is not None:
            hip_k_all.append(k_arr)

    # Femoral head center (좌우 평균, 선택 anchor) — 아래 두덩뼈 절단 판정에 쓰므로 먼저 구한다.
    head_k, _ = _femoral_head_k(seg_dir)
    _set_anchor("femoral_head_center", head_k)

    # hip 마스크가 볼륨 최하단(k=0)에 닿으면 두덩뼈 하연이 촬영 범위 밖일 수 있다. 다만 hip
    # 마스크에는 좌골(ischium)도 들어 있어, 두덩뼈가 스캔 안에서 정상적으로 끝나도 좌골 가지가
    # k=0까지 이어져 마스크가 경계에 닿는다(예: 465292 — 두덩결합이 마지막 3슬라이스 위에서
    # 끝나는데도 절단으로 오판했다).
    # 그래서 대퇴골두를 기준으로 가른다. 대퇴골두 중심은 두덩결합과 비슷한 높이거나 약간 위라,
    # 그 아래로 여유가 있으면 두덩뼈 하연이 범위 안에 있다고 본다. 대퇴골두조차 못 찾았으면
    # (복부만 찍은 스캔) 실제 절단이다.
    if hip_k_all:
        hip_min = int(np.concatenate(hip_k_all).min())
        status = "ok"
        if hip_min <= 0:
            spacing = _slice_spacing_mm(k_to_slice)
            below_mm = (head_k * spacing) if (head_k is not None and spacing) else None
            status = "ok" if (below_mm is not None and below_mm >= PUBIS_MARGIN_MM) else "truncated"
        _set_anchor("inferior_pubic_margin", hip_min, status)

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

    # anchor numbering/detection anomaly: 척추(S1~T10) k가 caudal→cranial 단조증가해야 함.
    # liver_dome은 해부학적으로 T9~T10 높이라 T10보다 아래(k 작음)에 오는 것이 정상이고,
    # femoral_head_center는 대퇴골 상위 25% 구간 근사라 노이즈가 섞일 수 있어 둘 다 순서
    # 검사에서 뺀다(넣으면 정상 환자가 대거 이상으로 찍힌다). liver_dome 위치는
    # liver_below_T10으로 따로 기록한다.
    spine_k = [row[f"{a}_k"] for a in ANCHOR_ORDER
               if a not in ("liver_dome", "femoral_head_center") and row[f"{a}_status"] == "ok"]
    row["vertebra_order_anomaly"] = any(
        spine_k[i] <= spine_k[i + 1] for i in range(len(spine_k) - 1)
    ) if len(spine_k) >= 2 else False
    row["liver_below_T10"] = (
        bool(row["liver_dome_k"] <= row["T10_center_k"])
        if row["liver_dome_status"] == "ok" and row["T10_center_status"] == "ok" else np.nan)

    found_core = sum(1 for a in CORE_ANCHORS if row[f"{a}_status"] == "ok")
    row["seg_status"] = "ok" if found_core == len(CORE_ANCHORS) else (
        "partial" if found_core > 0 else "both_missing")


def process_patient(pid: int, series_dir: str, tmp_dir: str,
                     report_path: str | None = None) -> tuple[dict, list[dict], dict, dict]:
    """랜드마크 검출·체성분·AEC를 한 번에 수행한다(DICOM을 한 번만 읽어 재사용).
    (landmarks row, body-composition rows, aec_total row, aec_landmark row)를 반환한다.
    AEC는 랜드마크 검출에 이미 읽은 by_inst를 그대로 쓰므로 추가 I/O가 없다 — 별도
    스크립트로 돌리면 전체 DICOM 헤더를 한 번 더 읽게 되는데, 병목이 디스크라 그 비용이 크다."""
    try:
        hdr_file = next(os.path.join(series_dir, f) for f in os.listdir(series_dir)
                         if os.path.isfile(os.path.join(series_dir, f)))
        hdr = pydicom.dcmread(hdr_file, stop_before_pixels=True)
        series_desc  = str(getattr(hdr, "SeriesDescription", ""))
        manufacturer = str(getattr(hdr, "ManufacturerModelName", ""))
    except StopIteration:
        row = _empty_landmark_row(pid, None, None, "no_series")
        return (row, _empty_bc_rows(pid, row), *_aec_rows(pid, row, None))
    except Exception:
        series_desc = manufacturer = None

    by_inst = read_slices(series_dir)
    if by_inst is None:
        row = _empty_landmark_row(pid, series_desc, manufacturer, "no_position_data")
        return (row, _empty_bc_rows(pid, row), *_aec_rows(pid, row, None))

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
            return (row, _empty_bc_rows(pid, row), *_aec_rows(pid, row, by_inst))

        # 한 폴더에 SeriesInstanceUID가 여러 개 섞여 있는 경우가 있다 — PACS 내보내기에서
        # 같은 스캔이 통째로 두 번 저장되는 식이다(예: 강남 328371, 88장짜리 동일 시리즈 2벌).
        # 예전엔 첫 UID의 파일 수가 폴더 전체 수와 달라 invalid_volume_match로 환자를 통째로
        # 버렸다. 슬라이스가 가장 많은(동률이면 z범위가 넓은) 하위 시리즈 하나를 골라 쓴다.
        basename_to_slice = {os.path.basename(r["file"]): r for r in by_inst}

        def _coverage(kf: list) -> tuple:
            zs = [basename_to_slice[b]["z"] for b in map(os.path.basename, kf)
                  if b in basename_to_slice]
            zs = [z for z in zs if not np.isnan(z)]
            return (len(kf), (max(zs) - min(zs)) if len(zs) >= 2 else 0.0)

        candidates = [list(reader.GetGDCMSeriesFileNames(series_dir, sid)) for sid in series_ids]
        candidates = [kf for kf in candidates if kf]
        if not candidates:
            row["seg_status"] = "invalid_volume_match"
            return (row, _empty_bc_rows(pid, row), *_aec_rows(pid, row, by_inst))
        k_files = max(candidates, key=_coverage)

        if len(series_ids) > 1:
            # 고른 하위 시리즈에 속한 슬라이스만 남긴다. 그러지 않으면 InstanceNumber가
            # 중복돼(각 벌이 1~88) AEC 인덱싱과 슬라이스 매핑이 어긋난다.
            sel = {os.path.basename(f) for f in k_files}
            by_inst = [r for r in by_inst if os.path.basename(r["file"]) in sel]
            n = len(by_inst)
            row["n_slices"] = n
            row["multi_series_in_folder"] = len(series_ids)
            basename_to_slice = {os.path.basename(r["file"]): r for r in by_inst}

        if len(k_files) != n:
            row["seg_status"] = "invalid_volume_match"
            return (row, _empty_bc_rows(pid, row), *_aec_rows(pid, row, by_inst))

        reader.SetFileNames(k_files)
        img = reader.Execute()

        # [중요] 일부 시리즈(특히 GE Revolution CT)는 ITK가 direction의 slice축 부호를 뒤집어
        # 좌수좌표계(det=-1)로 읽는다. 그러면 여기서 쓰는 NIfTI의 affine이 실제 DICOM의 z진행과
        # 반대가 되고(axcodes가 S가 아니라 I), TotalSegmentator가 환자를 머리-발 뒤집힌 상태로
        # 보게 돼 척추 번호 라벨이 뒤죽박죽이 된다(강남 Revolution CT 490명 중 454명에서 발생).
        # slice축을 IOP 두 축의 외적(=우수계)으로 되돌리고, 실제 IPP 진행 방향과 맞는지 확인한다.
        d = np.array(img.GetDirection()).reshape(3, 3)
        if np.linalg.det(d) < 0:
            d[:, 2] = np.cross(d[:, 0], d[:, 1])
            z_first = float(pydicom.dcmread(k_files[0], stop_before_pixels=True).ImagePositionPatient[2])
            z_last  = float(pydicom.dcmread(k_files[-1], stop_before_pixels=True).ImagePositionPatient[2])
            if (d[2, 2] > 0) != (z_last > z_first):
                # 외적으로 복원한 방향이 실제 슬라이스 진행과 반대면 보정할 수 없는 형상이다.
                row["seg_status"] = "direction_unresolved"
                return (row, _empty_bc_rows(pid, row), *_aec_rows(pid, row, by_inst))
            img.SetDirection(tuple(d.flatten()))
            row["direction_fixed"] = True

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
            return (row, _empty_bc_rows(pid, row), *_aec_rows(pid, row, by_inst))

        k_to_slice = [basename_to_slice.get(os.path.basename(f)) for f in k_files]

        compute_landmarks(row, seg_dir, nii_path, n_k, k_to_slice)

    except Exception as e:
        row["seg_status"] = f"error:{type(e).__name__}:{e}"
        return (row, _empty_bc_rows(pid, row), *_aec_rows(pid, row, by_inst))
    finally:
        if os.path.isfile(nii_path):
            os.remove(nii_path)
        if os.path.isdir(seg_dir):
            shutil.rmtree(seg_dir, ignore_errors=True)

    bc_rows = _compute_bc_rows(pid, row, img, n_k, tmp_dir, report_path=report_path)
    return (row, bc_rows, *_aec_rows(pid, row, by_inst))


# ── 출력 (기존 파일 보존 + 신규 행만 append, 시트 4개를 한 파일에) ────────────────

def _clean_ws(s):
    return re.sub(r"\s+", " ", str(s)).strip() if pd.notna(s) else s


SHEETS = {"landmarks": "landmarks", "bodycomp": "body_composition",
          "aec_total": "aec_total", "aec_landmark": "aec_landmark"}


# 기존 시트를 모두 읽어온다(없으면 빈 dict).
def _read_sheets(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        return pd.read_excel(path, sheet_name=None)
    except Exception:
        return {}


def _write_all(path: str, rows: list[dict], bc_rows: list[dict],
               aec_t: list[dict], aec_m: list[dict], site: str | None = None) -> None:
    """landmarks / body_composition / aec_total / aec_landmark 네 시트를 한 파일에
    한 번에 저장한다. 기존 행은 보존하고 신규만 append하며, 중복은 기존 값을 유지한다
    (시트마다 따로 저장하면 환자 1명당 파일을 세 번 열고 쓰게 된다)."""
    if not (rows or bc_rows or aec_t or aec_m):
        return
    old = _read_sheets(path)

    def merge(name, new, subset):
        o = old.get(SHEETS[name])
        n = pd.DataFrame(new) if new else None
        if o is None and n is None:
            return None
        c = pd.concat([o, n], ignore_index=True) if (o is not None and n is not None) else (o if n is None else n)
        return c.drop_duplicates(subset=subset, keep="first")

    lm = merge("landmarks", rows, ["PatientID"])
    if lm is not None:
        lm["series_description"] = lm["series_description"].map(_clean_ws)
        # Excel에서 TRUE/FALSE는 숫자로 집계·필터하기 불편해 0/1로 저장한다(결측은 유지).
        for col in ("direction_fixed", "vertebra_order_anomaly",
                    "hip_hardware_suspected", "liver_below_T10"):
            if col in lm.columns:
                lm[col] = lm[col].map(lambda v: v if pd.isna(v) else int(bool(v))).astype("Int64")
        lm = lm.sort_values("PatientID").reset_index(drop=True)

    bc = merge("bodycomp", bc_rows, ["PatientID", "anchor"])
    if bc is not None:
        bc["anchor"] = pd.Categorical(bc["anchor"], categories=ANCHOR_ORDER, ordered=True)
        bc = bc.sort_values(["PatientID", "anchor"]).reset_index(drop=True)
        bc["anchor"] = bc["anchor"].astype(str)

    # aec_total은 aec_full 리스트를 aec_1..aec_n 컬럼으로 펼친다.
    meta = ["PatientID", "series_description", "manufacturer_model",
            "n_slices", "seg_status", "aec_flat"]
    flat = []
    for r in (aec_t or []):
        d = {c: r.get(c) for c in meta}
        for i, v in enumerate(r.get("aec_full", [])):
            d[f"aec_{i + 1}"] = v
        flat.append(d)
    o_t = old.get(SHEETS["aec_total"])
    n_old = sum(1 for c in (o_t.columns if o_t is not None else [])
                if str(c).startswith("aec_") and str(c) != "aec_flat")
    n_new = max((len(r.get("aec_full", [])) for r in (aec_t or [])), default=0)
    at = None
    if flat or o_t is not None:
        cols = meta + [f"aec_{i + 1}" for i in range(max(n_old, n_new))]
        nt = pd.DataFrame(flat).reindex(columns=cols) if flat else None
        at = pd.concat([o_t, nt], ignore_index=True) if (o_t is not None and nt is not None) else (o_t if nt is None else nt)
        at = at.drop_duplicates(subset=["PatientID"], keep="first").sort_values("PatientID").reset_index(drop=True)

    am = merge("aec_landmark", aec_m, ["PatientID"])
    if am is not None:
        am = am.sort_values("PatientID").reset_index(drop=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.xlsx"
    with pd.ExcelWriter(tmp, engine="openpyxl") as w:
        for df, sheet in ((lm, SHEETS["landmarks"]), (bc, SHEETS["bodycomp"]),
                          (at, SHEETS["aec_total"]), (am, SHEETS["aec_landmark"])):
            if df is not None:
                df.to_excel(w, sheet_name=sheet, index=False)
    os.replace(tmp, path)

    # 본 결과 파일(site 지정)이 갱신될 때마다 full/filtered도 같이 갱신한다. 실패해도 본 처리는 계속한다.
    if site:
        try:
            import merge_metadata
            merge_metadata.write_full_and_filtered(
                site, SITE_SLUG[site],
                {SHEETS[k]: d for k, d in (("landmarks", lm), ("bodycomp", bc),
                                           ("aec_total", at), ("aec_landmark", am)) if d is not None})
        except Exception as e:
            tqdm.write(f"  full/filtered 갱신 실패 ({type(e).__name__}: {e})")


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


def _collect_failed(paths: dict, check_paths: dict) -> set[int]:
    """기존 결과에서 seg_status가 ok가 아닌 PatientID를 모은다(재처리 대상)."""
    failed: set[int] = set()
    for path in {check_paths["out"], paths["out"]}:
        d = _read_sheets(path).get(SHEETS["landmarks"])
        if d is not None and len(d):
            failed |= set(d[d["seg_status"].astype(str) != "ok"]["PatientID"].astype(int))
    return failed


def _drop_pids(paths: dict, pids: set[int]) -> None:
    """재처리 전에 해당 PatientID의 기존 행을 모든 시트에서 지운다. 저장 로직이 중복을
    keep='first'로 처리하므로, 남겨두면 새로 계산한 결과가 아니라 옛 실패 행이 살아남는다."""
    if not pids:
        return
    path = paths["out"]
    sheets = _read_sheets(path)
    if not sheets:
        return
    tmp = path + ".tmp.xlsx"
    with pd.ExcelWriter(tmp, engine="openpyxl") as w:
        for name, d in sheets.items():
            if "PatientID" in d.columns:
                d = d[~d["PatientID"].astype("Int64").isin(pids)]
            d.to_excel(w, sheet_name=name, index=False)
    os.replace(tmp, path)


def _shard_filter(pid: int, shard_id: int, num_shards: int) -> bool:
    return num_shards <= 1 or (pid % num_shards) == shard_id


def pick_series_folder(pid: int, patient_dir: str, subs: list[str]) -> str | None:
    """환자 폴더의 series 서브폴더들을 7_select_best_series.py와 동일한 phase 우선순위
    (Portal > Contrast > Delay > Arterial > Pre/NonContrast > Unknown > Scout/Tracking)로
    골라 폴더명을 반환한다. 시리즈가 여러 개인 환자는 DLO_Results의 Series_Desc 힌트 대신
    항상 이 함수로 고른다 — 힌트를 그대로 따르면 비조영을 집는 경우가 많다."""
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

def run_site(site: str, shard_id: int = 0, num_shards: int = 1,
             limit: int | None = None, seed: int = 0, pids: list[int] | None = None,
             retry_failed: bool = False):
    sharded = num_shards > 1
    paths = site_paths(site, shard=shard_id if sharded else None)
    if limit is not None or pids:
        # 샘플 모드: 본 결과 파일/리포트를 건드리지 않도록 _sample 경로로 분리한다.
        for key in ("out", "checkpoint"):
            base, ext = os.path.splitext(paths[key])
            paths[key] = f"{base}_sample{ext}"
        paths["report_dir"] += "_sample"
    tag = f"{site}#{shard_id}" if sharded else site
    print("=" * 78)
    print(f"[{tag}] 시작 (전체 range 랜드마크 QC)  |  DICOM: {paths['dicom_base']}")

    dlo = pd.read_excel(paths["dlo_path"])[["PatientID", "Series_Desc"]]
    dlo["PatientID"] = dlo["PatientID"].astype(int)
    series_hint = dict(zip(dlo["PatientID"], dlo["Series_Desc"]))

    check_paths = site_paths(site)

    def _done(path: str) -> set:
        d = _read_sheets(path).get(SHEETS["landmarks"])
        return set(d["PatientID"].astype(int)) if d is not None and len(d) else set()

    done_pids = _done(check_paths["out"])
    if sharded:
        done_pids |= _done(paths["out"])
    attempted = _load_checkpoint(paths["checkpoint"])
    done_pids |= attempted

    # 기존 결과 중 ok가 아닌 환자를 먼저 다시 돌린다(수정된 코드로 되살리기). 인자 없이 그냥
    # 실행해도 자동으로 동작한다 — --site/--retry-failed를 매번 챙길 필요가 없게 하기 위함.
    retry_pids: set[int] = set()
    if retry_failed:
        retry_pids = _collect_failed(paths, check_paths)
        if sharded:
            retry_pids = {p for p in retry_pids if _shard_filter(p, shard_id, num_shards)}
        done_pids -= retry_pids
        attempted -= retry_pids
        _drop_pids(paths, retry_pids)
        _save_checkpoint(paths["checkpoint"], attempted)
        print(f"[{tag}] 재처리 대상(ok 아님) {len(retry_pids)}명 — 기존 행 제거 후 맨 마지막에 처리")

    # DLO_Results.xlsx(연구 대상자 명단) 대신, 신촌 971명/강남 21명이 누락돼 있던 DICOM
    # 폴더 전체를 대상으로 한다 — DLO에 없는 환자는 series_hint가 없어 pick_series_folder로
    # phase 우선순위 폴백 선택을 쓴다.
    all_pids = sorted(int(f) for f in os.listdir(paths["dicom_base"])
                       if f.isdigit() and os.path.isdir(os.path.join(paths["dicom_base"], f)))

    if pids:
        pending_pids = [p for p in pids if p in all_pids]
        missing = [p for p in pids if p not in all_pids]
        if missing:
            print(f"[{tag}] DICOM 폴더에 없는 PatientID 무시: {missing}")
        print(f"[{tag}] 지정 환자 {len(pending_pids)}명 → {os.path.basename(paths['out'])}")
    elif limit is not None:
        # 샘플 모드: 이미 처리된 환자도 다시 계산해야 수정한 코드의 효과를 확인할 수 있으므로
        # 전체에서 뽑는다.
        rng = np.random.default_rng(seed)
        pending_pids = sorted(rng.choice(all_pids, size=min(limit, len(all_pids)),
                                         replace=False).tolist())
        print(f"[{tag}] 샘플 모드 {len(pending_pids)}명 (seed={seed}) → {os.path.basename(paths['out'])}")
    else:
        pending_pids = [pid for pid in all_pids if pid not in done_pids]
        if sharded:
            pending_pids = [pid for pid in pending_pids if _shard_filter(pid, shard_id, num_shards)]
        # PatientID 큰 쪽부터 역순으로 처리하고, 재처리 대상(ok 아님)은 맨 마지막에 둔다.
        pending_pids = ([p for p in reversed(pending_pids) if p not in retry_pids]
                        + [p for p in reversed(pending_pids) if p in retry_pids])

    print(f"[{tag}] 대상 {len(all_pids)}명(DICOM 폴더 전체)  |  완료 {len(done_pids)}명  |  "
          f"처리 예정(이 shard) {len(pending_pids)}명")

    rows: list[dict] = []
    bc_rows_all: list[dict] = []
    aec_t_all: list[dict] = []
    aec_m_all: list[dict] = []
    n_processed = 0
    if pending_pids:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for pid in tqdm(pending_pids, total=len(pending_pids), desc=f"{tag} 처리"):
                patient_dir = os.path.join(paths["dicom_base"], str(pid))
                subs = [s for s in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, s))]
                if not subs:
                    row = _empty_landmark_row(pid, None, None, "no_series")
                    bc_rows = _empty_bc_rows(pid, row)
                    aec_t, aec_m = _aec_rows(pid, row, None)
                else:
                    # 시리즈가 여러 개면 조영 phase 우선순위(Portal > Contrast > Delay >
                    # Arterial > Pre/NonContrast > Unknown)로 고른다. DLO_Results의
                    # Series_Desc 힌트를 그대로 따르면 비조영을 집는 경우가 많다 — 신촌
                    # 표본에서 다중 시리즈 86명 중 56명(65%)이 Pre/NonContrast였다.
                    # 시리즈가 하나뿐이면 선택할 여지가 없으므로 그대로 쓴다.
                    if len(subs) == 1:
                        folder = subs[0]
                    else:
                        folder = pick_series_folder(pid, patient_dir, subs)
                    if folder is None:   # phase 선택 실패 시에만 힌트로 폴백
                        hint = series_hint.get(pid)
                        folder = find_series_folder(patient_dir, hint) if hint is not None else None
                    if folder is None:
                        row = _empty_landmark_row(pid, None, None, "no_series")
                        bc_rows = _empty_bc_rows(pid, row)
                        aec_t, aec_m = _aec_rows(pid, row, None)
                    else:
                        series_dir = os.path.join(patient_dir, folder)
                        report_path = os.path.join(paths["report_dir"], f"{pid}.png")
                        try:
                            row, bc_rows, aec_t, aec_m = process_patient(
                                pid, series_dir, tmp_dir, report_path=report_path)
                        except Exception as e:
                            row = _empty_landmark_row(pid, None, None, f"error:{type(e).__name__}:{e}")
                            bc_rows = _empty_bc_rows(pid, row)
                            aec_t, aec_m = _aec_rows(pid, row, None)

                rows.append(row)
                bc_rows_all.extend(bc_rows)
                aec_t_all.append(aec_t)
                aec_m_all.append(aec_m)
                tqdm.write(f"  PID {pid}: {row['seg_status']}")

                attempted.add(pid)
                n_processed += 1
                if n_processed % BATCH_SIZE == 0 or n_processed == len(pending_pids):
                    _write_all(paths["out"], rows, bc_rows_all, aec_t_all, aec_m_all,
                               site=None if (sharded or limit is not None or pids) else site)
                    _save_checkpoint(paths["checkpoint"], attempted)
                    rows = []
                    bc_rows_all = []
                    aec_t_all = []
                    aec_m_all = []
                    tqdm.write(f"  [{tag} 체크포인트 저장 | {n_processed}/{len(pending_pids)}]")

    if os.path.exists(paths["checkpoint"]):
        os.remove(paths["checkpoint"])
    print(f"[{tag}] 완료")


def merge_shards(site: str, num_shards: int, delete_shard_files: bool = False):
    """샤드별 통합 파일을 본 파일 하나로 합친다."""
    main = site_paths(site)
    for shard_id in range(num_shards):
        sp = site_paths(site, shard=shard_id)
        sheets = _read_sheets(sp["out"])
        if sheets:
            lm = sheets.get(SHEETS["landmarks"])
            bc = sheets.get(SHEETS["bodycomp"])
            at = sheets.get(SHEETS["aec_total"])
            am = sheets.get(SHEETS["aec_landmark"])
            aec_cols = [c for c in (at.columns if at is not None else [])
                        if str(c).startswith("aec_") and str(c) != "aec_flat"]
            t_rows = []
            for _, r in (at.iterrows() if at is not None else []):
                d = r.drop(labels=aec_cols).to_dict()
                d["aec_full"] = [v for v in r[aec_cols].tolist() if pd.notna(v)]
                t_rows.append(d)
            _write_all(main["out"],
                       lm.to_dict("records") if lm is not None else [],
                       bc.to_dict("records") if bc is not None else [],
                       t_rows,
                       am.to_dict("records") if am is not None else [])
        print(f"[{site}] shard {shard_id} 병합 완료")
        if delete_shard_files:
            for key in ("out", "checkpoint"):
                if os.path.exists(sp[key]):
                    os.remove(sp[key])
            print(f"[{site}] shard {shard_id} 임시 파일 삭제 완료")


# ── QC 요약: site x scanner x anchor 완전 포함률 ────────────────────────────────

def summarize_qc(sites: list[str], out_path: str) -> None:
    frames = []
    for site in sites:
        p = site_paths(site)
        df = _read_sheets(p["out"]).get(SHEETS["landmarks"])
        if df is None or not len(df):
            print(f"[{site}] {p['out']} 없음 — QC 요약에서 제외")
            continue
        df = df.copy()
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
    """landmarks 시트에서 seg_status가 ok/partial인 환자 중 무작위 n_sample명을 뽑아
    찾아낸(status=ok) anchor의 Instance Number 슬라이스를 그대로 PNG로 저장한다.
    저장된 이미지를 열어 anchor 이름과 실제 해부학적 위치(예: L3 라벨 슬라이스가 정말
    L3 높이인지)가 맞는지 육안으로 확인하는 용도 — 세그멘테이션 재실행 없이 동작한다."""
    paths = site_paths(site)
    df = _read_sheets(paths["out"]).get(SHEETS["landmarks"])
    if df is None or not len(df):
        print(f"[{site}] {paths['out']} 없음 — 먼저 랜드마크를 추출하세요")
        return

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
    """세그멘테이션 없이 순수 로직(anchor 못찾음 처리, 시트 write dedup/정렬)만 검증."""
    assert len(CORE_ANCHORS) == 11
    assert BC_ANCHORS == CORE_ANCHORS[:1] + ["femoral_head_center"] + CORE_ANCHORS[1:]
    row = {f"{a}_status": "missing" for a in BC_ANCHORS}
    empty_rows = _compute_bc_rows(1, row, img=None, n_k=0, tmp_dir="")
    assert len(empty_rows) == 12, "femoral_head_center 포함 12개여야 함"
    assert all(r["seg_status"] == "anchor_not_found" for r in empty_rows)

    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        out_path = os.path.join(d, "test.xlsx")
        _write_all(out_path, [], [{"PatientID": 1, "anchor": "L3_center", "anchor_instance": 50, "VAT_sum_cm2": 10.0}], [], [])
        _write_all(out_path, [], [{"PatientID": 1, "anchor": "L3_center", "anchor_instance": 50, "VAT_sum_cm2": 999.0},
                                  {"PatientID": 1, "anchor": "L1_center", "anchor_instance": 20, "VAT_sum_cm2": 20.0}], [], [])
        result = pd.read_excel(out_path, sheet_name=SHEETS["bodycomp"])
        assert len(result) == 2, "중복 (PatientID, anchor)는 기존 값을 유지해야 함"
        assert float(result.loc[result["anchor"] == "L3_center", "VAT_sum_cm2"].iloc[0]) == 10.0
        assert list(result["anchor"]) == ["L1_center", "L3_center"], "ANCHOR_ORDER 순으로 정렬돼야 함"

        # 시트 4종이 한 파일에 공존하고, 기존 시트가 보존되는지
        _write_all(out_path, [{"PatientID": 1, "series_description": "x", "seg_status": "ok"}], [],
                   [{"PatientID": 1, "aec_full": [1.0, 2.0]}], [{"PatientID": 1, "L3_center_aec": 2.0}])
        got = set(pd.read_excel(out_path, sheet_name=None).keys())
        assert got == set(SHEETS.values()), f"시트 구성이 다름: {got}"
        assert len(pd.read_excel(out_path, sheet_name=SHEETS["bodycomp"])) == 2, "기존 시트 보존"

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
                        help="세그멘테이션 재실행 없이 기존 결과로 QC 요약만 생성")
    parser.add_argument("--pids", type=str, default=None,
                        help="쉼표로 구분한 PatientID만 처리(_sample 경로에 저장)")
    parser.add_argument("--limit", type=int, default=None,
                        help="무작위 N명만 처리하고 _sample 경로에 저장(코드 수정 검증용)")
    parser.add_argument("--seed", type=int, default=0)
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

    # 인자 없이 그냥 실행해도 중단된 지점부터 이어서 돌고, 실패 건(seg_status!=ok)도
    # 자동으로 먼저 재시도한다 — --site/--retry-failed를 매번 챙길 필요가 없게 하기 위함.
    for site in sites:
        run_site(site, shard_id=args.shard, num_shards=args.num_shards,
                 limit=args.limit, seed=args.seed,
                 pids=[int(x) for x in args.pids.split(",")] if args.pids else None,
                 retry_failed=True)

    if args.num_shards <= 1 and args.limit is None and not args.pids:
        summarize_qc(sites, qc_out)


if __name__ == "__main__":
    main()
