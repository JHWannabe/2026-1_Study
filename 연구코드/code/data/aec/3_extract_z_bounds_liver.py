"""
[전체 목적]
CT 영상에서 "간 상단 슬라이스 번호"와 "두덩뼈 하단 슬라이스 번호"를 자동으로 추출한다.
이 두 경계(z_bounds)는 AEC(자동노출제어) 신호를 clipping할 구간을 정의하는 데 사용된다.

[처리 흐름]
  1. DICOM 폴더를 환자별로 읽어 NIfTI(.nii.gz) 3D 볼륨으로 변환
  2. TotalSegmentator(AI)로 간(Liver)과 엉덩뼈(Hip)를 3D 분할(segmentation)
  3. 간 → 상단 슬라이스(cranial end) 탐지
  4. 엉덩뼈 → 두덩뼈 하단 슬라이스(pubis) 탐지
  5. 각 슬라이스의 DICOM Instance Number와 z 좌표를 엑셀로 저장

[출력 컬럼]
  liver_upper_slice : 간 상단 슬라이스의 DICOM Instance Number
  pubis_slice       : 두덩뼈 하단 슬라이스의 DICOM Instance Number
  n_slices          : 스캔 전체 슬라이스 수
  slices            : 클리핑 구간(간 상단 ~ 두덩뼈 하단) 슬라이스 수
  z_liver_upper     : 간 상단 슬라이스의 z 좌표 (mm, 환자 좌표계)
  z_pubis           : 두덩뼈 하단 슬라이스의 z 좌표
  z_range           : z_liver_upper - z_pubis 절댓값 (구간 길이 mm)

[용어 설명]
  DICOM        : CT 영상의 표준 파일 형식. 슬라이스 1장 = 파일 1개.
                 각 파일에는 픽셀 외에도 환자정보·위치·방향 등 메타데이터가 포함된다.
  NIfTI        : AI 분할 도구가 주로 사용하는 3D 의료영상 형식.
                 DICOM 슬라이스 묶음을 하나의 3D 배열(x, y, z)로 표현한다.
  Instance Number : DICOM 슬라이스마다 부여된 고유 번호.
                    AEC 신호 배열의 인덱스와 1:1 대응되어 경계 좌표로 사용된다.
  k-index      : NIfTI 3D 배열에서 슬라이스 순서를 나타내는 인덱스 (0-based).
                 dcm_files[k] 로 해당 DICOM 파일을 찾을 수 있다.
  cranial      : 머리 방향 (위쪽). 간은 머리 쪽에 위치한다.
  caudal/pubis : 발 방향 (아래쪽). 두덩뼈는 발 쪽에 위치한다.
  복셀(voxel)  : 3D 공간에서 부피를 가진 픽셀. CT에서 각 슬라이스의 픽셀 하나.

[segmentation 도구]
  Liver : liver_segments task → 간 구역 1~8 마스크를 합산 (정밀 분할)
  Hip   : total task, roi_subset=hip_left, hip_right → 엉덩뼈 마스크로 두덩뼈 위치 추정

[출력 파일]
  {SITE}_z_bounds.xlsx
  liver_pubis/ok/upper|bottom/{PatientID}_{Instance}.png      ← 정상 추출 미리보기
  liver_pubis/missing/upper|bottom/{PatientID}_{Instance}.png ← 추출 실패 미리보기
"""

import os
import io
import sys
import json
import shutil
import tempfile
import logging
import contextlib
import threading
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

# ── GPU 성능 최적화 ────────────────────────────────────────────────────────────
# TotalSegmentator(nnUNet v2)는 내부적으로 AMP(torch.cuda.amp.autocast)를 이미 사용함.
# 아래는 추가로 적용 가능한 설정이다.

# cuDNN benchmark: 입력 크기가 고정된 경우 가장 빠른 알고리즘을 자동 선택한다.
# CT 슬라이스 해상도가 환자마다 다를 수 있으므로 효과가 제한적일 수 있다.
torch.backends.cudnn.benchmark = True

# TF32: Ampere 이상(RTX 30 시리즈+) GPU에서 행렬 연산에 TensorFloat-32를 사용한다.
# PyTorch 1.7+ 에서 기본 활성화되어 있지만 명시적으로 설정한다.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32      = True

# ── TotalSegmentator / nnUNet 출력 억제 ───────────────────────────────────────
# quiet=True 로도 nnUNet 내부의 print() 호출("There are N cases …")은 막히지 않음.
# logging 경로는 레벨 차단, print() 경로는 thread-local 래퍼로 스레드별 독립 억제.
for _lg in ("nnunetv2", "totalsegmentator", "batchgeneratorsv2",
            "acvl_utils", "dynamic_network_architectures"):
    logging.getLogger(_lg).setLevel(logging.ERROR)


class _ThreadLocalStream:
    """
    sys.stdout / sys.stderr 를 대체하는 thread-local 래퍼.

    각 스레드가 독립적으로 출력 대상(sink)을 설정할 수 있다.
      · sink 설정 → 해당 스레드의 print() 출력이 sink(StringIO)로 흡수됨
      · sink 미설정 → 원래 스트림으로 출력 (tqdm 등 정상 출력)

    _silence() 가 이 래퍼를 통해 각 스레드별로 독립적으로 억제하므로
    totalsegmentator 가 락 없이 완전 병렬 실행된다.
    """
    def __init__(self, real_stream):
        self._real = real_stream
        self._local = threading.local()

    def _target(self):
        sink = getattr(self._local, "sink", None)
        return sink if sink is not None else self._real

    def write(self, data):
        return self._target().write(data)

    def flush(self):
        return self._target().flush()

    def __getattr__(self, name):
        # encoding, errors, isatty 등 나머지 속성은 원래 스트림에서 위임
        return getattr(self._real, name)


# 모듈 로드 시 1회 교체 — 이후 모든 print() 가 래퍼를 거친다.
# _ts_* 변수에 보관해 _silence() 에서 타입 오류 없이 _local 에 직접 접근한다.
_ts_stdout = _ThreadLocalStream(sys.stdout)
_ts_stderr = _ThreadLocalStream(sys.stderr)
sys.stdout = _ts_stdout  # type: ignore[assignment]
sys.stderr = _ts_stderr  # type: ignore[assignment]


@contextlib.contextmanager
def _silence():
    """
    현재 스레드의 print() 출력만 억제한다 (완전 thread-safe).

    다른 스레드(tqdm 등)는 원래 스트림을 그대로 사용하므로 영향 없음.
    totalsegmentator 는 락 없이 여러 스레드에서 동시에 실행된다.
    """
    sink = io.StringIO()
    _ts_stdout._local.sink = sink   # 현재 스레드에만 적용
    _ts_stderr._local.sink = sink
    try:
        yield
    finally:
        _ts_stdout._local.sink = None
        _ts_stderr._local.sink = None


# ── 설정 ──────────────────────────────────────────────────────────────────────

SITE       = "강남"
DICOM_BASE = rf"D:/영상제공/{SITE}/{SITE}_axial"   # 환자별 DICOM 폴더 최상위 경로
OUT_PATH   = rf"C:\Users\jhjun\OneDrive\Desktop\대학원\data\{SITE}\aec\liver_pubis\{SITE}_z_bounds.xlsx"
CHECKPOINT       = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\.z_bounds_checkpoint.json"
RETRY_CHECKPOINT = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\.z_bounds_retry_checkpoint.json"

# PNG 미리보기 저장 디렉토리
UPPER_DIR         = rf"C:\Users\jhjun\OneDrive\Desktop\대학원\data\{SITE}\aec\liver_pubis\ok\upper"
BOTTOM_DIR        = rf"C:\Users\jhjun\OneDrive\Desktop\대학원\data\{SITE}\aec\liver_pubis\ok\bottom"
MISSING_UPPER_DIR = rf"C:\Users\jhjun\OneDrive\Desktop\대학원\data\{SITE}\aec\liver_pubis\missing\upper"
MISSING_BOTTOM_DIR= rf"C:\Users\jhjun\OneDrive\Desktop\대학원\data\{SITE}\aec\liver_pubis\missing\bottom"

# 간 구역 이름 목록 (Couinaud 분류: 간을 8개 구역으로 나눈 표준)
LIVER_SEGMENTS = [f"liver_segment_{i}" for i in range(1, 9)]
ROI_SUBSET_HIP = ["hip_left", "hip_right"]   # 좌우 엉덩뼈

BATCH_SIZE   = 16   # 몇 명마다 중간 저장(체크포인트)할지
NUM_WORKERS  = 4   # 병렬 처리 스레드 수

# 분할 품질 기준값
MIN_LIVER_VOXELS = 3000  # 유효 간 마스크의 최소 복셀 수 (너무 적으면 오분할로 간주)

# seg_status 값 중 "재처리 대상"으로 분류되는 것들
MISSING_STATUSES = {"liver_missing", "pubis_missing", "both_missing"}

# "full"  : 전체 환자 처음부터 실행
# "retry" : 이전 실행에서 missing으로 기록된 환자만 재처리
MODE = "full"


# ── DICOM 유틸 ────────────────────────────────────────────────────────────────

def build_folder_map(dicom_base: str) -> dict[str, str]:
    """
    DICOM 최상위 경로를 스캔해 PatientID → 폴더 절대경로 매핑을 반환한다.

    지원하는 폴더명 형식:
      ① 신규: {PatientID}              (예: 5253702)
      ② 구형: {순번}_{PatientID}_...   (예: 0001_5253702_20180201_CT)
    """
    folder_map: dict[str, str] = {}
    if not os.path.isdir(dicom_base):
        raise FileNotFoundError(f"DICOM 경로 없음: {dicom_base}")
    for folder_name in os.listdir(dicom_base):
        full_path = os.path.join(dicom_base, folder_name)
        if not os.path.isdir(full_path):
            continue
        parts = folder_name.split("_")
        if len(parts) == 1:
            # ① 신규 형식: 폴더명 자체가 PatientID
            folder_map[folder_name] = full_path
        elif len(parts) >= 2:
            # ② 구형 형식: 두 번째 항목이 PatientID
            folder_map[parts[1]] = full_path
    return folder_map


def dicom_to_nifti_with_mapping(dcm_dir: str, out_path: str) -> list | None:
    """
    한 환자의 DICOM 폴더를 NIfTI 파일로 변환하고,
    슬라이스 순서(k-index)와 DICOM 파일 경로의 1:1 매핑을 반환한다.

    반환값: k=0, 1, 2, ... 순서에 대응되는 DICOM 파일 경로 리스트
      dcm_files[k] → k번째 슬라이스의 DICOM 파일 경로
    실패 시 None 반환.

    SimpleITK가 DICOM 시리즈를 자동으로 공간 순서(z좌표 기준)로 정렬하므로,
    반환 리스트의 인덱스와 NIfTI의 슬라이스 인덱스가 1:1 대응된다.
    """
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dcm_dir)       # 폴더 내 DICOM 시리즈 ID 목록
        if not series_ids:
            return None
        # 첫 번째 시리즈의 파일을 z 순서로 정렬
        dicom_names = reader.GetGDCMSeriesFileNames(dcm_dir, series_ids[0])
        reader.SetFileNames(dicom_names)
        sitk.WriteImage(reader.Execute(), out_path)          # NIfTI로 저장
        return list(dicom_names)
    except Exception:
        return None


def get_instance_number(dcm_path: str) -> int | None:
    """
    DICOM 파일에서 Instance Number(슬라이스 고유 번호)를 읽는다.
    Instance Number는 AEC 신호 배열의 인덱스와 대응되는 핵심 식별자다.
    픽셀 데이터는 읽지 않고 헤더만 읽어 속도를 높인다.
    """
    try:
        return int(pydicom.dcmread(dcm_path, stop_before_pixels=True).InstanceNumber)
    except Exception:
        return None


def read_z(dcm_path: str) -> float:
    """
    DICOM 파일에서 z 좌표(ImagePositionPatient[2])를 읽는다.
    z 좌표는 환자 좌표계에서 해당 슬라이스의 실제 위치(mm)를 나타낸다.
    z_range = |z_liver_upper - z_pubis| 계산에 사용된다.
    """
    try:
        return float(pydicom.dcmread(dcm_path, stop_before_pixels=True).ImagePositionPatient[2])
    except Exception:
        return float(np.nan)


def get_k_indices(mask_path: str) -> np.ndarray:
    """
    NIfTI 마스크 파일에서 분할된 부위(양성 복셀)가 존재하는
    슬라이스 인덱스(k-index) 배열을 반환한다.

    3D 마스크 배열의 shape = (x, y, z). z축(axis=2)이 슬라이스 방향이므로
    양성 복셀의 z 인덱스만 모아 반환한다.
    """
    if not os.path.exists(mask_path):
        return np.array([])
    data = cast(Nifti1Image, nib.load(mask_path)).get_fdata()
    idx = np.argwhere(data > 0.5)          # 0.5 이상 = 분할된(양성) 복셀
    return idx[:, 2].astype(int) if len(idx) > 0 else np.array([])


# ── PNG 저장 유틸 ─────────────────────────────────────────────────────────────

# 각 구조물에 사용할 오버레이 색상 (R, G, B), 0~1 범위
_SEG_COLORS = {
    "liver":    (1.0, 0.25, 0.25),   # 빨강 계열
    "hip_left": (0.2,  0.45, 1.0),   # 파랑
    "hip_right":(0.2,  0.75, 1.0),   # 하늘색
}


def _read_dicom_pixels(dcm_path: str) -> np.ndarray:
    """
    DICOM 픽셀을 눈으로 보기 좋은 형태(uint8, 0~255)로 변환한다.

    CT 픽셀값은 Hounsfield Unit(HU) 단위로 저장된다.
      실제 HU = 픽셀값 × RescaleSlope + RescaleIntercept
    의료 영상 뷰어는 Window Center/Width 설정으로 특정 HU 범위를 밝기에 매핑한다.
    Window가 없으면 1~99 percentile로 자동 계산해 명암을 조정한다.
    """
    dcm = pydicom.dcmread(dcm_path)
    pixels = dcm.pixel_array.astype(np.float32)
    # HU 변환: 저장 픽셀값 → 실제 밀도값
    pixels = pixels * float(getattr(dcm, "RescaleSlope", 1) or 1) \
                    + float(getattr(dcm, "RescaleIntercept", 0) or 0)

    # Window 설정 읽기 (일부 DICOM은 리스트로 저장되어 있어 첫 원소를 사용)
    wc = getattr(dcm, "WindowCenter", None)
    ww = getattr(dcm, "WindowWidth", None)
    if wc is not None and ww is not None:
        wc = float(wc[0]) if hasattr(wc, "__iter__") and not isinstance(wc, str) else float(wc)
        ww = float(ww[0]) if hasattr(ww, "__iter__") and not isinstance(ww, str) else float(ww)
    else:
        # Window가 없으면 outlier를 제외한 범위로 자동 설정
        lo, hi = np.percentile(pixels, [1, 99])
        wc, ww = (lo + hi) / 2, hi - lo

    # Window 범위 밖의 값을 잘라내고 0~255로 정규화
    low, high = wc - ww / 2, wc + ww / 2
    return ((np.clip(pixels, low, high) - low) / max(high - low, 1e-6) * 255).astype(np.uint8)


def save_slice_png(dcm_path: str, out_path: str) -> bool:
    """DICOM 슬라이스 1장을 segmentation 오버레이 없이 단순 8-bit PNG로 저장."""
    try:
        img = _read_dicom_pixels(dcm_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(img).save(out_path)
        return True
    except Exception:
        return False


def save_slice_png_with_seg(dcm_path: str, seg_masks: dict, out_path: str) -> bool:
    """
    DICOM 슬라이스 위에 segmentation 마스크를 반투명하게 오버레이하여 PNG로 저장한다.
    seg_masks: {"구조물명": 2D 마스크 배열} 형태.
    마스크가 있는 픽셀에 색상을 alpha=0.45로 덧그린다.
    """
    try:
        img = _read_dicom_pixels(dcm_path)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(img, cmap="gray")
        patches = []
        for name, mask in seg_masks.items():
            if mask is None or mask.sum() == 0:
                continue
            color = _SEG_COLORS.get(name, (0.0, 1.0, 0.0))
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)   # 투명 캔버스 생성
            rgba[mask > 0.5] = [*color, 0.45]                      # 마스크 영역에만 색 입힘
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
                           upper_dir: str = UPPER_DIR, bottom_dir: str = BOTTOM_DIR) -> tuple[bool, bool]:
    """
    경계 슬라이스 2장(간 상단, 두덩뼈 하단)을 PNG로 저장해 시각 검수용으로 남긴다.
    성공/실패 여부에 따라 ok/ 또는 missing/ 폴더에 저장된다.
    overlay 저장 실패 시 마스크 없이 단순 PNG로 대체 저장한다.
    """
    def _save(dcm_path: str, masks: dict | None, out_dir: str) -> bool:
        inst = get_instance_number(dcm_path)
        fname = f"{pid_str}_{inst}.png" if inst is not None else f"{pid_str}.png"
        out = os.path.join(out_dir, fname)
        if masks:
            ok = save_slice_png_with_seg(dcm_path, masks, out)
            if not ok:
                tqdm.write(f"    [PNG] overlay 저장 실패 → 단순 저장으로 전환: {fname}")
                ok = save_slice_png(dcm_path, out)
            return ok
        return save_slice_png(dcm_path, out)

    upper_ok  = _save(upper_dcm,  upper_masks,  upper_dir)  if upper_dcm  else False
    bottom_ok = _save(bottom_dcm, bottom_masks, bottom_dir) if bottom_dcm else False
    return upper_ok, bottom_ok


def _remove_missing_pngs(pid_str: str):
    """
    retry에서 이전 missing 환자가 성공으로 전환되면,
    missing 폴더에 남아 있던 해당 환자의 PNG 파일을 삭제한다.
    새로 ok 폴더에 저장되므로 중복을 방지한다.
    """
    for d in (MISSING_UPPER_DIR, MISSING_BOTTOM_DIR):
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.startswith(pid_str) and fname.endswith(".png"):
                os.remove(os.path.join(d, fname))
                tqdm.write(f"    [cleanup] 삭제: {os.path.join(d, fname)}")


def _count_pngs() -> dict:
    """각 출력 폴더의 PNG 파일 수를 반환한다 (최종 요약 출력용)."""
    def count(d: str) -> int:
        return len([f for f in os.listdir(d) if f.endswith(".png")]) if os.path.isdir(d) else 0
    return {
        "ok/upper":      count(UPPER_DIR),
        "ok/bottom":     count(BOTTOM_DIR),
        "missing/upper": count(MISSING_UPPER_DIR),
        "missing/bottom":count(MISSING_BOTTOM_DIR),
    }


# ── Segmentation 보조 함수 ────────────────────────────────────────────────────

def keep_largest_component(vol: np.ndarray) -> np.ndarray:
    """
    3D 마스크 볼륨에서 가장 큰 연결 성분(connected component)만 남기고 나머지를 제거한다.

    AI 분할 결과에는 실제 간 외에도 작은 노이즈 복셀 덩어리가 산발적으로 포함될 수 있다.
    이 노이즈는 apex(간 최상단) 탐지를 방해하므로, 가장 큰 덩어리(= 실제 간)만 유지한다.

    예: [큰 간 덩어리] [작은 노이즈] [작은 노이즈] → [큰 간 덩어리]만 남김
    """
    binary = vol > 0.5
    labeled, n = cast(tuple[np.ndarray, int], ndimage.label(binary))  # 연결된 덩어리마다 번호 부여
    if n <= 1:
        return vol   # 덩어리가 1개 이하면 그대로 반환
    # 각 덩어리의 복셀 수를 계산해 가장 큰 것의 번호를 찾음
    sizes = ndimage.sum(binary, labeled, list(range(1, n + 1)))
    return (labeled == int(np.argmax(sizes)) + 1).astype(np.float32)


def is_liver_anatomically_valid(liver_k: np.ndarray, n_k: int, cranial_is_higher_k: bool) -> bool:
    """
    분할된 간의 위치가 해부학적으로 타당한지 검증한다.

    정상적인 CT에서 간은 전체 스캔 범위의 대략 30~70% 구간 내에 위치한다.
    간의 슬라이스 인덱스 중앙값이 전체의 30% 이상(머리 방향 기준)이면 유효로 판단한다.

    이 검증은 AI가 간을 잘못 감지하거나 이상한 부위를 분할했을 때를 걸러내기 위한 안전장치다.
    """
    if len(liver_k) == 0:
        return False
    median_k = float(np.median(liver_k))
    # cranial_is_higher_k=True: 머리가 높은 k쪽 → 간이 k의 30% 이상 위치해야 함
    # cranial_is_higher_k=False: 머리가 낮은 k쪽 → 간이 k의 70% 이하 위치해야 함
    return median_k > n_k * 0.3 if cranial_is_higher_k else median_k < n_k * 0.7


def _segment_liver(nii_path: str, seg_dir: str,
                   method: str = "liver_segments",
                   quiet: bool = True) -> np.ndarray | None:
    """
    TotalSegmentator로 간 3D 마스크를 생성하고 반환한다.

    method="liver_segments" (기본값, 정밀):
      - 간 구역 1~8을 개별로 분할한 뒤 OR 연산으로 합쳐 하나의 전체 간 마스크를 만든다.
      - 간 구역(liver segment)이란 혈관 분포에 따른 외과적 분류(Couinaud, 1957)다.
    method="total" (빠름):
      - total task의 roi_subset=["liver"]로 단일 마스크를 직접 추출한다.

    최소 복셀 수(MIN_LIVER_VOXELS=3000) 미만이면 분할 실패로 간주하고 None을 반환한다.
    """
    try:
        if method == "liver_segments":
            with _silence():
                totalsegmentator(input=nii_path, output=seg_dir,
                                 task="liver_segments", quiet=quiet, device="gpu")
            liver_vol: np.ndarray | None = None
            for seg_name in LIVER_SEGMENTS:    # liver_segment_1 ~ liver_segment_8
                p = os.path.join(seg_dir, f"{seg_name}.nii.gz")
                if os.path.exists(p):
                    vol = cast(Nifti1Image, nib.load(p)).get_fdata()
                    # 구역별 마스크를 누적 OR로 합산 → 전체 간 마스크
                    liver_vol = vol if liver_vol is None \
                                else np.logical_or(liver_vol, vol > 0.5).astype(np.float32)
        else:  # method == "total"
            with _silence():
                totalsegmentator(input=nii_path, output=seg_dir,
                                 task="total", roi_subset=["liver"],
                                 quiet=quiet, device="gpu")
            p = os.path.join(seg_dir, "liver.nii.gz")
            liver_vol = cast(Nifti1Image, nib.load(p)).get_fdata() if os.path.exists(p) else None

        # 복셀 수가 최소 기준 이상이어야 유효한 분할로 인정
        if liver_vol is not None and int((liver_vol > 0.5).sum()) >= MIN_LIVER_VOXELS:
            return liver_vol
        tqdm.write(f"    [liver] {method} 결과 부족 → liver_missing")
    except Exception as e:
        tqdm.write(f"    [liver] {method} 실패({type(e).__name__}: {e}) → liver_missing")
    return None


# ── 경계 추출 (환자 1명) ──────────────────────────────────────────────────────

def extract_bounds(pid_str: str, tmp_dir: str, folder_map: dict,
                   liver_method: str = "liver_segments") -> dict:
    """
    환자 1명에 대해 간 상단 / 두덩뼈 하단 경계를 추출하고 결과 딕셔너리를 반환한다.

    [처리 단계]
      1. DICOM → NIfTI 변환
      2. AI 분할: 간(liver_segments), 엉덩뼈(hip)
      3. 스캔 방향 판별 (머리가 높은 k인지 낮은 k인지)
      4. 간 마스크 정제 + 유효성 검증
      5. 간 상단(apex) 슬라이스 탐지 (99%ile + 후퇴 알고리즘)
      6. 두덩뼈 하단(bottom) 슬라이스 탐지
      7. 결과 저장 + PNG 미리보기 생성
      8. 임시 파일 정리(finally)

    [seg_status 값]
      "ok"                      : 간·두덩뼈 모두 정상 추출
      "liver_missing"           : 간 분할 실패
      "pubis_missing"           : 엉덩뼈 분할 실패
      "both_missing"            : 둘 다 실패
      "instance_number_missing" : 분할은 성공했으나 DICOM Instance Number를 읽지 못함
      "no_dicom" / "no_series"  : DICOM 파일 자체가 없음
      "nifti_fail"              : NIfTI 변환 실패
      "invalid_volume_match"    : 슬라이스 수 불일치
      "error:..."               : 예외 발생
    """
    # 결과 초기값: 모든 수치를 nan으로, 상태를 "failed"로 초기화
    result = {
        "liver_upper_slice": np.nan, "pubis_slice": np.nan,
        "slices": np.nan,            "n_slices": np.nan,
        "z_liver_upper": np.nan,     "z_pubis": np.nan,
        "z_range": np.nan,           "seg_status": "failed",
    }

    # ── 사전 검증: DICOM 경로 존재 여부 ──────────────────────────────────────
    if pid_str not in folder_map:
        result["seg_status"] = "no_dicom"
        return result

    subfolders = [s for s in os.listdir(folder_map[pid_str])
                  if os.path.isdir(os.path.join(folder_map[pid_str], s))]
    if not subfolders:
        result["seg_status"] = "no_series"
        return result

    # 실제 DICOM 파일이 있는 서브폴더 (시리즈 폴더)
    dcm_dir       = os.path.join(folder_map[pid_str], subfolders[0])
    nii_path      = os.path.join(tmp_dir, f"{pid_str}.nii.gz")      # 변환된 NIfTI 저장 경로
    seg_dir_liver = os.path.join(tmp_dir, f"{pid_str}_seg_liver")   # 간 분할 결과 디렉토리
    seg_dir_hip   = os.path.join(tmp_dir, f"{pid_str}_seg_hip")     # 엉덩뼈 분할 결과 디렉토리

    try:
        # ── STEP 1: DICOM → NIfTI 변환 ───────────────────────────────────────
        # dcm_files[k] = k번째 슬라이스의 DICOM 파일 경로 (z좌표 오름차순 정렬)
        dcm_files = dicom_to_nifti_with_mapping(dcm_dir, nii_path)
        if not dcm_files:
            result["seg_status"] = "nifti_fail"
            return result

        # ── STEP 2: AI Segmentation ───────────────────────────────────────────
        os.makedirs(seg_dir_liver, exist_ok=True)
        liver_vol = _segment_liver(nii_path, seg_dir_liver, method=liver_method,
                                   quiet=True)

        os.makedirs(seg_dir_hip, exist_ok=True)
        # hip_left.nii.gz, hip_right.nii.gz 생성
        with _silence():
            totalsegmentator(input=nii_path, output=seg_dir_hip,
                             task="total", roi_subset=ROI_SUBSET_HIP,
                             fast=False, quiet=True, device="gpu")

        # 전체 슬라이스 수 확인 (NIfTI의 z축 크기)
        n_k = int(cast(Nifti1Image, nib.load(nii_path)).shape[2])
        result["n_slices"] = n_k
        # NIfTI 슬라이스 수와 DICOM 파일 수가 다르면 매핑이 불가능 → 중단
        if n_k < 2 or len(dcm_files) != n_k:
            result["seg_status"] = "invalid_volume_match"
            return result

        # ── STEP 3: 엉덩뼈 슬라이스 인덱스 수집 ─────────────────────────────
        # 좌우 엉덩뼈 마스크에서 복셀이 존재하는 슬라이스 인덱스를 합친다
        hip_k = np.concatenate([
            get_k_indices(os.path.join(seg_dir_hip, f"{side}.nii.gz"))
            for side in ("hip_left", "hip_right")
        ]).astype(int)
        hip_found = len(hip_k) > 0

        # ── STEP 4: 스캔 방향 판별 ────────────────────────────────────────────
        # CT는 기기마다 두부→족부(머리→발) 또는 족부→두부 순으로 스캔한다.
        # 간은 머리 쪽, 두덩뼈는 발 쪽에 위치하므로,
        # 두 구조물의 슬라이스 중앙값을 비교해 방향을 자동으로 판별한다.
        #
        #   cranial_is_higher_k = True  → 높은 k(인덱스)가 머리 방향
        #                                 (족부→두부 스캔: k가 증가할수록 머리 방향)
        #   cranial_is_higher_k = False → 낮은 k(인덱스)가 머리 방향
        #                                 (두부→족부 스캔: k=0이 머리 방향)
        if liver_vol is not None and hip_found:
            raw_liver_k = np.argwhere(liver_vol > 0.5)[:, 2].astype(int)
            # 간의 중앙 슬라이스 > 엉덩뼈의 중앙 슬라이스 이면 → 간이 높은 k쪽 = 머리가 높은 k
            cranial_is_higher_k = bool(np.median(raw_liver_k) > np.median(hip_k))
        elif liver_vol is not None:
            raw_liver_k = np.argwhere(liver_vol > 0.5)[:, 2].astype(int)
            # 엉덩뼈 정보 없으면 간이 전체 슬라이스 중앙보다 위에 있는지로 판단
            cranial_is_higher_k = bool(np.median(raw_liver_k) > n_k / 2)
        elif hip_found:
            # 간 정보 없으면 엉덩뼈가 아래쪽(낮은 k)에 있으면 머리가 높은 k라고 판단
            cranial_is_higher_k = bool(np.median(hip_k) < n_k / 2)
        else:
            cranial_is_higher_k = True   # 모두 없으면 기본값 사용

        # ── STEP 5: 간 마스크 정제 및 유효성 검증 ────────────────────────────
        if liver_vol is not None:
            # 노이즈 제거: 가장 큰 연결 덩어리만 남긴다
            liver_vol = keep_largest_component(liver_vol)
            liver_k = np.argwhere(liver_vol > 0.5)[:, 2].astype(int)
            # 복셀 수 기준 + 해부학적 위치 기준 모두 통과해야 유효
            liver_found = len(liver_k) >= MIN_LIVER_VOXELS \
                          and is_liver_anatomically_valid(liver_k, n_k, cranial_is_higher_k)
            if not liver_found:
                tqdm.write(f"    [liver] 해부학적 검증 실패 (voxels={len(liver_k)}) → liver_missing")
        else:
            liver_k = np.array([], dtype=int)
            liver_found = False

        # ── STEP 6: 간 상단(apex) 슬라이스 탐지 ─────────────────────────────
        # keep_largest_component로 잡음 덩어리가 제거된 상태이므로,
        # 남은 liver_k에서 cranial 방향의 끝 k = 간 단면적이 가장 작은 슬라이스.
        #
        # cranial_is_higher_k=True  → liver_k.max() (k가 높은 쪽이 머리 방향)
        # cranial_is_higher_k=False → liver_k.min() (k가 낮은 쪽이 머리 방향)
        if liver_found:
            apex_k = int(liver_k.max() if cranial_is_higher_k else liver_k.min())
        else:
            # 간을 찾지 못한 경우: 머리 방향의 끝 슬라이스를 fallback으로 사용
            apex_k = n_k - 1 if cranial_is_higher_k else 0

        # 인덱스가 유효 범위를 벗어나지 않도록 클리핑
        apex_k = max(0, min(apex_k, n_k - 1))
        upper_dcm = dcm_files[apex_k]                            # 해당 슬라이스의 DICOM 파일
        result["z_liver_upper"]     = read_z(upper_dcm)          # z 좌표 (mm)
        result["liver_upper_slice"] = get_instance_number(upper_dcm) or np.nan  # Instance Number

        # ── STEP 7: 두덩뼈 하단(bottom) 슬라이스 탐지 ───────────────────────
        # 엉덩뼈 마스크 중 가장 낮은 k(발쪽 끝) 슬라이스를 두덩뼈 하단으로 사용한다.
        # 두덩뼈(pubic bone)는 엉덩뼈 구조물 중 가장 발쪽에 위치하기 때문이다.
        # k의 오름차순 정렬에서 첫 번째 값 = 가장 낮은 k = 발쪽 끝
        bottom_k = int(sorted(np.unique(hip_k))[0]) if hip_found \
                   else (0 if cranial_is_higher_k else n_k - 1)  # 실패 시 fallback
        bottom_k = max(0, min(bottom_k, n_k - 1))
        bottom_dcm = dcm_files[bottom_k]
        result["pubis_slice"] = get_instance_number(bottom_dcm) or np.nan
        result["z_pubis"]     = read_z(bottom_dcm)

        # ── STEP 8: 구간 통계 계산 ────────────────────────────────────────────
        result["slices"] = abs(apex_k - bottom_k) + 1   # 경계 간 슬라이스 수
        if pd.notna(result["z_liver_upper"]) and pd.notna(result["z_pubis"]):
            result["z_range"] = round(abs(result["z_liver_upper"] - result["z_pubis"]), 2)

        # ── STEP 9: seg_status 결정 ───────────────────────────────────────────
        if liver_found and hip_found:
            result["seg_status"] = (
                "instance_number_missing"
                if pd.isna(result["liver_upper_slice"]) or pd.isna(result["pubis_slice"])
                else "ok"
            )
        elif liver_found:
            result["seg_status"] = "pubis_missing"     # 엉덩뼈만 실패
        elif hip_found:
            result["seg_status"] = "liver_missing"     # 간만 실패
        else:
            result["seg_status"] = "both_missing"      # 둘 다 실패

        # ── STEP 10: PNG 미리보기 저장 ────────────────────────────────────────
        # ok 폴더 또는 missing 폴더 중 하나에 저장
        is_ok = result["seg_status"] == "ok"
        if is_ok:
            _remove_missing_pngs(pid_str)   # 이전 retry에서 생긴 missing PNG 정리

        # bottom 슬라이스의 엉덩뼈 마스크: 3D 마스크에서 해당 슬라이스(bottom_k)만 추출
        bottom_masks = {
            side: cast(Nifti1Image, nib.load(os.path.join(seg_dir_hip, f"{side}.nii.gz"))).get_fdata()[:, :, bottom_k].T
            for side in ("hip_left", "hip_right")
            if os.path.exists(os.path.join(seg_dir_hip, f"{side}.nii.gz"))
        }
        # upper 슬라이스의 간 마스크: apex_k 슬라이스 추출 (.T로 전치해 imshow에 맞춤)
        if liver_vol is not None and len(liver_k) > 0:
            upper_png_dcm  = upper_dcm                           # dcm_files[apex_k]
            upper_png_mask = {"liver": liver_vol[:, :, apex_k].T}
        else:
            upper_png_dcm, upper_png_mask = upper_dcm, {}
        save_boundary_previews(pid_str, upper_png_dcm, bottom_dcm, upper_png_mask, bottom_masks,
                               UPPER_DIR if is_ok else MISSING_UPPER_DIR,
                               BOTTOM_DIR if is_ok else MISSING_BOTTOM_DIR)

    except Exception as e:
        tqdm.write(f"  [ERROR] PID {pid_str}: {type(e).__name__}: {e}")
        result["seg_status"] = f"error:{type(e).__name__}:{e}"

    finally:
        # ── 임시 파일 정리 (성공/실패 여부와 무관하게 항상 실행) ──────────────
        # NIfTI와 분할 결과 폴더를 삭제해 디스크 공간을 확보한다.
        for path in (nii_path, seg_dir_liver, seg_dir_hip):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    return result


# ── 체크포인트 ────────────────────────────────────────────────────────────────
# 체크포인트는 중간 진행 상태를 JSON 파일로 저장하는 기능이다.
# 프로그램이 중단되더라도 마지막 저장 지점부터 이어서 실행할 수 있다.

def _clean_for_json(v):
    """
    numpy 타입(np.int64, np.float64 등)은 JSON으로 직렬화할 수 없으므로
    Python 기본 타입(int, float)으로 변환한다.
    NaN(결측값)은 JSON에서 지원하지 않으므로 None으로 바꾼다.
    """
    if hasattr(v, "item"):
        return v.item()           # numpy scalar → Python scalar
    if isinstance(v, float) and np.isnan(v):
        return None               # NaN → None (JSON의 null)
    return v


def save_checkpoint(completed_pids: set, rows: list, path: str = CHECKPOINT):
    """
    완료된 환자 ID 집합(completed_pids)과 결과 행(rows)을 JSON 파일로 저장한다.
    병렬 처리 환경에서 완료 순서가 달라지므로 카운터 대신 PID 집합으로 관리한다.
    """
    data = {
        "processed":     len(completed_pids),          # 이전 형식 호환용 (참고값)
        "completed_pids": sorted(completed_pids),       # 완료된 PID 목록 (재시작 필터링용)
        "rows": [{k: _clean_for_json(v) for k, v in r.items()} for r in rows],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_checkpoint(path: str = CHECKPOINT) -> tuple[set, list]:
    """
    이전에 저장된 체크포인트를 불러온다.
    파일이 없으면 (set(), [])을 반환해 처음부터 시작함을 나타낸다.
    JSON의 None은 다시 np.nan으로 복원한다.

    [형식 호환]
    · 신 형식 (병렬): completed_pids 키 존재 → 그대로 로드
    · 구 형식 (순차): completed_pids 키 없음 → rows의 PatientID로 자동 재구성
      (순차 실행 중 중단 후 병렬 모드로 재시작할 때도 정상 동작)
    """
    if not os.path.exists(path):
        return set(), []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = [{k: (np.nan if v is None else v) for k, v in r.items()} for r in data["rows"]]

    if "completed_pids" in data:
        # 신 형식: 저장된 PID 집합을 그대로 사용
        completed_pids = set(map(str, data["completed_pids"]))
    else:
        # 구 형식 마이그레이션: rows에 있는 PatientID로 완료 집합을 재구성
        completed_pids = {str(int(r["PatientID"])) for r in rows if "PatientID" in r}
        if completed_pids:
            print(f"[체크포인트] 구 형식 감지 → PatientID {len(completed_pids)}명으로 재구성")

    return completed_pids, rows


# ── 엑셀 출력 / xlsx 기반 체크포인트 ─────────────────────────────────────────

def _load_completed_from_xlsx(path: str = OUT_PATH) -> tuple[set, list]:
    """
    xlsx 결과 파일을 읽어 "완료 PID 집합"과 "기존 rows"를 반환한다.
    JSON 체크포인트 대신 xlsx 자체를 재시작 기준으로 사용한다.

    [완료 기준 — "칼럼데이터가 모두 채워진 환자"]
      seg_status == "ok"
      AND liver_upper_slice, pubis_slice 모두 유효 (not NaN)

    → error / missing / no_dicom 등은 완료로 보지 않으므로,
      재시작 시 자동으로 pending_pids 에 포함되어 재처리된다.

    파일이 없거나 읽기 실패 시 (set(), []) 을 반환해 처음부터 시작한다.
    """
    if not os.path.exists(path):
        return set(), []
    try:
        df = pd.read_excel(path)
        if "seg_status" not in df.columns or "PatientID" not in df.columns:
            return set(), []

        # 세 조건 모두 충족한 행만 "완료"
        done_mask = (
            (df["seg_status"].astype(str) == "ok")
            & df["liver_upper_slice"].notna()
            & df["pubis_slice"].notna()
        )
        completed_pids = set(df.loc[done_mask, "PatientID"].astype(int).astype(str))

        # 기존 rows 반환: NaN 은 그대로 유지 (_write_output 에서 처리)
        rows = df.to_dict("records")
        return completed_pids, rows

    except Exception as e:
        print(f"[경고] xlsx 읽기 실패 ({e}). 처음부터 시작합니다.")
        return set(), []


def _write_output(rows: list):
    """
    결과 딕셔너리 리스트를 엑셀(.xlsx)로 저장한다.

    - 동일 PatientID가 중복되면 마지막 결과만 유지한다 (retry 결과 반영).
    - PatientID 오름차순 정렬.
    - Instance Number·슬라이스 수 등 정수 컬럼은 Int64 타입으로 지정해
      NaN이 포함되어도 정수로 표시되게 한다.
    """
    df = pd.DataFrame(rows).drop_duplicates(subset=["PatientID"], keep="last")
    df = df.sort_values("PatientID").reset_index(drop=True)
    for col in ["liver_upper_slice", "pubis_slice", "slices", "n_slices"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")   # pandas nullable integer (NaN 허용)
    col_order = ["PatientID", "n_slices", "slices",
                 "liver_upper_slice", "pubis_slice", "z_range", "z_liver_upper", "z_pubis", "seg_status"]
    cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df[cols].to_excel(OUT_PATH, index=False)


def _print_summary(label: str):
    """최종 결과 파일을 읽어 상태 분포·통계·PNG 수를 콘솔에 출력한다."""
    df = pd.read_excel(OUT_PATH)
    print(f"\n[{label}] {OUT_PATH}")
    print(f"  shape : {df.shape}")
    print(f"  상태  :\n{df['seg_status'].value_counts().to_string()}")
    for col in ("liver_upper_slice", "pubis_slice"):
        if col in df.columns and not df[col].isna().all():
            print(f"  {col}: median={df[col].median():.0f}  std={df[col].std():.1f}")
    print(f"  PNG   : {_count_pngs()}")


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def main():
    """
    [full 모드] 전체 환자를 NUM_WORKERS개 스레드로 병렬 처리한다.

    [체크포인트 방식]
      JSON 파일 대신 xlsx 결과 파일을 직접 읽어 재시작 기준으로 삼는다.
      완료 기준 = seg_status=="ok" AND 주요 컬럼이 모두 채워진 환자.
      error/missing 환자는 완료로 보지 않으므로 재시작 시 자동 재처리된다.

    BATCH_SIZE명마다 xlsx를 중간 저장해 체크포인트 역할을 수행한다.
    """
    for d in (UPPER_DIR, BOTTOM_DIR, MISSING_UPPER_DIR, MISSING_BOTTOM_DIR):
        os.makedirs(d, exist_ok=True)

    folder_map = build_folder_map(DICOM_BASE)
    all_pids = sorted(folder_map.keys(), key=lambda x: int(x))
    total = len(all_pids)

    # xlsx 에서 완료 PID 파악 (JSON 체크포인트 대신)
    completed_pids, rows = _load_completed_from_xlsx()
    n_done = len(completed_pids)
    pending_pids = [pid for pid in all_pids if pid not in completed_pids]

    print(f"DICOM 환자: {total}명  |  스레드: {NUM_WORKERS}  |  출력: {OUT_PATH}")
    if n_done > 0:
        print(f"xlsx 체크포인트: ok {n_done}명 완료 → {len(pending_pids)}명 처리 예정")
    # rows 가 비어 있으면 (첫 실행) 명시 초기화
    if not rows:
        rows = []

    lock = threading.Lock()   # rows·completed_pids 동시 접근 보호
    n_processed = 0           # 이번 실행에서 처리한 수 (pbar.n 은 n_done 오프셋 포함이라 사용 불가)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tqdm(total=total, desc="경계 추출", initial=n_done) as pbar:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                future_to_pid = {
                    executor.submit(extract_bounds, pid, tmp_dir, folder_map): pid
                    for pid in pending_pids
                }

                for future in as_completed(future_to_pid):
                    pid = future_to_pid[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        tqdm.write(f"  [ERROR] PID {pid}: {type(e).__name__}: {e}")
                        result = {
                            "liver_upper_slice": np.nan, "pubis_slice": np.nan,
                            "slices": np.nan,            "n_slices": np.nan,
                            "z_liver_upper": np.nan,     "z_pubis": np.nan,
                            "z_range": np.nan,
                            "seg_status": f"error:{type(e).__name__}:{e}",
                        }
                    result["PatientID"] = int(pid)

                    with lock:
                        rows.append(result)
                        # ok 환자만 completed_pids 에 추가 (진행률 표시 목적)
                        if result.get("seg_status") == "ok":
                            completed_pids.add(pid)
                        pbar.update(1)
                        n_processed += 1

                        # BATCH_SIZE마다 또는 전체 완료 시 xlsx 중간 저장
                        if n_processed % BATCH_SIZE == 0 or n_processed == len(pending_pids):
                            _write_output(rows)
                            status_counts = (
                                pd.Series([r["seg_status"] for r in rows])
                                .value_counts().to_dict()
                            )
                            tqdm.write(f"  [{n_done + n_processed}/{total}] {status_counts}")

    _print_summary("완료")


def retry_failed():
    """
    [retry 모드] 이전 full 실행에서 missing으로 기록된 환자만
    NUM_WORKERS개 스레드로 병렬 재처리한다.

    - 기존 엑셀(OUT_PATH)을 읽어 seg_status가 MISSING_STATUSES에 해당하는 환자만 추출한다.
    - 재처리 결과를 기존 엑셀에 덮어써 최신 상태로 갱신한다.
    - retry 전용 체크포인트를 사용해 중단 시 이어서 실행할 수 있다.
    """
    if not os.path.exists(OUT_PATH):
        raise FileNotFoundError(f"z_bounds 파일 없음: {OUT_PATH}")

    for d in (UPPER_DIR, BOTTOM_DIR, MISSING_UPPER_DIR, MISSING_BOTTOM_DIR):
        os.makedirs(d, exist_ok=True)

    folder_map = build_folder_map(DICOM_BASE)
    df_base = pd.read_excel(OUT_PATH)

    retry_pids = (
        df_base[df_base["seg_status"].astype(str).isin(MISSING_STATUSES)]["PatientID"]
        .astype(str).tolist()
    )
    total = len(retry_pids)
    breakdown = (
        df_base[df_base["seg_status"].astype(str).isin(MISSING_STATUSES)]["seg_status"]
        .value_counts().to_dict()
    )
    print(f"재처리 대상: {total}명 → {breakdown}  |  스레드: {NUM_WORKERS}")
    if total == 0:
        print("재처리할 환자 없음.")
        return

    completed_pids, updated_rows = load_checkpoint(RETRY_CHECKPOINT)
    n_done = len(completed_pids)

    if n_done > 0:
        # 재시작: 이미 완료된 환자 제외, 이전 updated_rows 유지
        pending_pids = [pid for pid in retry_pids if pid not in completed_pids]
        print(f"체크포인트 감지: {n_done}/{total}명 완료, {len(pending_pids)}명 남음")
    else:
        # 처음 실행: updated_rows 초기화
        updated_rows = []
        pending_pids = retry_pids

    lock = threading.Lock()

    def _flush():
        """lock 안에서 호출. updated_rows를 df_base에 병합해 저장한다."""
        df_merged = df_base.copy()
        for row_data in updated_rows:
            pid_val = int(row_data["PatientID"])
            mask = df_merged["PatientID"].astype(int) == pid_val
            if mask.any():
                for col, val in row_data.items():
                    df_merged.loc[mask, col] = val
            else:
                df_merged = pd.concat(
                    [df_merged, pd.DataFrame([row_data])], ignore_index=True
                )
        _write_output(df_merged.to_dict("records"))
        save_checkpoint(completed_pids, updated_rows, RETRY_CHECKPOINT)
        done = len(completed_pids)
        tqdm.write(
            f"  [{done}/{total}] "
            f"{pd.Series([r['seg_status'] for r in updated_rows]).value_counts().to_dict()}"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tqdm(total=total, desc="실패 재처리", initial=n_done) as pbar:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                future_to_pid = {
                    executor.submit(extract_bounds, pid, tmp_dir, folder_map): pid
                    for pid in pending_pids
                }

                for future in as_completed(future_to_pid):
                    pid = future_to_pid[future]
                    tqdm.write(f"\n[재처리] PID {pid}")
                    try:
                        result = future.result()
                    except Exception as e:
                        tqdm.write(f"  [ERROR] PID {pid}: {type(e).__name__}: {e}")
                        result = {
                            "liver_upper_slice": np.nan, "pubis_slice": np.nan,
                            "slices": np.nan,            "n_slices": np.nan,
                            "z_liver_upper": np.nan,     "z_pubis": np.nan,
                            "z_range": np.nan,
                            "seg_status": f"error:{type(e).__name__}:{e}",
                        }
                    result["PatientID"] = int(pid)

                    with lock:
                        updated_rows.append(result)
                        completed_pids.add(pid)
                        pbar.update(1)
                        done = len(completed_pids)

                        if done % BATCH_SIZE == 0 or done == total:
                            _flush()

    if os.path.exists(RETRY_CHECKPOINT):
        os.remove(RETRY_CHECKPOINT)
    _print_summary("재처리 완료")


if __name__ == "__main__":
    if MODE == "retry":
        retry_failed()
    else:
        main()
