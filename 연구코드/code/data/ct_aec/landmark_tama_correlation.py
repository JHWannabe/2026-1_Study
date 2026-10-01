"""
{SITE}_z_bounds.xlsx 에 이미 계산된 pubis~liver 구간(k-index 범위) 내에서,
뼈에서 뽑아낼 수 있는 "표준화된 landmark"별로 그 위치 한 슬라이스의 TAMA(=NAMA+LAMA, 근육 단면적)를
구하고, 환자 전체(Volumetric) 근육량인 metadata의 "총근육량" 컬럼과의 correlation을 landmark별로 비교한다.

[배경]
  단순히 크롭 구간을 128등분(aec_128과 동일한 방식) 하면 환자마다 척추 레벨이 다른 위치에 오게 되어
  "표준화"가 되지 않는다. 대신 TotalSegmentator로 자동 검출 가능한 뼈 기준점을 표준 landmark로 삼는다.
    - 흉추/요추/천추: T10 ~ T12, L1 ~ L5, S1  (vertebrae_* 클래스, 단면적이 가장 큰 슬라이스)
                      T7~T9는 간 상단(liver_upper_k)보다 위쪽인 경우가 많아 pubis~liver 크롭 범위에
                      거의 걸리지 않으므로 landmark 목록에서 제외
    - femoral_head : femur_left/right 단면적이 가장 큰 슬라이스 (대퇴골두 근사)
    - pubis        : 크롭 구간의 pubis측 경계 슬라이스 그대로 사용
                      (z_bounds의 pubis_k 자체가 "hip 최하단 = inferior pubic margin"으로 정의되어 있음)

[방법]
  1) z_bounds.xlsx의 series_description/manufacturer_model로 원 세그멘테이션에 쓰인 시리즈를 다시 찾아
     NIfTI로 변환 → pubis_k~liver_upper_k 구간으로 볼륨을 크롭
  2) TotalSegmentator "total" task를 위 landmark들의 roi_subset으로 실행해 각 landmark의 대표 슬라이스(k)를 탐지
  3) TotalSegmentator "tissue_4_types" task로 skeletal_muscle을 얻고, HU로 NAMA(+30~+150)/LAMA(-30~+30)를
     재분류해 슬라이스별 TAMA(=NAMA+LAMA) 배열을 구함
  4) 각 landmark의 대표 슬라이스 위치에서 TAMA 값을 샘플링
  5) landmark별 단일 상관(correlation) 비교뿐 아니라, "실제 volume prediction"까지 확인하기 위해
     landmark 조합(1장, 2장, ... n장)으로 선형회귀를 돌려 "총근육량"을 얼마나 잘 예측하는지
     (5-fold cross-validated R²) 조합 크기별 최선의 조합을 exhaustive search로 찾는다.
     → "total volume을 구하려면 몇 장이면 충분한가"에 대한 답 (R²가 포화되는 지점)

[출력] (모두 CT_AEC_process/landmark_tama/ 폴더 하나에 모아서 저장)
  {SITE}_landmark_tama.xlsx              : PatientID별 landmark별 (k, TAMA) + seg_status
  landmark_tama_correlation_summary.xlsx : 강남+신촌 합쳐 landmark별 "총근육량"과의 Pearson/Spearman correlation
  landmark_tama_correlation.png          : 위 correlation을 landmark 순서(T10→S1→femoral_head→pubis)로 시각화
  landmark_tama_best_subset.xlsx         : 조합 크기(1~BEST_SUBSET_MAX_K장)별 "총근육량" 예측 최적 조합 + CV R²
  landmark_tama_best_subset.png          : 조합 크기별 CV R² 곡선 (몇 장부터 충분한지 시각 확인용)
"""

import os
import io
import sys
import re
import shutil
import pickle
import logging
import tempfile
import textwrap
import itertools
import contextlib
from difflib import SequenceMatcher
from typing import cast

import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import SimpleITK as sitk
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
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

SITE = "강남"   # "강남" 으로 바꿔서 한 번 더 돌려야 두 사이트 모두 correlation 계산에 반영됨
SITE_EN = {"강남": "Gangnam", "신촌": "Sinchon"}[SITE]
SITES = ("강남", "신촌")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR      = PROJECT_ROOT
RESULT_DIR   = rf"{OUT_DIR}\results\landmark_tama"   # 이 스크립트의 산출물은 전부 이 폴더 하나에 모음
DICOM_BASE   = rf"E:\영상제공\{SITE}\{SITE}_axial"
DATA_DIR     = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}"
ZBOUNDS_PATH = rf"{DATA_DIR}\aec\{SITE}_z_bounds.xlsx"
MERGED_PATH  = rf"{OUT_DIR}\data\{SITE_EN}\{SITE_EN}.xlsx"
OUT_PATH     = rf"{RESULT_DIR}\{SITE}_landmark_tama.xlsx"

TEST_N = None   # 정수 지정 시 앞 N명만 처리 (파일럿 검증용)

CORR_OUT_PATH  = rf"{RESULT_DIR}\landmark_tama_correlation_summary.xlsx"
CORR_PLOT_PATH = rf"{RESULT_DIR}\landmark_tama_correlation.png"

BEST_SUBSET_OUT_PATH  = rf"{RESULT_DIR}\landmark_tama_best_subset.xlsx"
BEST_SUBSET_PLOT_PATH = rf"{RESULT_DIR}\landmark_tama_best_subset.png"
BEST_SUBSET_MAX_K     = 8    # 조합 크기(슬라이스 개수) 최대값. exhaustive search이므로 너무 크게 잡지 말 것
BEST_SUBSET_MIN_N     = 4 if TEST_N is not None else 15   # 조합별 최소 표본 수 (CV 폴드당 2표본 이상 필요). TEST_N 파일럿 모드에서는 완화
BEST_SUBSET_CV_FOLDS  = 5

PLOT_FONT_SCALE = 3   # 그래프 폰트 크기를 matplotlib 기본값(10pt) 대비 몇 배로 키울지

CHECKPOINT_PATH = rf"{RESULT_DIR}\{SITE}_landmark_tama_checkpoint.pkl"

TASK      = "tissue_4_types"
VERT_TASK = "total"          # landmark 위치 탐지용
NAMA_HU   = (30, 150)        # 정상감쇠근육 (Normal Attenuation Muscle Area)
LAMA_HU   = (-30, 30)        # 저감쇠근육   (Low Attenuation Muscle Area)

VERT_LEVELS    = ["T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1"]
# T7~T9는 간 상단(liver_upper_k)보다 위쪽일 때가 많아 pubis~liver 크롭 범위에 거의 걸리지 않으므로 제외
SEG_LANDMARKS  = VERT_LEVELS + ["femoral_head"]          # 단면적 최대 슬라이스로 탐지
ALL_LANDMARKS  = SEG_LANDMARKS + ["pubis"]                # pubis는 크롭 경계(pubis_k) 그대로 사용
BONE_ROI       = [f"vertebrae_{lv}" for lv in VERT_LEVELS] + ["femur_left", "femur_right"]

BATCH_SIZE  = 1      # 몇 명 처리마다 체크포인트/중간 저장을 할지
MERGE_EVERY = 10     # 몇 명 처리마다 correlation 요약을 다시 계산할지


@contextlib.contextmanager
def _silence():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# ── DICOM 폴더/시리즈 매칭 (total_crop_aec.py / body_composition_L3.py 와 동일한 규칙) ──

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
    result = {"PatientID": pid, "seg_status": "failed", "n_slices_range": np.nan}
    for lm in ALL_LANDMARKS:
        result[f"{lm}_k"]    = np.nan
        result[f"{lm}_TAMA"] = np.nan
    return result


def _landmark_mask_paths(seg_dir: str) -> dict[str, list[str]]:
    """landmark 이름 → 합쳐야 할 마스크 파일 경로 목록."""
    paths: dict[str, list[str]] = {}
    for lv in VERT_LEVELS:
        paths[lv] = [os.path.join(seg_dir, f"vertebrae_{lv}.nii.gz")]
    paths["femoral_head"] = [
        os.path.join(seg_dir, "femur_left.nii.gz"),
        os.path.join(seg_dir, "femur_right.nii.gz"),
    ]
    return paths


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

    cropped_nii_path = os.path.join(tmp_dir, f"{pid}_cropped.nii.gz")
    tissue_seg_dir   = os.path.join(tmp_dir, f"{pid}_tissue_seg")
    land_seg_dir     = os.path.join(tmp_dir, f"{pid}_land_seg")

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

        pubis_k, liver_k = int(zrow["pubis_k"]), int(zrow["liver_upper_k"])
        lo_k = max(0, min(pubis_k, liver_k, n_k - 1))
        hi_k = max(0, min(max(pubis_k, liver_k), n_k - 1))
        if hi_k < lo_k:
            result["seg_status"] = "invalid_bounds"
            return result

        size  = [img.GetSize()[0], img.GetSize()[1], hi_k - lo_k + 1]
        index = [0, 0, lo_k]
        cropped_img = sitk.RegionOfInterest(img, size, index)
        sitk.WriteImage(cropped_img, cropped_nii_path)

        n_slices_range = hi_k - lo_k + 1
        result["n_slices_range"] = n_slices_range

        # pubis landmark: z_bounds의 pubis_k 자체가 "hip 최하단"으로 정의돼 있으므로
        # 크롭 경계 중 pubis 쪽 인덱스를 그대로 사용 (재탐지 불필요)
        pubis_local_k = 0 if pubis_k <= liver_k else n_slices_range - 1
        result["pubis_k"] = pubis_local_k

        # ── 1) landmark 위치 탐지 (vertebrae T10~S1, femoral_head) ──
        os.makedirs(land_seg_dir, exist_ok=True)
        with _silence():
            totalsegmentator(input=cropped_nii_path, output=land_seg_dir, task=VERT_TASK,
                             roi_subset=BONE_ROI, fast=False, quiet=True, device=DEVICE)

        landmark_k: dict[str, int] = {}
        for lm, mask_paths in _landmark_mask_paths(land_seg_dir).items():
            combined = None
            for p in mask_paths:
                if not os.path.exists(p):
                    continue
                mask = cast(Nifti1Image, nib.load(p)).get_fdata() > 0.5
                combined = mask if combined is None else (combined | mask)
            if combined is None or not np.any(combined):
                continue
            area_per_slice = combined.sum(axis=(0, 1))
            if np.any(area_per_slice > 0):
                landmark_k[lm] = int(np.argmax(area_per_slice))

        # ── 2) 슬라이스별 TAMA(=NAMA+LAMA) 배열 ──
        os.makedirs(tissue_seg_dir, exist_ok=True)
        with _silence():
            totalsegmentator(input=cropped_nii_path, output=tissue_seg_dir, task=TASK,
                             fast=False, quiet=True, device=DEVICE)

        muscle_path = os.path.join(tissue_seg_dir, "skeletal_muscle.nii.gz")
        if not os.path.exists(muscle_path):
            warnings.append("skeletal_muscle: 세그멘테이션 결과 없음")
            result["seg_status"] = "tissue_seg_missing"
            return result
        muscle_mask = cast(Nifti1Image, nib.load(muscle_path)).get_fdata() > 0.5

        cropped_nii = cast(Nifti1Image, nib.load(cropped_nii_path))
        hu_data     = cropped_nii.get_fdata()
        zooms = cast(tuple[float, ...], cropped_nii.header.get_zooms())
        pixel_area_cm2 = float(zooms[0]) * float(zooms[1]) / 100.0

        nama_mask = muscle_mask & (hu_data >= NAMA_HU[0]) & (hu_data <= NAMA_HU[1])
        lama_mask = muscle_mask & (hu_data >= LAMA_HU[0]) & (hu_data <= LAMA_HU[1])
        tama_per_slice = (nama_mask.sum(axis=(0, 1)) + lama_mask.sum(axis=(0, 1))) * pixel_area_cm2

        # ── 3) landmark 위치에서 TAMA 샘플링 ──
        if 0 <= pubis_local_k < n_slices_range:
            result["pubis_TAMA"] = round(float(tama_per_slice[pubis_local_k]), 2)
        for lm, k in landmark_k.items():
            result[f"{lm}_k"]    = k
            result[f"{lm}_TAMA"] = round(float(tama_per_slice[k]), 2)

        result["seg_status"] = "ok"

    except Exception as e:
        result["seg_status"] = f"error:{type(e).__name__}:{e}"

    finally:
        for path in (cropped_nii_path, tissue_seg_dir, land_seg_dir):
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
    col_order = ["PatientID", "seg_status", "n_slices_range"]
    for lm in ALL_LANDMARKS:
        col_order += [f"{lm}_k", f"{lm}_TAMA"]
    cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df[cols].to_excel(OUT_PATH, sheet_name="landmark_tama", index=False)


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


# ── 공용: 강남+신촌 landmark_tama.xlsx + 총근육량 병합 ─────────────────────────

def _load_combined_df() -> pd.DataFrame | None:
    """양쪽 사이트의 {SITE}_landmark_tama.xlsx와 metadata의 "총근육량"(Volumetric TAMA)을 병합."""
    frames = []
    for site in SITES:
        site_en       = SITE_EN if site == SITE else {"강남": "Gangnam", "신촌": "Sinchon"}[site]
        landmark_path = rf"{RESULT_DIR}\{site}_landmark_tama.xlsx"
        merged_path   = rf"{OUT_DIR}\data\{site_en}\{site_en}.xlsx"
        if not (os.path.exists(landmark_path) and os.path.exists(merged_path)):
            print(f"[landmark] {site}: {os.path.basename(landmark_path)} 없음 → 건너뜀")
            continue

        land_df = pd.read_excel(landmark_path, sheet_name="landmark_tama")
        land_df = land_df[land_df["seg_status"] == "ok"]
        meta_df = pd.read_excel(merged_path, sheet_name="metadata")[["PatientID", "총근육량"]]
        merged  = land_df.merge(meta_df, on="PatientID", how="inner")
        merged["site"] = site
        frames.append(merged)

    if not frames:
        print("[landmark] 계산할 데이터가 없습니다 (양쪽 사이트 landmark_tama.xlsx 필요).")
        return None

    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["총근육량"])


def _apply_plot_style() -> None:
    """그래프에서 한글이 네모(□)로 깨지지 않도록 한글 폰트를 지정하고,
    폰트 크기를 matplotlib 기본값 대비 PLOT_FONT_SCALE배로 키운다."""
    import matplotlib

    base = 10 * PLOT_FONT_SCALE
    matplotlib.rcParams.update({
        "font.family":      "Malgun Gothic",   # Windows 기본 한글 폰트
        "axes.unicode_minus": False,           # 한글 폰트 사용 시 마이너스 기호 깨짐 방지
        "font.size":        base,
        "axes.titlesize":   base * 1.2,
        "axes.labelsize":   base,
        "xtick.labelsize":  base * 0.8,
        "ytick.labelsize":  base * 0.8,
        "legend.fontsize":  base * 0.8,
    })


def _wrap_title(title: str, width: int = 28) -> str:
    """폰트가 커진 만큼 제목이 figure 밖으로 넘치지 않도록 여러 줄로 감싼다."""
    return "\n".join(textwrap.wrap(title, width=width))


# ── 1) landmark별 단순 correlation ────────────────────────────────────────────

def compute_correlation(df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """landmark별 슬라이스 TAMA 하나와 "총근육량"(Volumetric TAMA) 간 Pearson/Spearman correlation."""
    if df is None:
        df = _load_combined_df()
    if df is None:
        return None
    os.makedirs(RESULT_DIR, exist_ok=True)

    rows = []
    for lm in ALL_LANDMARKS:
        col = f"{lm}_TAMA"
        if col not in df.columns:
            continue
        sub = df[[col, "총근육량"]].dropna()
        if len(sub) < 3:
            rows.append({"landmark": lm, "n": len(sub), "pearson_r": np.nan,
                        "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan})
            continue
        pr, pp = cast(tuple[float, float], stats.pearsonr(sub[col], sub["총근육량"]))
        sr, sp = cast(tuple[float, float], stats.spearmanr(sub[col], sub["총근육량"]))
        rows.append({"landmark": lm, "n": len(sub), "pearson_r": round(float(pr), 4), "pearson_p": float(pp),
                    "spearman_r": round(float(sr), 4), "spearman_p": float(sp)})

    summary = pd.DataFrame(rows).sort_values("pearson_r", ascending=False, key=abs).reset_index(drop=True)
    summary.to_excel(CORR_OUT_PATH, sheet_name="correlation_summary", index=False)
    print(f"\n[correlation] 저장 완료: {CORR_OUT_PATH}")
    print(summary.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _apply_plot_style()

        order = [lm for lm in ALL_LANDMARKS if lm in summary["landmark"].values]
        plot_df = summary.set_index("landmark").loc[order].reset_index()
        fig, ax = plt.subplots(figsize=(18, 11))
        ax.bar(plot_df["landmark"], plot_df["pearson_r"])
        ax.set_ylabel("Pearson r\n(landmark TAMA vs 총근육량)")
        fig.suptitle(_wrap_title(f"Landmark별 TAMA - 총근육량(Volumetric) Correlation ({'+'.join(SITES)})"))
        ax.axhline(0, color="black", linewidth=0.8)
        plt.xticks(rotation=45)
        fig.tight_layout()
        fig.savefig(CORR_PLOT_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[correlation] 그래프 저장 완료: {CORR_PLOT_PATH}")
    except Exception as e:
        print(f"[correlation] 그래프 저장 생략 ({type(e).__name__}: {e})")

    return summary


# ── 2) landmark 조합(1장 ~ n장)으로 실제 volume 예측 (best subset) ─────────────

def _cv_r2(X: np.ndarray, y: np.ndarray) -> float:
    # 폴드마다 테스트 표본이 최소 2개는 있어야 R²가 정의됨 (1개면 분산이 0이라 nan)
    n_splits = min(BEST_SUBSET_CV_FOLDS, len(y) // 2)
    if n_splits < 2:
        return np.nan
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    if np.any(np.isnan(scores)):
        return np.nan
    return float(np.mean(scores))


def compute_best_subset(df: pd.DataFrame | None = None, max_k: int = BEST_SUBSET_MAX_K) -> pd.DataFrame | None:
    """landmark 조합 크기(1장, 2장, ... max_k장)별로, "총근육량"(Volumetric TAMA)을 가장 잘 예측하는
    조합을 exhaustive search + 5-fold cross-validated R²로 찾는다.
    → "Best 1 slice", "Best n slice 조합", "total volume 구하는 데 몇 장이면 충분한가"에 대한 답."""
    if df is None:
        df = _load_combined_df()
    if df is None:
        return None
    os.makedirs(RESULT_DIR, exist_ok=True)

    candidates = [lm for lm in ALL_LANDMARKS if f"{lm}_TAMA" in df.columns]
    max_k = min(max_k, len(candidates))

    rows = []
    for k in range(1, max_k + 1):
        best = None
        for combo in itertools.combinations(candidates, k):
            cols = [f"{lm}_TAMA" for lm in combo]
            sub = df[cols + ["총근육량"]].dropna()
            n = len(sub)
            if n < max(BEST_SUBSET_MIN_N, k + 3):
                continue
            r2 = _cv_r2(np.asarray(sub[cols].values), np.asarray(sub["총근육량"].values))
            if np.isnan(r2):
                continue
            if best is None or r2 > best["cv_r2"]:
                best = {"n_slices": k, "landmarks": "+".join(combo), "n_patients": n, "cv_r2": round(r2, 4)}
        if best:
            rows.append(best)
            tqdm.write(f"  [best-subset] {k}장: {best['landmarks']} "
                       f"(n={best['n_patients']}, CV R²={best['cv_r2']})")
        else:
            tqdm.write(f"  [best-subset] {k}장: 표본 수 부족(<{BEST_SUBSET_MIN_N}, 폴드당 2표본 미만 포함)으로 후보 없음")

    if not rows:
        print("[best-subset] 계산할 수 있는 조합이 없습니다 (표본 수 부족). "
             "TEST_N 파일럿 모드에서는 흔한 상황이며, 전체 데이터로 돌리면 해결됩니다.")
        return None

    summary = pd.DataFrame(rows)
    summary.to_excel(BEST_SUBSET_OUT_PATH, sheet_name="best_subset", index=False)
    print(f"\n[best-subset] 저장 완료: {BEST_SUBSET_OUT_PATH}")
    print(summary.to_string(index=False))

    valid = summary.dropna(subset=["cv_r2"])
    if valid.empty:
        print("\n[best-subset] 유효한 CV R²가 없어 최적 조합을 정할 수 없습니다.")
    else:
        best_row = valid.loc[valid["cv_r2"].idxmax()]
        best_r2  = float(cast(float, best_row["cv_r2"]))
        print(f"\n[best-subset] 최고 CV R² = {best_row['cv_r2']} ({best_row['n_slices']}장: {best_row['landmarks']})")
        if best_r2 <= 0:
            print("[best-subset] CV R²가 0 이하라 아직 예측력이 없습니다 "
                 "(표본 수가 너무 적거나 landmark가 예측에 충분치 않을 수 있음) → '몇 장이면 충분한지'는 판단 보류")
        else:
            # R²가 음수인 조합도 섞여있을 수 있어 절대값 기준 대신 (최고 - 여유폭) 기준으로 판단
            threshold = best_r2 - 0.1 * abs(best_r2)
            plateau = valid[valid["cv_r2"] >= threshold].iloc[0]
            print(f"[best-subset] 최고치의 90% 이상을 처음 달성하는 지점 = {plateau['n_slices']}장 "
                 f"(CV R²={plateau['cv_r2']}, {plateau['landmarks']}) → total volume 추정에 대략 이 정도 슬라이스면 충분")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _apply_plot_style()

        fig, ax = plt.subplots(figsize=(16, 11))
        ax.plot(summary["n_slices"], summary["cv_r2"], marker="o", markersize=10, linewidth=3)
        ax.set_xlabel("조합에 사용한 landmark(슬라이스) 개수")
        ax.set_ylabel("5-fold CV R²\n(예측 vs 총근육량)")
        fig.suptitle(_wrap_title(f"조합 크기별 total volume 예측 성능 ({'+'.join(SITES)})"))
        ax.set_ylim(top=1.0)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(BEST_SUBSET_PLOT_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[best-subset] 그래프 저장 완료: {BEST_SUBSET_PLOT_PATH}")
    except Exception as e:
        print(f"[best-subset] 그래프 저장 생략 ({type(e).__name__}: {e})")

    return summary


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
        with tqdm(total=len(zdf), initial=len(processed), desc="Landmark TAMA 추출") as pbar:
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
                        combined = _load_combined_df()
                        compute_correlation(combined)
                        compute_best_subset(combined)
                    except Exception as e:
                        tqdm.write(f"  [correlation/best-subset 경고] {type(e).__name__}: {e}")

    _write_output(results)
    counts = pd.Series([r["seg_status"] for r in results]).value_counts().to_dict()
    print(f"\n[완료] 총 {len(results)}명  |  {counts}  |  {OUT_PATH}")

    # 두 사이트 모두 landmark_tama.xlsx가 준비돼 있어야 의미있는 correlation/best-subset이 나옴
    combined = _load_combined_df()
    compute_correlation(combined)
    compute_best_subset(combined)


if __name__ == "__main__":
    main()
