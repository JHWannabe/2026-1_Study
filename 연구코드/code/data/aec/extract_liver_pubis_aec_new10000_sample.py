"""
[목적] 새 데이터 10000명 코호트에서 extract_liver_pubis_aec.py의 세그멘테이션·크롭·체성분
계산 로직(liver/hip 랜드마크 → z_bounds → crop → tissue_4_types → VAT/SAT/TAMA/LAMA/NAMA/IMATA,
간→두덩뼈 방향)이 이 코호트에서도 그대로 동작하는지 소수 샘플로 검증한다.

[gangnam/신촌과의 차이 — 이 파일에서만 새로 구현하는 부분]
  - DICOM 폴더: E:\\영상제공\\{site}\\{site}_axial\\{PatientID}\\{series 서브폴더}\\*.dcm 구조가 아니라
    D:\\데이터서비스팀 요청(이홍선)\\{No}_{PatientID}_{date}_CT\\*.dcm 처럼 시리즈 구분 없이
    한 폴더에 여러 시리즈(Arterial/Portal/Delay/Pre 등)의 DICOM이 섞여 있다.
  - 따라서 서브폴더 매칭(find_series_folder) 대신, best_series.xlsx로 미리 골라둔
    환자당 1개 series_desc와 SeriesDescription 태그가 일치하는 파일만 걸러
    InstanceNumber 오름차순으로 정렬해 사용한다.
  - 세그멘테이션(task="total"/liver+hip)과 체성분 계산(compute_body_composition,
    tissue_4_types, NAMA/LAMA HU 구간, aec_1=간쪽 방향 정렬)은 extract_liver_pubis_aec.py의
    함수를 그대로 import해서 재사용한다 — 로직 중복 없음.

[출력] 콘솔에 n_slices, seg_status, VAT/SAT/TAMA_sum_cm2, aec/체성분 슬라이스 곡선 앞부분을 출력한다.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_liver_pubis_aec as base  # noqa: E402  (같은 폴더의 gangnam/신촌 파이프라인 재사용)

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

DICOM_BASE = r"D:\데이터서비스팀 요청(이홍선)"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "새 데이터 10000명"
BEST_SERIES_PATH = DATA_DIR / "best_series.xlsx"

N_SAMPLE = 3


def find_patient_dir(pid: int) -> str | None:
    import glob
    matches = glob.glob(os.path.join(DICOM_BASE, f"*_{pid}_*_CT"))
    return matches[0] if matches else None


def read_slices_for_series(patient_dir: str, target_series_desc: str) -> list[dict] | None:
    """폴더 내 평면 DICOM에서 target_series_desc와 SeriesDescription이 일치하는 파일만 골라
    InstanceNumber 오름차순(간→두덩뼈 방향, base.read_slices와 동일 형식)으로 정렬해 반환."""
    target = base._norm_series_name(target_series_desc)
    rows = []
    for fname in os.listdir(patient_dir):
        fp = os.path.join(patient_dir, fname)
        if not os.path.isfile(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
        except Exception:
            continue
        sd = str(getattr(ds, "SeriesDescription", ""))
        if base._norm_series_name(sd) != target:
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        aec = getattr(ds, "XRayTubeCurrent", None) or getattr(ds, "TubeCurrent", None)
        rows.append({
            "file": fp,
            "inst": int(getattr(ds, "InstanceNumber", 0)),
            "z": float(ipp[2]) if ipp is not None else float("nan"),
            "aec": float(aec) if aec is not None else float("nan"),
        })
    if not rows:
        return None
    rows.sort(key=lambda r: r["inst"])
    return rows


def process_patient_new10000(pid: int, patient_dir: str, target_series_desc: str, tmp_dir: str) -> dict:
    """extract_liver_pubis_aec.process_patient()과 동일 로직이나, 시리즈 서브폴더 대신
    read_slices_for_series()로 이미 걸러진 파일 리스트를 SimpleITK reader에 직접 넣는다."""
    result = {"PatientID": pid, "seg_status": "failed"}

    by_inst = read_slices_for_series(patient_dir, target_series_desc)
    if by_inst is None:
        result["seg_status"] = "series_not_found"
        return result

    hdr = pydicom.dcmread(by_inst[0]["file"], stop_before_pixels=True)
    series_desc = re.sub(r"\s+", " ", str(getattr(hdr, "SeriesDescription", "")).replace("/", "_")).strip()
    manufacturer = str(getattr(hdr, "ManufacturerModelName", ""))
    n = len(by_inst)
    # 키 이름을 base.py의 _write_zbounds/_write_body_composition/_write_body_comp_cropped가
    # 기대하는 스키마(series_description/manufacturer_model)에 맞춰 두면 그 writer들을
    # 그대로 재사용할 수 있다(전체 코호트 실행 스크립트에서 재사용).
    result.update({"series_desc": series_desc, "series_description": series_desc,
                   "manufacturer": manufacturer, "manufacturer_model": manufacturer,
                   "n_slices": n, "inst_min": int(min(r["inst"] for r in by_inst))})

    aec_full = [r["aec"] for r in by_inst]  # aec_1=간 방향, aec_n=두덩뼈 방향
    result["aec_excluded_signal"] = base._is_excluded_signal(base._aec_values(by_inst))
    if not result["aec_excluded_signal"]:
        # gangnam/신촌의 aec_total 시트(크롭 전 전체 슬라이스 곡선)와 동일한 스키마 —
        # base._write_aec_total()이 그대로 재사용 가능. 세그멘테이션 성공 여부와 무관하게
        # (간/두덩뼈 랜드마크를 못 찾아도) series를 찾았으면 저장한다.
        result["aec_total_row"] = {"PatientID": pid, "n_slices": n, "series_desc": series_desc,
                                    "manufacturer": manufacturer, "aec_full": aec_full}

    nii_path = os.path.join(tmp_dir, f"{pid}.nii.gz")
    seg_dir = os.path.join(tmp_dir, f"{pid}_seg")
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([r["file"] for r in by_inst])
        img = reader.Execute()
        sitk.WriteImage(img, nii_path)

        os.makedirs(seg_dir, exist_ok=True)
        with base._silence():
            # nr_thr_resamp/nr_thr_saving 기본값(1/6)은 환자 1명을 단독 실행할 때 기준이라,
            # 여러 환자를 동시에 병렬 처리할 때(extract_liver_pubis_aec_new10000.py)는
            # 호출마다 내부적으로 저장용 서브프로세스가 최대 6개씩 더 생겨 CPU가 과다
            # 구독된다 — 1로 고정해 외부 병렬 워커 수만으로 동시성을 제어한다.
            base.totalsegmentator(input=nii_path, output=seg_dir, task="total",
                                   roi_subset=base.ROI_SUBSET, fast=False, quiet=True, device=base.DEVICE,
                                   nr_thr_resamp=1, nr_thr_saving=1)

        n_k = img.GetSize()[2]
        if n_k != n:
            result["seg_status"] = "invalid_volume_match"
            return result

        import nibabel as nib
        from scipy import ndimage

        hip_k: list[int] = []
        for side in ("hip_left", "hip_right"):
            p = os.path.join(seg_dir, f"{side}.nii.gz")
            if os.path.exists(p):
                idx = np.argwhere(nib.load(p).get_fdata() > 0.5)
                if len(idx) > 0:
                    hip_k.extend(idx[:, 2].astype(int).tolist())
        hip_found = len(hip_k) > 0

        liver_vol = base._load_liver_mask(seg_dir)
        liver_found = False
        liver_k = np.array([], dtype=int)
        if liver_vol is not None:
            binary = liver_vol > 0.5
            labeled, n_comp = ndimage.label(binary)
            if n_comp > 1:
                sizes = ndimage.sum(binary, labeled, list(range(1, n_comp + 1)))
                binary = labeled == int(np.argmax(sizes)) + 1
            # 5mm 슬라이스 두께 탓에 횡격막 부근에서 간과 무관한 흉부 조직이 부분용적효과로
            # 아주 얇은 다리(voxel 수 극소)를 통해 같은 connected component로 이어지는 경우가
            # 있다(실측: 24565번 환자 k=3~4에서 319복셀까지 급감 후 다시 상승) — 슬라이스당
            # 면적이 최대치의 10%(또는 200복셀) 미만인 가장자리는 진짜 간이 아닐 가능성이 커
            # 경계 판정에서 제외한다.
            total_voxels = int(binary.sum())  # MIN_LIVER_VOXELS 판정은 슬라이스 수가 아니라 전체 복셀 수 기준
            per_k = binary.sum(axis=(0, 1))
            liver_k = np.where(per_k >= max(float(per_k.max()) * 0.1, 200))[0] if per_k.max() > 0 else np.array([], dtype=int)
            # gangnam/신촌용 base.py는 median(liver_k) > n_k*0.3(전체 스캔 길이 대비 위치)로
            # 판별하지만, 이 코호트는 series별 FOV 길이가 훨씬 다양해(예: liver가 전체의
            # 15% 지점에 있어도 정상) 절대 위치 기준이 아니라 hip보다 cranial(더 낮은 k)에
            # 있는지 상대 위치로 판별한다.
            liver_found = total_voxels >= base.MIN_LIVER_VOXELS and len(liver_k) > 0
            if liver_found and hip_found:
                liver_found = float(np.median(liver_k)) < float(np.median(hip_k))
        del liver_vol

        if liver_found and hip_found:
            result["seg_status"] = "ok"
        elif liver_found:
            result["seg_status"] = "pubis_missing"
        elif hip_found:
            result["seg_status"] = "liver_missing"
        else:
            result["seg_status"] = "both_missing"

        # 크롭 범위는 항상 "간의 상단(가장 cranial한 지점) ~ 두덩뼈의 하단(가장 caudal한
        # 지점)"이어야 한다. base.py(gangnam/신촌, raw ascending=두덩뼈→간)는 liver_k.max()가
        # 곧 간 상단(가장 높은 k=가장 cranial)이고 hip_k.min()이 곧 두덩뼈 하단(가장 낮은
        # k=가장 caudal)이라 그 공식이 맞지만, 이 코호트는 raw ascending 방향이 반대
        # (간→두덩뼈)라 같은 공식을 쓰면 liver_k.max()가 간의 "하단"(두덩뼈에 가까운 쪽),
        # hip_k.min()이 두덩뼈의 "상단"(간에 가까운 쪽)이 되어 두 마스크 사이의 좁은 틈만
        # 크롭하게 된다 — 방향을 판별해 매번 진짜 간 상단/두덩뼈 하단을 고른다.
        if liver_found and hip_found:
            if float(np.median(liver_k)) < float(np.median(hip_k)):
                apex_k, bottom_k = int(liver_k.min()), int(max(hip_k))   # 간→두덩뼈 방향
            else:
                apex_k, bottom_k = int(liver_k.max()), int(min(hip_k))   # 두덩뼈→간 방향 (gangnam과 동일)
        else:
            apex_k = bottom_k = None
        if apex_k is None or bottom_k is None:
            if result["seg_status"] == "ok":
                result["seg_status"] = "invalid_bounds"
            return result

        result["liver_upper_k"] = apex_k
        result["pubis_k"] = bottom_k
        result["liver_upper_slice"] = by_inst[apex_k]["inst"]
        result["pubis_slice"] = by_inst[bottom_k]["inst"]
        result["z_liver_upper"] = by_inst[apex_k]["z"]
        result["z_pubis"] = by_inst[bottom_k]["z"]
        if pd.notna(result["z_liver_upper"]) and pd.notna(result["z_pubis"]):
            result["z_range"] = round(abs(result["z_liver_upper"] - result["z_pubis"]), 2)

        lo, hi = min(apex_k, bottom_k), max(apex_k, bottom_k)
        result["slices"] = hi - lo + 1
        if not result["aec_excluded_signal"]:
            result["aec_cropped_liver_to_pubis"] = aec_full[lo:hi + 1]

        lo_k, hi_k = min(apex_k, bottom_k), max(apex_k, bottom_k)
        bc = base.compute_body_composition(img, lo_k, hi_k, tmp_dir, pid, reverse=apex_k > bottom_k)
        result["body_composition"] = bc

    except Exception as e:
        result["seg_status"] = f"error:{type(e).__name__}:{e}"
    finally:
        if os.path.isfile(nii_path):
            os.remove(nii_path)
        if os.path.isdir(seg_dir):
            import shutil
            shutil.rmtree(seg_dir, ignore_errors=True)

    return result


def main():
    best = pd.read_excel(BEST_SERIES_PATH).head(N_SAMPLE)
    print(f"샘플 {len(best)}명 검증 시작")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for _, r in best.iterrows():
            pid = int(r["PatientID"])
            patient_dir = find_patient_dir(pid)
            print(f"\n{'='*60}\nPID {pid}  series_desc(선정)={r['series_desc']!r}")
            if patient_dir is None:
                print("  CT 폴더 없음")
                continue
            print(f"  폴더: {patient_dir}")

            result = process_patient_new10000(pid, patient_dir, r["series_desc"], tmp_dir)
            print(f"  n_slices={result.get('n_slices')}  seg_status={result['seg_status']}")
            if result["seg_status"] == "ok":
                print(f"  liver_upper_k={result['liver_upper_k']}  pubis_k={result['pubis_k']}  "
                      f"slices(crop)={result['slices']}  z_range={result.get('z_range')}mm")
                aec = result.get("aec_cropped_liver_to_pubis")
                print(f"  aec_cropped 앞 5개(간쪽부터): {aec[:5] if aec else None}")
                bc = result.get("body_composition", {})
                print(f"  VAT_sum_cm2={bc.get('VAT_sum_cm2')}  SAT_sum_cm2={bc.get('SAT_sum_cm2')}  "
                      f"NAMA_sum_cm2={bc.get('NAMA_sum_cm2')}  LAMA_sum_cm2={bc.get('LAMA_sum_cm2')}  "
                      f"IMATA_sum_cm2={bc.get('IMATA_sum_cm2')}")
                vfa = bc.get("VFA_slices")
                print(f"  VFA(VAT) 슬라이스 곡선 앞 5개(간쪽부터): {vfa[:5] if vfa else None}")


if __name__ == "__main__":
    main()
