"""
check_landmarks.py의 랜드마크 QC(11개 core anchor)를 새 데이터 10000명 코호트에 대해서도
실행해 new10000_landmarks.xlsx로 저장한다.

[신촌/강남과의 차이] DICOM 폴더에 시리즈 구분 없이 여러 series가 섞여 있으므로,
extract_liver_pubis_aec_new10000_sample.py의 find_patient_dir/read_slices_for_series로
이미 골라둔 series_desc(best_series.xlsx)만 걸러 InstanceNumber 오름차순으로 사용한다.
anchor 탐지 로직(compute_landmarks)과 xlsx append/체크포인트는 check_landmarks.py를 그대로 재사용한다.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_landmarks as cl  # noqa: E402
from extract_liver_pubis_aec_new10000_sample import (  # noqa: E402
    find_patient_dir, read_slices_for_series,
)

import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "새 데이터 10000명"
FINAL_DATASET_PATH = DATA_DIR / "new10000_final_dataset.xlsx"
BEST_SERIES_PATH = DATA_DIR / "best_series.xlsx"

PATHS = {
    "landmarks": str(DATA_DIR / "new10000_landmarks.xlsx"),
    "checkpoint": str(DATA_DIR / "new10000_landmarks_checkpoint.pkl"),
}
BATCH_SIZE = 5


def process_patient(pid: int, patient_dir: str, series_desc: str, tmp_dir: str) -> dict:
    by_inst = read_slices_for_series(patient_dir, series_desc)
    if by_inst is None:
        return cl._empty_landmark_row(pid, None, None, "series_not_found")

    hdr = pydicom.dcmread(by_inst[0]["file"], stop_before_pixels=True)
    series_desc_clean = cl._clean_ws(str(getattr(hdr, "SeriesDescription", "")))
    manufacturer = str(getattr(hdr, "ManufacturerModelName", ""))
    n = len(by_inst)
    row = cl._empty_landmark_row(pid, series_desc_clean, manufacturer, "failed")
    row["n_slices"] = n

    nii_path = os.path.join(tmp_dir, f"{pid}.nii.gz")
    seg_dir = os.path.join(tmp_dir, f"{pid}_seg")
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([r["file"] for r in by_inst])
        img = reader.Execute()
        sitk.WriteImage(img, nii_path)

        n_k = img.GetSize()[2]
        if n_k != n:
            row["seg_status"] = "invalid_volume_match"
            return row

        os.makedirs(seg_dir, exist_ok=True)
        with cl._silence():
            cl.totalsegmentator(input=nii_path, output=seg_dir, task="total",
                                roi_subset=cl.ROI_SUBSET, fast=False, quiet=True, device=cl.DEVICE,
                                nr_thr_resamp=1, nr_thr_saving=1)

        cl.compute_landmarks(row, seg_dir, nii_path, n_k, by_inst)

    except Exception as e:
        row["seg_status"] = f"error:{type(e).__name__}:{e}"
    finally:
        if os.path.isfile(nii_path):
            os.remove(nii_path)
        if os.path.isdir(seg_dir):
            import shutil
            shutil.rmtree(seg_dir, ignore_errors=True)

    return row


def main():
    meta_ids = pd.read_excel(FINAL_DATASET_PATH, sheet_name="metadata", usecols=["PatientID"])
    best = pd.read_excel(BEST_SERIES_PATH, usecols=["PatientID", "series_desc"])
    targets = best.merge(meta_ids, on="PatientID")

    done_pids = set()
    if os.path.exists(PATHS["landmarks"]):
        done_pids |= set(pd.read_excel(PATHS["landmarks"])["PatientID"].astype(int))
    attempted = cl._load_checkpoint(PATHS["checkpoint"])
    done_pids |= attempted

    pending = targets[~targets["PatientID"].isin(done_pids)]
    print(f"대상 {len(targets)}명 | 완료 {len(done_pids)}명 | 처리 예정 {len(pending)}명")

    rows: list[dict] = []
    n_processed = 0
    if not pending.empty:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for _, r in tqdm(pending.iterrows(), total=len(pending), desc="new10000 랜드마크 QC"):
                pid = int(r["PatientID"])
                patient_dir = find_patient_dir(pid)
                if patient_dir is None:
                    rows.append(cl._empty_landmark_row(pid, None, None, "no_dicom"))
                else:
                    try:
                        row = process_patient(pid, patient_dir, r["series_desc"], tmp_dir)
                    except Exception as e:
                        row = cl._empty_landmark_row(pid, None, None, f"error:{type(e).__name__}:{e}")
                    rows.append(row)
                    tqdm.write(f"  PID {pid}: {row['seg_status']}")

                attempted.add(pid)
                n_processed += 1
                if n_processed % BATCH_SIZE == 0 or n_processed == len(pending):
                    cl._write_landmarks(PATHS, rows)
                    cl._save_checkpoint(PATHS["checkpoint"], attempted)
                    rows = []
                    tqdm.write(f"  [체크포인트 저장 | {n_processed}/{len(pending)}]")

    if os.path.exists(PATHS["checkpoint"]):
        os.remove(PATHS["checkpoint"])
    print("완료")


if __name__ == "__main__":
    main()
