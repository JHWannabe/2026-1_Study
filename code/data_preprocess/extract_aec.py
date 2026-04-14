"""
AEC Curve Extraction from DICOM files
- Input : D:/데이터서비스팀 영상제공/강남/강남_axial/{patient}/{series}/*.dcm
- Output: aec_gangnam.xlsx (3 sheets)
    Sheet1 metadata     : PatientID, ManufacturerModelName, SeriesDescription, n_slices, z_range_mm
    Sheet2 raw_aec      : 원본 tube current 벡터 (NaN padding)
    Sheet3 normalized_aec: 128점 physical interpolation 벡터
"""

import os
import numpy as np
import pandas as pd
import pydicom
from scipy.interpolate import interp1d
from tqdm import tqdm

# ── 설정 ─────────────────────────────────────────────────────────────────────
BASE_DIR   = "D:/데이터서비스팀 영상제공/신촌/신촌_axial"
OUTPUT     = "C:/Users/user/Desktop/Study/code/0417/신촌_AEC.xlsx"
N_POINTS   = 128   # 보간 목표 길이 (median of slice counts)
# ─────────────────────────────────────────────────────────────────────────────


def extract_patient(patient_dir: str) -> dict | None:
    """
    환자 폴더에서 AEC curve를 추출한다.
    - SliceLocation 기준 오름차순 정렬
    - XRayTubeCurrent가 없는 슬라이스는 제외
    반환: dict 또는 None (읽기 실패 시)
    """
    dcm_data = []  # (slice_location, tube_current)

    for series in os.listdir(patient_dir):
        series_path = os.path.join(patient_dir, series)
        if not os.path.isdir(series_path):
            continue

        for fname in os.listdir(series_path):
            if not fname.endswith(".dcm"):
                continue
            fpath = os.path.join(series_path, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            except Exception:
                continue

            tube_current = getattr(ds, "XRayTubeCurrent", None)
            slice_loc    = getattr(ds, "SliceLocation", None)

            # SliceLocation 없으면 ImagePositionPatient z값으로 대체
            if slice_loc is None:
                ipp = getattr(ds, "ImagePositionPatient", None)
                if ipp is not None:
                    slice_loc = float(ipp[2])

            if tube_current is None or slice_loc is None:
                continue

            dcm_data.append((float(slice_loc), float(tube_current)))

    if len(dcm_data) < 2:
        return None

    # SliceLocation 오름차순 정렬
    dcm_data.sort(key=lambda x: x[0])
    z_positions   = np.array([d[0] for d in dcm_data])
    tube_currents = np.array([d[1] for d in dcm_data])

    # 메타데이터는 첫 번째 DCM에서 추출 (재사용)
    first_series = os.listdir(patient_dir)[0]
    first_dcm    = sorted(os.listdir(os.path.join(patient_dir, first_series)))[0]
    ds0 = pydicom.dcmread(
        os.path.join(patient_dir, first_series, first_dcm),
        stop_before_pixels=True
    )

    return {
        "PatientID"            : getattr(ds0, "PatientID", ""),
        "ManufacturerModelName": getattr(ds0, "ManufacturerModelName", ""),
        "SeriesDescription"    : getattr(ds0, "SeriesDescription", ""),
        "n_slices"             : len(dcm_data),
        "z_range_mm"           : round(z_positions[-1] - z_positions[0], 2),
        "z_positions"          : z_positions,
        "tube_currents"        : tube_currents,
    }


def physical_interpolate(z: np.ndarray, mA: np.ndarray, n: int) -> np.ndarray:
    """
    SliceLocation(mm) 기준으로 n점 등간격 linear interpolation.
    x축을 물리적 위치로 사용하므로 스캔 범위가 달라도 해부학적 위치가 일치함.
    """
    z_new = np.linspace(z[0], z[-1], n)
    f = interp1d(z, mA, kind="linear", fill_value="extrapolate")
    return f(z_new)


def main():
    patient_dirs = [
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]

    records      = []  # metadata
    raw_aec      = []  # 원본 벡터
    norm_aec     = []  # 166점 보간 벡터
    failed       = []  # 실패 환자

    for p_dir in tqdm(patient_dirs, desc="Processing patients"):
        try:
            result = extract_patient(p_dir)
        except Exception as e:
            failed.append((os.path.basename(p_dir), str(e)))
            continue

        if result is None:
            failed.append((os.path.basename(p_dir), "No valid slices"))
            continue

        # metadata
        records.append({
            "PatientID"            : result["PatientID"],
            "ManufacturerModelName": result["ManufacturerModelName"],
            "SeriesDescription"    : result["SeriesDescription"],
            "n_slices"             : result["n_slices"],
            "z_range_mm"           : result["z_range_mm"],
        })

        # raw AEC (리스트로 저장, 나중에 NaN padding)
        raw_aec.append(result["tube_currents"].tolist())

        # normalized AEC
        # norm_vec = physical_interpolate(
        #     result["z_positions"], result["tube_currents"], N_POINTS
        # )
        # norm_aec.append(norm_vec.tolist())

    # ── DataFrame 생성 ────────────────────────────────────────────────────────

    df_meta = pd.DataFrame(records)

    # raw: 최대 길이에 맞춰 NaN padding
    max_len   = max(len(v) for v in raw_aec)
    raw_padded = [v + [np.nan] * (max_len - len(v)) for v in raw_aec]
    df_raw = pd.DataFrame(
        raw_padded,
        columns=[f"slice_{i}" for i in range(max_len)]
    )
    df_raw.insert(0, "PatientID", df_meta["PatientID"].values)

    # normalized
    # df_norm = pd.DataFrame(
    #     norm_aec,
    #     columns=[f"pt_{i}" for i in range(N_POINTS)]
    # )
    # df_norm.insert(0, "PatientID", df_meta["PatientID"].values)

    # ── Excel 저장 ────────────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT} ...")
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        df_meta.to_excel(writer, sheet_name="metadata",       index=False)
        df_raw .to_excel(writer, sheet_name="raw_aec",        index=False)
        # df_norm.to_excel(writer, sheet_name="normalized_aec", index=False)

    print(f"Done. Processed: {len(records)} patients, Failed: {len(failed)}")

    if failed:
        print("\nFailed patients:")
        for name, reason in failed:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
