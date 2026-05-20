"""
DICOM 파일에서 직접 AEC(XRayTubeCurrent) 값을 읽어 T12~두덩뼈 구간으로 crop한 뒤
128/256포인트로 선형 보간한다.

배경:
  - DICOM 정렬: ImagePositionPatient z좌표 오름차순 (caudal=낮은z=낮은idx → cranial=높은z=높은idx)
  - z_bounds의 t12_slice / pubis_slice: DICOM Instance Number (cranial-to-caudal 취득)
  - 변환식: z_sorted_idx = inst_max - instance + 1
      inst_max = 해당 환자의 최대 Instance Number (DCM에서 직접 읽음)
      T12 (cranial) → 높은 z_sorted_idx
      두덩뼈 (caudal) → 낮은 z_sorted_idx

출력: 강남_aec_cropped.xlsx
  - 기본 시트 : No, PatientID, n_slices_cropped, z_range, aec_1 ~ aec_N
  - aec_128  : No, PatientID, n_slices_cropped, z_range, aec_1 ~ aec_128 (128포인트 보간)
  - aec_256  : No, PatientID, n_slices_cropped, z_range, aec_1 ~ aec_256 (256포인트 보간)
"""

import os
import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm

SITE       = "강남"
DICOM_BASE = rf"D:\영상제공\{SITE}\{SITE}_axial"
DATA_BASE  = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}"

Z_BOUNDS_PATH    = rf"{DATA_BASE}\aec\{SITE}_z_bounds.xlsx"
Z_BOUNDS_OK_PATH = rf"{DATA_BASE}\aec\{SITE}_z_bounds_ok.xlsx"
OUT_PATH         = rf"{DATA_BASE}\aec\{SITE}_aec_cropped.xlsx"
OUT_OK_PATH      = rf"{DATA_BASE}\aec\{SITE}_aec_cropped_ok.xlsx"

INTERP_SIZES = [128, 256]
BATCH_SIZE   = 20


def build_folder_map(dicom_base: str) -> dict[int, str]:
    """PatientID(int) → DICOM 시리즈 폴더 경로."""
    folder_map: dict[int, str] = {}
    for name in os.listdir(dicom_base):
        parts = name.split("_")
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        patient_dir = os.path.join(dicom_base, name)
        # 하위 시리즈 폴더 중 DCM 파일이 가장 많은 것 선택
        best = None
        best_count = 0
        for sub in os.listdir(patient_dir):
            sub_path = os.path.join(patient_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            n = len([f for f in os.listdir(sub_path) if f.endswith(".dcm")])
            if n > best_count:
                best_count = n
                best = sub_path
        if best:
            folder_map[pid] = best
    return folder_map


def read_patient_aec(series_dir: str) -> pd.DataFrame | None:
    """시리즈 폴더의 모든 DCM을 읽어 z 오름차순 정렬된 DataFrame 반환.

    컬럼: inst(InstanceNumber), z(ImagePositionPatient z), aec(XRayTubeCurrent)
    """
    rows = []
    for fname in os.listdir(series_dir):
        if not fname.endswith(".dcm"):
            continue
        fpath = os.path.join(series_dir, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            z   = float(ipp[2]) if ipp is not None else float("nan")
            inst = int(getattr(ds, "InstanceNumber", 0))
            aec  = getattr(ds, "XRayTubeCurrent", None)
            if aec is None:
                aec = getattr(ds, "TubeCurrent", None)
            rows.append({"inst": inst, "z": z, "aec": aec})
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("z").reset_index(drop=True)
    return df


def interp_row(vals: np.ndarray, n: int) -> np.ndarray:
    x_orig = np.linspace(0, 1, len(vals))
    x_new  = np.linspace(0, 1, n)
    return np.interp(x_new, x_orig, vals)


def save_cropped(results: list[dict], path: str) -> None:
    if not results:
        return
    max_len  = max(r["n_slices_cropped"] for r in results)
    all_cols = (["No", "PatientID", "n_slices_cropped", "z_range"]
                + [f"aec_{i + 1}" for i in range(max_len)])
    pd.DataFrame(results, columns=all_cols).to_excel(path, sheet_name="aec_cropped", index=False)


def save_interp(results: list[dict], path: str) -> None:
    meta_cols = ["No", "PatientID", "n_slices_cropped", "z_range"]

    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for n_pts in INTERP_SIZES:
            interp_rows = []
            for r in results:
                aec_keys = sorted(
                    [k for k in r if k.startswith("aec_")],
                    key=lambda x: int(x.split("_")[1])
                )
                vals = np.array([r[k] for k in aec_keys], dtype=float)
                interped = interp_row(vals, n_pts) if len(vals) >= 2 else np.full(n_pts, np.nan)
                interp_rows.append(
                    {k: r[k] for k in meta_cols}
                    | {f"aec_{i + 1}": round(float(v), 2) for i, v in enumerate(interped)}
                )

            out_cols = meta_cols + [f"aec_{i + 1}" for i in range(n_pts)]
            result_df = pd.DataFrame(interp_rows, columns=out_cols)
            sheet_name = f"aec_{n_pts}"
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"\n[{sheet_name}] 저장 완료 → {path}")
            print(f"  shape : {result_df.shape}")
            print(f"  NaN 행: {result_df['aec_1'].isna().sum()}명")
            print(f"  aec_1      샘플: {result_df['aec_1'].head(3).values}")
            print(f"  aec_{n_pts} 샘플: {result_df[f'aec_{n_pts}'].head(3).values}")


def crop_and_interp():
    z_bounds = pd.read_excel(Z_BOUNDS_PATH)
    z_bounds["PatientID"] = z_bounds["PatientID"].astype(int)

    z_bounds_ok = z_bounds[z_bounds["seg_status"] == "ok"].reset_index(drop=True)
    z_bounds_ok.to_excel(Z_BOUNDS_OK_PATH, index=False)
    ok_pids = set(z_bounds_ok["PatientID"])

    print(f"z_bounds : {len(z_bounds)}명 (전체) / {len(z_bounds_ok)}명 (seg_status=ok)")
    print(f"z_bounds_ok 저장 → {Z_BOUNDS_OK_PATH}")
    print(f"DICOM 기반 : {DICOM_BASE}")

    folder_map = build_folder_map(DICOM_BASE)
    print(f"DCM 폴더 인식 : {len(folder_map)}개\n")

    results: list[dict] = []
    n_skip = 0

    for i, (_, zrow) in enumerate(
        tqdm(z_bounds.iterrows(), total=len(z_bounds), desc="crop"), start=1
    ):
        pid = int(zrow["PatientID"])
        if pd.isna(zrow["t12_slice"]) or pd.isna(zrow["pubis_slice"]):
            tqdm.write(f"  [SKIP] {pid}: t12_slice 또는 pubis_slice 누락")
            n_skip += 1
            continue
        t12_inst = int(zrow["t12_slice"])
        pub_inst = int(zrow["pubis_slice"])

        if pid not in folder_map:
            tqdm.write(f"  [SKIP] {pid}: DCM 폴더 없음")
            n_skip += 1
            continue

        df = read_patient_aec(folder_map[pid])
        if df is None or df.empty:
            tqdm.write(f"  [SKIP] {pid}: DCM 읽기 실패")
            n_skip += 1
            continue

        # 강남 190명은 Instance Number가 1부터 시작하지 않고 ~65 오프셋 이후(예: 66~232)부터 시작함.
        # n_slices(=167)와 inst_max(=232)가 불일치하므로 반드시 DICOM에서 직접 읽어야 함.
        # z_bounds.xlsx의 inst_max 컬럼으로 해당 환자 확인 가능.
        inst_max = int(df["inst"].max())

        # DICOM Instance → z-오름차순 1-based 인덱스 (cranial-to-caudal 취득)
        pub_idx = inst_max - pub_inst + 1   # 두덩뼈 (caudal → 낮은 idx)
        t12_idx = inst_max - t12_inst + 1   # T12    (cranial → 높은 idx)

        lo = max(1, min(pub_idx, t12_idx))
        hi = min(len(df), max(pub_idx, t12_idx))

        region = df.iloc[lo - 1 : hi]       # 0-based slice (lo~hi 포함)
        vals   = region["aec"].tolist()

        if not vals:
            tqdm.write(f"  [SKIP] {pid}: 유효 crop 범위 없음 (lo={lo}, hi={hi}, n={len(df)})")
            n_skip += 1
            continue

        if len(set(vals)) == 1:
            tqdm.write(f"  [SKIP] {pid}: AEC 고정값 ({vals[0]})")
            n_skip += 1
            continue

        z_range = round(float(region["z"].max()) - float(region["z"].min()), 1)

        results.append(
            {
                "No":               zrow["No"],
                "PatientID":        pid,
                "n_slices_cropped": len(vals),
                "z_range":          z_range,
            }
            | {f"aec_{j + 1}": v for j, v in enumerate(vals)}
        )

        if i % BATCH_SIZE == 0:
            save_cropped(results, OUT_PATH)
            tqdm.write(f"  [저장] {i}/{len(z_bounds)}명 처리 완료 (성공 {len(results)}, 스킵 {n_skip})")

    if not results:
        print("[오류] crop된 데이터가 없습니다.")
        return

    save_cropped(results, OUT_PATH)

    print(f"\n[crop 완료] {len(results)}명 저장 → {OUT_PATH}")
    if n_skip:
        print(f"  스킵: {n_skip}명")
    nc = pd.Series([r["n_slices_cropped"] for r in results])
    print(f"  crop 슬라이스 수: median={nc.median():.0f}  "
          f"mean={nc.mean():.1f}  범위=[{nc.min()}, {nc.max()}]")

    save_interp(results, OUT_PATH)

    ok_results = [r for r in results if r["PatientID"] in ok_pids]
    if ok_results:
        save_cropped(ok_results, OUT_OK_PATH)
        save_interp(ok_results, OUT_OK_PATH)
        print(f"\n[ok 전용] {len(ok_results)}명 저장 → {OUT_OK_PATH}")


if __name__ == "__main__":
    crop_and_interp()
