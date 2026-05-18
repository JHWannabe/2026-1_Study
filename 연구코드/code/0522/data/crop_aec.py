"""
강남_z_bounds_.xlsx의 t12_slice / pubis_slice (DICOM Instance Number)를 이용해
강남_aec_raw.xlsx의 AEC 신호를 T12 상단 ~ 두덩뼈 하단 구간으로 crop한다.

배경:
  - aec_raw의 slice_N 컬럼: z좌표 오름차순 정렬 기반 1-based index
      slice_1 = 다리(caudal, z 최소)  →  slice_N = 머리(cranial, z 최대)
  - z_bounds_의 t12_slice / pubis_slice: DICOM Instance Number
      강남 데이터는 cranial→caudal 취득 (Instance 1 = 머리 방향, Instance 증가 = 다리 방향)
      → aec_raw 순서와 반대이므로 변환 필요
  - 변환식 (cranial-to-caudal): z_sorted_idx = n_slices - instance + 1
      T12 (cranial, z 크다)   → 높은 z_sorted_idx (머리 쪽, slice_N 부근)
      두덩뼈 (caudal, z 작다) → 낮은 z_sorted_idx (다리 쪽, slice_1 부근)

출력: 강남_aec_cropped.xlsx
  컬럼: No, PatientID, n_slices_cropped, aec_1 ~ aec_N
"""

import pandas as pd
import numpy as np

SITE = "강남"
BASE = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}"

Z_BOUNDS_PATH = rf"{BASE}\{SITE}_z_bounds.xlsx"
AEC_RAW_PATH  = rf"{BASE}\{SITE}_aec_raw.xlsx"
OUT_PATH      = rf"{BASE}\{SITE}_aec_cropped.xlsx"

# DICOM 취득 방향
# True  = cranial-to-caudal: Instance 1 = 머리(높은 z), 번호 증가 = 다리 방향
#         → aec_raw(다리→머리)와 반대이므로 z_sorted = n - instance + 1
# False = caudal-to-cranial: Instance 1 = 다리(낮은 z)
#         → aec_raw와 같은 방향이므로 z_sorted = instance
CRANIAL_TO_CAUDAL = True  # 강남 데이터는 머리 방향부터 촬영


def instance_to_z_sorted(instance: int, n_slices: int) -> int:
    """DICOM Instance Number → aec_raw z-오름차순 1-based 인덱스."""
    if CRANIAL_TO_CAUDAL:
        return n_slices - instance + 1
    return instance


def crop():
    z_bounds = pd.read_excel(Z_BOUNDS_PATH)
    aec_raw  = pd.read_excel(AEC_RAW_PATH)

    z_df = z_bounds.copy()
    z_df["PatientID"] = z_df["PatientID"].astype(int)
    aec_raw["PatientID"] = aec_raw["PatientID"].astype(int)

    print(f"z_bounds      : {len(z_df)}명")
    print(f"aec_raw       : {len(aec_raw)}명")

    aec_idx       = aec_raw.set_index("PatientID")
    available_set = set(aec_raw.columns)

    results = []
    n_skip  = 0

    for _, zrow in z_df.iterrows():
        pid      = int(zrow["PatientID"])
        n        = int(zrow["n_slices"])
        t12_inst = int(zrow["t12_slice"])
        pub_inst = int(zrow["pubis_slice"])

        if pid not in aec_idx.index:
            print(f"  [SKIP] {pid}: aec_raw 없음")
            n_skip += 1
            continue

        # Instance Number → z-sorted 1-based 인덱스 변환
        pub_sorted = instance_to_z_sorted(pub_inst, n)  # 두덩뼈 (더 caudal = 낮은 인덱스)
        t12_sorted = instance_to_z_sorted(t12_inst, n)  # T12    (더 cranial = 높은 인덱스)

        lo = max(1, min(pub_sorted, t12_sorted))
        hi = min(n, max(pub_sorted, t12_sorted))

        crop_cols = [f"slice_{i}" for i in range(lo, hi + 1)
                     if f"slice_{i}" in available_set]

        if not crop_cols:
            print(f"  [SKIP] {pid}: 유효한 crop 범위 없음 (lo={lo}, hi={hi})")
            n_skip += 1
            continue

        vals = aec_idx.loc[pid, crop_cols].values.tolist()

        results.append(
            {
                "No":               zrow["No"],
                "PatientID":        pid,
                "n_slices_cropped": len(vals),
            }
            | {f"aec_{i + 1}": v for i, v in enumerate(vals)}
        )

    if not results:
        print("[오류] crop된 데이터가 없습니다.")
        return pd.DataFrame()

    max_len   = max(r["n_slices_cropped"] for r in results)
    all_cols  = (["No", "PatientID", "n_slices_cropped"]
                 + [f"aec_{i + 1}" for i in range(max_len)])
    out_df    = pd.DataFrame(results, columns=all_cols)
    out_df.to_excel(OUT_PATH, index=False)

    print(f"\n[완료] {len(results)}명 저장 → {OUT_PATH}")
    if n_skip:
        print(f"  스킵: {n_skip}명")
    nc = out_df["n_slices_cropped"]
    print(f"  crop 슬라이스 수: median={nc.median():.0f}  "
          f"mean={nc.mean():.1f}  범위=[{nc.min()}, {nc.max()}]")

    return out_df


if __name__ == "__main__":
    import time
    # 7시간 30분 대기 후 실행(1시간마다 countdown)
    for i in range(28800, 0, -1):
        if i % 3600 == 0:  # 1시간마다 출력
            print(f"실행까지 {i // 3600}시간...")
        time.sleep(1)
    print("=== AEC Crop 시작 ===")
    crop()
