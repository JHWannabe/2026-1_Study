"""
[전체 목적]
각 환자의 DICOM 파일에서 AEC(자동노출제어) 값을 읽고,
z_bounds 엑셀에 기록된 간 상단~두덩뼈 구간으로 잘라낸(crop) 뒤
128·256포인트로 선형 보간해 저장한다.

[AEC란?]
AEC(Automatic Exposure Control, 자동노출제어)는 CT 스캔 중 각 슬라이스마다
방사선 선량을 자동 조절하는 기능이다.
DICOM 태그 XRayTubeCurrent(0018,1151)에 슬라이스별 관전류(mA) 값으로 저장된다.
장기나 지방 등 밀도가 높은 구간에서는 더 높은 mA를 사용하므로,
AEC 신호 파형이 해당 구간의 조직 밀도 분포를 간접적으로 반영한다.

[처리 흐름]
  STEP 1. z_bounds 엑셀에서 각 환자의 경계 슬라이스 Instance Number 로드
  STEP 2. DICOM 폴더에서 환자별 AEC 값·z좌표·Instance Number 읽기
  STEP 3. Instance Number → z정렬 인덱스 변환 (강남 오프셋 처리 포함)
  STEP 4. 간 상단~두덩뼈 구간으로 AEC 신호 crop
  STEP 5. 128·256포인트 선형 보간
  STEP 6. 엑셀 저장 (기본 시트 + aec_128 + aec_256)

[Instance Number → z정렬 인덱스 변환]
  CT는 cranial(머리)→caudal(발) 방향으로 스캔하며 Instance Number를 부여한다.
  DICOM을 z좌표 오름차순으로 정렬하면 caudal(발)이 앞(낮은 인덱스)에 온다.
  따라서: z_sorted_idx = inst_max - instance_number + 1
  예: inst_max=200, liver_upper_inst=150 → liver_upper_idx = 200-150+1 = 51번째

  ※ 강남 190명 특이사항:
    Instance Number가 1이 아닌 66 이상에서 시작하는 환자가 있다.
    n_slices(=167)와 inst_max(=232)가 다르므로 DICOM에서 직접 inst_max를 읽어야 한다.

[보간(interpolation) 이유]
  환자마다 crop된 AEC 구간의 슬라이스 수가 다르다.
  모델 입력으로 사용하려면 고정 길이 벡터가 필요하므로
  선형 보간으로 128·256포인트 표준 길이로 변환한다.

[출력 파일: 강남_aec_cropped.xlsx]
  aec_cropped 시트 : PatientID, n_slices_cropped, z_range, aec_1 ~ aec_N (원본 길이)
  aec_128 시트     : PatientID, n_slices_cropped, z_range, aec_1 ~ aec_128
  aec_256 시트     : PatientID, n_slices_cropped, z_range, aec_1 ~ aec_256
"""

import os
import json
import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm

# ── 설정 ─────────────────────────────────────────────────────────────────────

SITE       = "강남"
DICOM_BASE = rf"D:\영상제공\{SITE}\{SITE}_axial"   # 환자별 DICOM 폴더 최상위 경로
DATA_BASE  = rf"C:\Users\jhjun\OneDrive\Desktop\대학원\data\{SITE}"

Z_BOUNDS_PATH = rf"{DATA_BASE}\aec\liver_pubis\{SITE}_z_bounds.xlsx"
OUT_OK_PATH   = rf"{DATA_BASE}\aec\liver_pubis\{SITE}_aec_cropped_ok.xlsx"

INTERP_SIZES = [128, 256]   # 보간 목표 포인트 수
BATCH_SIZE   = 20           # 몇 명마다 중간 저장할지

# done_pids(스킵할 PID 목록)
CHECKPOINT = rf"{DATA_BASE}\aec\liver_pubis\.crop_aec_checkpoint.json"
# 재시작 위치(processed 인덱스, n_skip 카운터)
PROGRESS   = rf"{DATA_BASE}\aec\liver_pubis\.crop_aec_progress.json"


# ── 체크포인트 / 진행 상태 ────────────────────────────────────────────────────

def save_checkpoint(done_pids: set[int]) -> None:
    """완료(성공+스킵) PID 목록만 JSON으로 저장한다."""
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"done_pids": sorted(done_pids)}, f, ensure_ascii=False)


def load_checkpoint() -> set[int]:
    """저장된 done_pids를 반환한다. 파일 없으면 빈 set."""
    if not os.path.exists(CHECKPOINT):
        return set()
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        return set(json.load(f).get("done_pids", []))


def save_progress(processed: int, n_skip: int) -> None:
    """중단 위치(processed 인덱스)와 스킵 카운터를 저장한다."""
    with open(PROGRESS, "w", encoding="utf-8") as f:
        json.dump({"processed": processed, "n_skip": n_skip}, f, ensure_ascii=False)


def load_progress() -> tuple[int, int]:
    """저장된 (processed, n_skip)을 반환한다. 파일 없으면 (0, 0)."""
    if not os.path.exists(PROGRESS):
        return 0, 0
    with open(PROGRESS, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("processed", 0), data.get("n_skip", 0)


def load_results_from_excel(path: str) -> list[dict]:
    """저장된 aec_cropped 시트에서 results 리스트를 복원한다."""
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_excel(path, sheet_name="aec_cropped")
        return df.to_dict("records")
    except Exception:
        return []


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────

def build_folder_map(dicom_base: str) -> dict[int, str]:
    """
    DICOM 최상위 폴더를 스캔해 PatientID(int) → 실제 시리즈 폴더 경로 딕셔너리를 만든다.

    지원하는 폴더명 형식:
      ① 신규: {PatientID}              (예: 5253702)
      ② 구형: {순번}_{PatientID}_...   (예: 0001_5253702_20180201_CT)
    시리즈 폴더가 여러 개면 DCM 파일이 가장 많은 폴더를 선택한다.
    """
    folder_map: dict[int, str] = {}
    for name in os.listdir(dicom_base):
        patient_dir = os.path.join(dicom_base, name)
        if not os.path.isdir(patient_dir):
            continue
        parts = name.split("_")
        try:
            pid = int(parts[0]) if len(parts) == 1 else int(parts[1])
        except ValueError:
            continue
        # 하위 시리즈 폴더 중 .dcm 파일이 가장 많은 것 선택
        best, best_count = None, 0
        for sub in os.listdir(patient_dir):
            sub_path = os.path.join(patient_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            n = len([f for f in os.listdir(sub_path) if f.endswith(".dcm")])
            if n > best_count:
                best_count = n
                best       = sub_path
        if best:
            folder_map[pid] = best
    return folder_map


def read_patient_aec(series_dir: str) -> pd.DataFrame | None:
    """
    한 환자의 DICOM 시리즈 폴더 전체를 읽어
    z좌표 오름차순으로 정렬된 DataFrame을 반환한다.

    반환 컬럼:
      inst : DICOM InstanceNumber (슬라이스 고유 번호)
      z    : ImagePositionPatient[2] — 슬라이스의 z 좌표(mm)
      aec  : XRayTubeCurrent — 해당 슬라이스의 관전류(mA)

    [z 오름차순 정렬의 의미]
    z가 낮을수록 발 방향(caudal), 높을수록 머리 방향(cranial)이다.
    정렬하면 df.index 0 = 가장 발쪽 슬라이스가 된다.

    [XRayTubeCurrent 대체 태그]
    일부 DICOM은 TubeCurrent 태그를 대신 사용한다. 두 태그 모두 시도한다.

    실패 시 None 반환.
    """
    rows = []
    for fname in os.listdir(series_dir):
        if not fname.endswith(".dcm"):
            continue
        fpath = os.path.join(series_dir, fname)
        try:
            ds   = pydicom.dcmread(fpath, stop_before_pixels=True)
            ipp  = getattr(ds, "ImagePositionPatient", None)
            z    = float(ipp[2]) if ipp is not None else float("nan")
            inst = int(getattr(ds, "InstanceNumber", 0))
            aec  = getattr(ds, "XRayTubeCurrent", None)
            if aec is None:
                aec = getattr(ds, "TubeCurrent", None)   # 대체 태그 시도
            rows.append({"inst": inst, "z": z, "aec": aec})
        except Exception:
            continue   # 읽기 실패한 파일은 건너뜀

    if not rows:
        return None

    # z 좌표 오름차순 정렬: idx 0 = caudal(발쪽), idx -1 = cranial(머리쪽)
    df = pd.DataFrame(rows).sort_values("z").reset_index(drop=True)
    return df


def interp_row(vals: np.ndarray, n: int) -> np.ndarray:
    """
    길이가 다른 AEC 신호를 n포인트로 선형 보간한다.

    원리: 원본 배열의 인덱스를 0~1 범위로 정규화한 x_orig와
         목표 배열의 인덱스 x_new를 생성하고 np.interp로 매핑한다.
    예: [100, 200, 150]을 5포인트로 → [100, 133, 167, 200, 175, 150] 방향으로 보간
    """
    x_orig = np.linspace(0, 1, len(vals))   # 원본 포인트 위치 (균등 간격)
    x_new  = np.linspace(0, 1, n)           # 목표 포인트 위치 (균등 간격)
    return np.interp(x_new, x_orig, vals)


def save_cropped(results: list[dict], path: str) -> None:
    """
    crop된 원본 AEC 시퀀스를 'aec_cropped' 시트로 저장한다.
    환자마다 crop 길이가 다르므로 최대 길이를 기준으로 컬럼을 생성한다.
    (짧은 환자의 빈 컬럼은 NaN으로 채워진다.)
    """
    if not results:
        return
    max_len  = max(r["n_slices_cropped"] for r in results)
    all_cols = (["PatientID", "n_slices_cropped", "z_range"]
                + [f"aec_{i + 1}" for i in range(max_len)])
    pd.DataFrame(results, columns=all_cols).to_excel(
        path, sheet_name="aec_cropped", index=False
    )


def save_interp(results: list[dict], path: str) -> None:
    """
    모든 환자의 AEC 신호를 각 보간 크기(128, 256포인트)로 변환해
    별도 시트(aec_128, aec_256)로 저장한다.

    보간이 불가능한 경우(슬라이스 수 < 2): NaN 배열로 채운다.
    기존 파일에 시트를 추가하기 위해 'append' 모드로 ExcelWriter를 연다.
    """
    meta_cols = ["PatientID", "n_slices_cropped", "z_range"]

    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for n_pts in INTERP_SIZES:
            interp_rows = []
            for r in results:
                # aec_1, aec_2, ... 컬럼을 숫자 순서로 정렬해 numpy 배열로 변환
                aec_keys = sorted(
                    [k for k in r if k.startswith("aec_")],
                    key=lambda x: int(x.split("_")[1])
                )
                vals     = np.array([r[k] for k in aec_keys], dtype=float)
                # 슬라이스 수가 2 이상이어야 보간 가능
                interped = interp_row(vals, n_pts) if len(vals) >= 2 else np.full(n_pts, np.nan)
                interp_rows.append(
                    {k: r[k] for k in meta_cols}
                    | {f"aec_{i + 1}": round(float(v), 2) for i, v in enumerate(interped)}
                )

            out_cols  = meta_cols + [f"aec_{i + 1}" for i in range(n_pts)]
            result_df = pd.DataFrame(interp_rows, columns=out_cols)
            sheet_name = f"aec_{n_pts}"
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"\n[{sheet_name}] 저장 완료 → {path}")
            print(f"  shape : {result_df.shape}")
            print(f"  NaN 행: {result_df['aec_1'].isna().sum()}명")
            print(f"  aec_1      샘플: {result_df['aec_1'].head(3).values}")
            print(f"  aec_{n_pts} 샘플: {result_df[f'aec_{n_pts}'].head(3).values}")


# ── 메인 함수 ─────────────────────────────────────────────────────────────────

def crop_and_interp():
    """
    [전체 처리 흐름]
    1. z_bounds 엑셀 로드 → seg_status=ok 환자만 별도 추출
    2. DICOM 폴더 맵 생성
    3. 각 환자에 대해:
       a. DICOM에서 AEC·z·Instance Number 읽기
       b. inst_max 계산 (강남 오프셋 문제 대응)
       c. Instance Number → z정렬 인덱스 변환
       d. [pubis_idx, liver_upper_idx] 구간으로 crop
       e. AEC 고정값(신호 없음) 필터링
    4. BATCH_SIZE마다 중간 저장
    5. 선형 보간(128·256포인트) 후 최종 저장
    """
    # ── z_bounds 로드 ─────────────────────────────────────────────────────────
    z_bounds = pd.read_excel(Z_BOUNDS_PATH)
    z_bounds["PatientID"] = z_bounds["PatientID"].astype(int)

    z_bounds_ok = z_bounds[z_bounds["seg_status"] == "ok"].reset_index(drop=True)

    print(f"z_bounds : {len(z_bounds)}명 (전체) / {len(z_bounds_ok)}명 (seg_status=ok)")
    print(f"DICOM 기반 : {DICOM_BASE}")

    folder_map = build_folder_map(DICOM_BASE)
    print(f"DCM 폴더 인식 : {len(folder_map)}개\n")

    # 체크포인트 / 진행 상태 로드
    done_pids              = load_checkpoint()
    start_idx, n_skip      = load_progress()
    results: list[dict]    = load_results_from_excel(OUT_OK_PATH) if start_idx > 0 else []
    total                  = len(z_bounds_ok)

    if start_idx > 0:
        print(f"재시작 감지: {total}명 중 {start_idx}번째부터 재개 "
              f"(기존 성공 {len(results)}명, 스킵 {n_skip}명)")
    if done_pids:
        print(f"완료 PID {len(done_pids)}개 로드 → 재실행 시 자동 스킵")

    # ── 환자별 처리 루프 ──────────────────────────────────────────────────────
    for idx in tqdm(range(start_idx, total), desc="crop", initial=start_idx, total=total):
        zrow = z_bounds_ok.iloc[idx]
        i    = idx + 1   # 1-based 인덱스 (출력 및 저장 주기 계산용)
        pid  = int(zrow["PatientID"])

        # 이전 실행에서 처리 완료된 PID는 DICOM 읽기 없이 바로 스킵
        if pid in done_pids:
            continue

        # ── 경계 슬라이스 Instance Number 확인 ───────────────────────────────
        # liver_upper_slice 또는 pubis_slice가 누락이면 crop 불가
        if pd.isna(zrow["liver_upper_slice"]) or pd.isna(zrow["pubis_slice"]):
            tqdm.write(f"  [SKIP] {pid}: liver_upper_slice 또는 pubis_slice 누락")
            n_skip += 1
            done_pids.add(pid)
            continue
        liver_upper_inst = int(zrow["liver_upper_slice"])   # 간 상단 Instance Number
        pub_inst         = int(zrow["pubis_slice"])          # 두덩뼈 Instance Number

        # ── DICOM 폴더 존재 확인 ──────────────────────────────────────────────
        if pid not in folder_map:
            tqdm.write(f"  [SKIP] {pid}: DCM 폴더 없음")
            n_skip += 1
            done_pids.add(pid)
            continue

        # ── DICOM 읽기: AEC·z·Instance Number 수집 ───────────────────────────
        df = read_patient_aec(folder_map[pid])
        if df is None or df.empty:
            tqdm.write(f"  [SKIP] {pid}: DCM 읽기 실패")
            n_skip += 1
            done_pids.add(pid)
            continue

        # ── inst_max 계산 ─────────────────────────────────────────────────────
        # [강남 오프셋 주의]
        # 강남 190명 중 일부는 Instance Number가 1이 아닌 66 이상에서 시작한다.
        # n_slices(=167)와 inst_max(=232)가 불일치하므로
        # 단순히 n_slices로 계산하면 오차가 생긴다.
        # DICOM에서 직접 읽은 최대 Instance Number를 사용해야 정확하다.
        inst_max = int(df["inst"].max())

        # ── Instance Number → z정렬 1-based 인덱스 변환 ─────────────────────
        # CT 스캔 방향: cranial(머리)→caudal(발) = Instance Number 증가 방향
        # z정렬 방향:   caudal(발)→cranial(머리) = z 증가 방향 (인덱스 0=발쪽)
        # 변환식: z_sorted_idx = inst_max - instance_number + 1
        #   - 큰 Instance Number(머리쪽) → 작은 z_sorted_idx
        #   - 작은 Instance Number(발쪽) → 큰 z_sorted_idx
        pub_idx         = inst_max - pub_inst         + 1  # 두덩뼈: caudal → 낮은 인덱스
        liver_upper_idx = inst_max - liver_upper_inst + 1  # 간 상단: cranial → 높은 인덱스

        # crop 범위: [lo, hi] (1-based, 양 끝 포함)
        # lo = 두 인덱스 중 작은 값 (caudal쪽), hi = 큰 값 (cranial쪽)
        lo = max(1, min(pub_idx, liver_upper_idx))
        hi = min(len(df), max(pub_idx, liver_upper_idx))

        # ── AEC 값 추출 ────────────────────────────────────────────────────────
        # iloc은 0-based이므로 lo-1:hi로 슬라이싱 (hi는 포함)
        region = df.iloc[lo - 1 : hi]
        vals   = region["aec"].tolist()

        if not vals:
            tqdm.write(f"  [SKIP] {pid}: 유효 crop 범위 없음 (lo={lo}, hi={hi}, n={len(df)})")
            n_skip += 1
            done_pids.add(pid)
            continue

        # ── AEC 고정값 필터링 ─────────────────────────────────────────────────
        # 전체 구간이 동일한 값이면 AEC가 비활성화된 것으로 판단하고 제외
        # (AEC 비활성 기기는 mA가 스캔 내내 동일한 고정값으로 유지됨)
        if len(set(vals)) == 1:
            tqdm.write(f"  [SKIP] {pid}: AEC 고정값 ({vals[0]})")
            n_skip += 1
            done_pids.add(pid)
            continue

        # z_range: crop 구간의 실제 길이(mm)
        z_range = round(float(region["z"].max()) - float(region["z"].min()), 1)

        # 결과 딕셔너리 구성: 메타데이터 + aec_1, aec_2, ..., aec_N
        results.append(
            {
                "PatientID":        pid,
                "n_slices_cropped": len(vals),    # crop된 슬라이스 수
                "z_range":          z_range,       # crop 구간 길이(mm)
            }
            | {f"aec_{j + 1}": v for j, v in enumerate(vals)}  # aec_1 ~ aec_N
        )
        done_pids.add(pid)

        # BATCH_SIZE마다 중간 저장
        if i % BATCH_SIZE == 0 or i == total:
            save_cropped(results, OUT_OK_PATH)
            save_checkpoint(done_pids)
            save_progress(i, n_skip)
            tqdm.write(f"  [저장] {i}/{total}명 처리 완료 (성공 {len(results)}, 스킵 {n_skip})")

    if not results:
        print("[오류] crop된 데이터가 없습니다.")
        return

    # ── 최종 저장 ─────────────────────────────────────────────────────────────
    save_cropped(results, OUT_OK_PATH)

    # 완료 후 progress 초기화 (done_pids는 유지 → 재실행 시 바로 스킵)
    save_checkpoint(done_pids)
    save_progress(0, 0)

    print(f"\n[crop 완료] {len(results)}명 저장 → {OUT_OK_PATH}")
    if n_skip:
        print(f"  스킵: {n_skip}명")
    nc = pd.Series([r["n_slices_cropped"] for r in results])
    print(f"  crop 슬라이스 수: median={nc.median():.0f}  "
          f"mean={nc.mean():.1f}  범위=[{nc.min()}, {nc.max()}]")

    # ── 선형 보간 시트 추가 (128·256포인트) ───────────────────────────────────
    save_interp(results, OUT_OK_PATH)


if __name__ == "__main__":
    crop_and_interp()
