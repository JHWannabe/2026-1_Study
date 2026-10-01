"""
[목적]
TotalSegmentator를 돌릴 CT 볼륨을 환자당 1개만 고르기 위해, DICOM 원본
(D:\\데이터서비스팀 요청(이홍선)\\{No}_{PatientID}_{date}_CT\\*.dcm, 시리즈 구분 없이 한
폴더에 여러 series가 섞여 있음)을 환자별로 직접 스캔해 series_desc별 우선순위를 매기고
환자당 베스트 시리즈 1개를 선정한다.

(예전에는 aec_cropped_novatsat.xlsx의 aec_cropped 시트를 후보 목록으로 썼지만, 그 파일을
더 이상 쓰지 않기로 해 DICOM에서 직접 series를 열거한다. 대상 환자 목록은
예후데이터.xlsx의 대상자명단 시트(연구등록번호)를 쓴다.)

[series_desc → phase 분류 우선순위]
  1. Portal   : portal/venous phase — 체성분 분석 표준 레퍼런스
  2. Contrast : 표시된 phase 없이 "With/Post Contrast"만 있는 단일 조영 시리즈
  3. Delay    : delay/delayed phase
  4. Arterial : arterial phase
  5. Pre/NonContrast : pre-contrast, non-contrast, w/o contrast — 조영 안 된 baseline
  6. Unknown  : 위 키워드로 분류 안 되는 나머지

[liver~pubis 커버리지 필터]
  같은 환자라도 series(phase)별로 실제 촬영 범위(FOV)가 크게 다를 수 있다 — 예:
  PatientID 24565는 Portal이 z_range 395mm(80슬라이스×5mm)인 반면 Arterial/Delay는
  189/186mm(약 64슬라이스×3mm)에 그쳐, 간 상단부터 두덩뼈 하단까지를 담기엔 부족하다.
  series의 실제 간/두덩뼈 포함 여부는 세그멘테이션(3단계 TotalSegmentator)을 돌려봐야
  확실히 알 수 있지만, 그건 이 선정 단계에서 하기엔 비용이 너무 크므로(환자당 후보
  3~5개 × 13000명분 세그멘테이션), 대신 DICOM의 ImagePositionPatient z좌표로 계산한
  물리적 촬영 범위(z_range_mm)를 저비용 대리 지표로 써서 MIN_RANGE_MM 미만인 후보는
  같은 phase의 다른 후보가 있으면 제외한다.
  # ponytail: MIN_RANGE_MM=300은 위 예시(189mm 불충분 vs 395mm 충분)를 참고한 휴리스틱
  # 임계값이며 실제 세그멘테이션 검증을 대체하지 않는다. best_series.xlsx의 range_ok=False
  # 환자가 많으면(3단계 seg_status=pubis_missing/liver_missing 비율로 확인) 값을 조정할 것.
  같은 phase 내에서는 range_ok(있으면 우선) → z_range_mm 큰 쪽 → slice thickness
  얇은 쪽 순으로 고른다.

[재시작 가능] BATCH_SIZE명마다 체크포인트(.pkl)와 중간 best_series.xlsx를 저장한다.
  중단 후 재실행하면 이미 스캔한 PatientID는 건너뛰고 이어서 진행한다.

[출력] best_series.xlsx: PatientID, series_desc, manufacturer, kVp,
  n_slices, slice_thickness_mm, z_range_mm, range_ok
  (phase/phase_rank는 선정 과정에서만 쓰는 내부 정렬 키라 저장 시 제외한다)

[kVp] 어차피 series당 DICOM 헤더를 전부 열어보므로 KVP(0018,0060) 태그도 같이 저장해둔다
  (gangnam_final_dataset.xlsx의 metadata에는 있지만 예전 aec_cropped_novatsat.xlsx에는
  없던 컬럼).
"""

import os
import pickle
import re
from pathlib import Path

import pandas as pd
import pydicom
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "새 데이터 10000명"

PROGNOSIS_PATH = DATA_DIR / "예후데이터.xlsx"
OUT_PATH = DATA_DIR / "best_series.xlsx"
CHECKPOINT_PATH = DATA_DIR / "best_series_checkpoint.pkl"
DICOM_BASE = r"D:\데이터서비스팀 요청(이홍선)"

PHASE_RANK = {"Portal": 1, "Contrast": 2, "Delay": 3, "Arterial": 4, "Pre/NonContrast": 5,
              "Unknown": 6, "Scout/Tracking": 7}   # Scout=볼루스 추적용, 진단용 아님
MIN_RANGE_MM = 300  # 위 [liver~pubis 커버리지 필터] 참고
BATCH_SIZE = 50


def _norm_series_name(s) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


def classify_phase(series_desc: str) -> str:
    s = series_desc.lower()
    # 볼루스 추적/스카우트 — 진단용이 아니라 몇 장짜리 모니터링 시리즈다. 최하위로 뺀다.
    if re.search(r"smart\s*prep|monitor|tracker|tracking|bolus|locator|scout|topogram", s):
        return "Scout/Tracking"
    # 신촌 명명 관례: HVP=hepatic venous phase(문맥기), LAP=late arterial, EP=equilibrium(지연).
    # 이걸 넣지 않으면 신촌 시리즈의 52%가 Unknown(rank 6)이 되어 비조영(rank 5)보다 밀린다.
    if re.search(r"\bhvp\b", s):
        return "Portal"
    if re.search(r"\blap\b", s):
        return "Arterial"
    if re.search(r"\bep\b", s):
        return "Delay"
    if "arterial" in s or "artery" in s:
        return "Arterial"
    if "delay" in s:
        return "Delay"
    if "portal" in s or "venous" in s:
        return "Portal"
    if re.search(r"\bpre(?:\b|(?=\d))|\bnon\b|\bw/o\b|\bwithout\b", s):
        return "Pre/NonContrast"
    if "contrast" in s or "post" in s or "idose" in s or s.strip() == "":
        return "Contrast"
    return "Unknown"


def build_patient_dir_map(dicom_base: str) -> dict[int, str]:
    """DICOM_BASE를 한 번만 스캔해 PatientID → 환자 폴더 경로 매핑을 만든다
    (환자마다 매번 glob하는 대신 폴더 목록 한 번만 읽음)."""
    m: dict[int, str] = {}
    for name in os.listdir(dicom_base):
        if not name.endswith("_CT"):
            continue
        parts = name.split("_")
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        m[pid] = os.path.join(dicom_base, name)
    return m


def scan_patient_series(patient_dir: str) -> dict[str, dict]:
    """환자 폴더의 모든 DICOM을 한 번 스캔해 series_desc별로 슬라이스 수, 두께,
    제조사, z좌표 목록을 모은다."""
    groups: dict[str, dict] = {}
    for fname in os.listdir(patient_dir):
        fp = os.path.join(patient_dir, fname)
        if not os.path.isfile(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
        except Exception:
            continue
        sd = _norm_series_name(getattr(ds, "SeriesDescription", ""))
        g = groups.setdefault(sd, {"n_slices": 0, "slice_thickness_mm": None,
                                    "manufacturer": None, "kVp": None, "z_values": []})
        g["n_slices"] += 1
        if g["slice_thickness_mm"] is None:
            st = getattr(ds, "SliceThickness", None)
            if st is not None:
                try:
                    g["slice_thickness_mm"] = float(st)
                except (TypeError, ValueError):
                    pass
        if g["manufacturer"] is None:
            mfr = getattr(ds, "ManufacturerModelName", None) or getattr(ds, "Manufacturer", None)
            if mfr:
                g["manufacturer"] = str(mfr)
        if g["kVp"] is None:
            kvp = getattr(ds, "KVP", None)
            if kvp is not None:
                try:
                    g["kVp"] = float(kvp)
                except (TypeError, ValueError):
                    pass
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = float(ipp[2]) if ipp is not None else getattr(ds, "SliceLocation", None)
        if z is not None:
            try:
                g["z_values"].append(float(z))
            except (TypeError, ValueError):
                pass
    return groups


def load_targets() -> list[int]:
    df = pd.read_excel(PROGNOSIS_PATH, sheet_name="대상자명단", usecols=["연구등록번호"])
    return sorted(df["연구등록번호"].dropna().astype(int).unique().tolist())


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "rb") as f:
            return pickle.load(f)
    return {"done": set(), "rows": []}


def save_checkpoint(state: dict) -> None:
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(state, f)


def pick_best(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df["range_ok"] = df["z_range_mm"] >= MIN_RANGE_MM
    df_sorted = df.sort_values(
        by=["PatientID", "range_ok", "phase_rank", "z_range_mm", "slice_thickness_mm"],
        ascending=[True, False, True, False, True],
    )
    return df_sorted.groupby("PatientID", as_index=False).first()


def main():
    targets = load_targets()
    state = load_checkpoint()
    done, rows = state["done"], state["rows"]
    pending = [pid for pid in targets if pid not in done]
    print(f"대상 {len(targets)}명 | 완료 {len(done)}명 | 처리 예정 {len(pending)}명")

    dir_map = build_patient_dir_map(DICOM_BASE)
    print(f"DICOM 폴더 매핑: {len(dir_map)}개")

    n_processed = 0
    best_df = pd.DataFrame()
    for pid in tqdm(pending, desc="best series 선정 (DICOM 스캔)"):
        patient_dir = dir_map.get(pid)
        if patient_dir is not None:
            for sd, g in scan_patient_series(patient_dir).items():
                z_vals = g["z_values"]
                z_range_mm = round(max(z_vals) - min(z_vals), 2) if len(z_vals) >= 2 else 0.0
                phase = classify_phase(sd)
                rows.append({
                    "PatientID": pid, "series_desc": sd,
                    "manufacturer": g["manufacturer"], "kVp": g["kVp"],
                    "phase": phase, "phase_rank": PHASE_RANK[phase],
                    "n_slices": g["n_slices"],
                    "slice_thickness_mm": g["slice_thickness_mm"] if g["slice_thickness_mm"] is not None else 99.0,
                    "z_range_mm": z_range_mm,
                })
        done.add(pid)
        n_processed += 1
        if n_processed % BATCH_SIZE == 0 or n_processed == len(pending):
            save_checkpoint({"done": done, "rows": rows})
            if rows:
                best_df = pick_best(pd.DataFrame(rows))
                best_df.drop(columns=["phase", "phase_rank"]).to_excel(OUT_PATH, index=False)
            tqdm.write(f"  체크포인트 저장 | {n_processed}/{len(pending)}명 처리 | "
                       f"best_series.xlsx 중간 저장 {len(best_df)}명")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\n환자 수: {len(best_df)}")
    print("선정된 베스트 시리즈의 phase 분포:")
    print(best_df["phase"].value_counts())
    n_range_ng = (~best_df["range_ok"]).sum()
    print(f"z_range_mm < {MIN_RANGE_MM}mm(range_ok=False, 같은 phase 내 대안 없어 부득이 선정): {n_range_ng}명")
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
