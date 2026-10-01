"""
DICOM 환자 전체(강남 2054 + 신촌 3228)에 대해 PatientSex/PatientAge/Height/Weight를 모아
{site}/metadata/{slug}_patient_metadata.xlsx 로 저장한다.

[출처]
  - 촬영일    : DICOM StudyDate (없으면 SeriesDate/AcquisitionDate) — 환자당 헤더 1개만 읽는다.
  - 성별      : DICOM PatientSex (없으면 코호트 파일)
  - 나이      : DICOM PatientAge(촬영 시점) → DLO_Results → 코호트 순. 0은 미입력으로 보고
                다음 출처로 넘어간다. 세 곳 모두 없으면 결측으로 남긴다.
  - 코호트    : raw/radiation_data_260421_VBcohort.xlsx
                이 파일의 '연구등록번호'가 곧 DICOM 폴더명(PatientID)이다 — 5282명 중 5276명 일치.
                여기에 없는 환자만 DICOM 헤더(PatientSex/PatientAge)로 보충한다.
  - 키/몸무게 : 1순위 {site}/metadata/{site}_DLO_Results.xlsx (연구용으로 정리된 임상 데이터,
                환자당 1행, 날짜 없음)
                2순위 예후데이터/이홍선(2025401459)/1-2. 키몸무게BMI_예진정보★★.xlsx
                (환자당 여러 측정 기록이 있어 촬영일과 가장 가까운 기록을 고른다)
                DICOM의 PatientSize/PatientWeight 태그는 사실상 비어 있어(표본 48명 전원 결측) 쓰지 않는다.

[날짜 기준 선택] 예후데이터는 한 환자에 측정 기록이 여러 건일 수 있다. 촬영일과 측정일
  (입력일자, 없으면 검사시행일자)의 차이가 가장 작은 기록을 쓴다.

[출력 컬럼] PatientID, PatientSex, PatientAge, Height(cm), Weight(kg), BMI, hw_implausible.

[결측 처리] 요청에 따라 값을 채우거나 행을 버리지 않는다. 두 출처 모두 없는 환자는
  Height/Weight가 NaN으로 남는다.

[hw_implausible] 원자료에 Weight 0.6kg, Height 0.0cm 같은 입력 오류가 있다. 값 자체는
  손대지 않고 범위를 벗어난 행만 True로 표시해 하류에서 거를 수 있게 한다.
"""

import os
import re

import pandas as pd
import pydicom
from tqdm import tqdm

DATA_DIR = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data"
COHORT = os.path.join(DATA_DIR, "raw", "radiation_data_260421_VBcohort.xlsx")
PROGNOSIS_HW = os.path.join(DATA_DIR, "예후데이터", "이홍선(2025401459)",
                            "1-2. 키몸무게BMI_예진정보★★.xlsx")
SITES = {"강남": "gangnam", "신촌": "sinchon"}

# 코호트에 소아(2~14세)가 섞여 있어 단일 기준으로는 성인 입력 오류와 정상 소아를 함께
# 처리할 수 없다. 나이로 기준을 나눈다. 나이를 모르면 성인 기준을 쓴다.
# 성인 기준 하한을 130cm로 둔 이유: Height=100(자리표시자)이 실제로 섞여 있다.
ADULT = {"height": (130.0, 220.0), "weight": (30.0, 200.0), "bmi": (10.0, 60.0)}
CHILD = {"height": (40.0, 200.0), "weight": (2.0, 150.0), "bmi": (8.0, 45.0)}
AGE_RANGE = (0.0, 120.0)


# 나이에 맞는 판정 범위를 고른다(18세 미만은 소아 기준).
def _ranges(age) -> dict:
    return CHILD if (pd.notna(age) and float(age) < 18) else ADULT


# DICOM 폴더에 있는 PatientID 목록을 읽는다.
def dicom_pids(site: str) -> list[int]:
    base = rf"E:\영상제공\{site}\{site}_axial"
    return sorted(int(f) for f in os.listdir(base)
                  if f.isdigit() and os.path.isdir(os.path.join(base, f)))


# 환자 폴더에서 헤더 1개만 읽어 촬영일·성별·나이를 가져온다("050Y" → 50).
def read_header(site: str, pid: int) -> dict:
    pdir = rf"E:\영상제공\{site}\{site}_axial\{pid}"
    try:
        for sub in sorted(os.listdir(pdir)):
            sd = os.path.join(pdir, sub)
            if not os.path.isdir(sd):
                continue
            for f in sorted(os.listdir(sd)):
                fp = os.path.join(sd, f)
                if not os.path.isfile(fp):
                    continue
                ds = pydicom.dcmread(fp, stop_before_pixels=True)
                raw_date = ""
                for tag in ("StudyDate", "SeriesDate", "AcquisitionDate"):
                    raw_date = str(getattr(ds, tag, "") or "")
                    if len(raw_date) == 8:
                        break
                age_raw = str(getattr(ds, "PatientAge", "") or "")
                m = re.match(r"^(\d+)\s*Y?$", age_raw.strip())
                return {
                    "study_date": pd.to_datetime(raw_date, format="%Y%m%d", errors="coerce"),
                    "sex": (str(getattr(ds, "PatientSex", "") or "") or None),
                    "age": (int(m.group(1)) if m else None),
                }
    except Exception:
        pass
    return {"study_date": pd.NaT, "sex": None, "age": None}


# 코호트 파일에서 PatientID → (성별, 나이) 매핑을 만든다.
def load_cohort() -> dict[int, tuple]:
    d = pd.read_excel(COHORT)
    pid = pd.to_numeric(d["연구등록번호"], errors="coerce")
    return {int(p): (sex, age) for p, sex, age in zip(pid, d["성별"], d["내원시나이"])
            if pd.notna(p)}


# DLO_Results_SMI_kVp100에서 PatientID → (키, 몸무게) 매핑을 만든다.
# 같은 임상 데이터의 다른 정리본으로, 환자 집합은 DLO_Results의 부분집합이다(추가 환자 0명).
# 두 파일의 값은 전반적으로 꽤 다르므로(몸무게 5kg 초과 차이가 강남 15%·신촌 24%) 전면
# 교체하지 않고, DLO_Results 값이 생리학적 범위를 벗어난 경우에만 대체한다.
def load_smi_hw(site: str) -> dict[int, tuple]:
    f = os.path.join(DATA_DIR, site, "metadata", f"{site}_DLO_Results_SMI_kVp100.xlsx")
    if not os.path.exists(f):
        return {}
    d = pd.read_excel(f)
    out = {}
    for _, r in d.iterrows():
        if pd.notna(r.get("PatientID")):
            out[int(r["PatientID"])] = (r.get("Height"), r.get("Weight"))
    return out


# 키·몸무게가 둘 다 있고 생리학적 범위 안이면 True.
def hw_ok(h, w, age=None) -> bool:
    if not (pd.notna(h) and pd.notna(w)):
        return False
    h, w = float(h), float(w)
    r = _ranges(age)
    if not (r["height"][0] <= h <= r["height"][1] and r["weight"][0] <= w <= r["weight"][1]):
        return False
    bmi = w / (h / 100.0) ** 2
    return r["bmi"][0] <= bmi <= r["bmi"][1]


# DLO_Results에서 PatientID → PatientAge 매핑(DICOM 헤더에 나이가 없을 때의 폴백).
def load_dlo_age(site: str) -> dict[int, float]:
    f = os.path.join(DATA_DIR, site, "metadata", f"{site}_DLO_Results.xlsx")
    if not os.path.exists(f):
        return {}
    d = pd.read_excel(f)
    return {int(r["PatientID"]): r.get("PatientAge") for _, r in d.iterrows()
            if pd.notna(r.get("PatientID")) and pd.notna(r.get("PatientAge"))}


# DLO_Results에서 PatientID → (키, 몸무게) 매핑을 만든다(환자당 1행, 날짜 없음).
def load_dlo_hw(site: str) -> dict[int, tuple]:
    f = os.path.join(DATA_DIR, site, "metadata", f"{site}_DLO_Results.xlsx")
    if not os.path.exists(f):
        return {}
    d = pd.read_excel(f)
    out = {}
    for _, r in d.iterrows():
        if pd.notna(r.get("PatientID")):
            out[int(r["PatientID"])] = (r.get("Height"), r.get("Weight"))
    return out


# 예후데이터에서 PatientID → [(측정일, 키, 몸무게), ...] 목록을 만든다.
def load_prognosis_hw() -> dict[int, list[tuple]]:
    if not os.path.exists(PROGNOSIS_HW):
        return {}
    d = pd.read_excel(PROGNOSIS_HW)
    pid = pd.to_numeric(d["연구등록번호"], errors="coerce")
    when = pd.to_datetime(d["입력일자"], errors="coerce").fillna(
        pd.to_datetime(d["검사시행일자"], errors="coerce"))
    out: dict[int, list[tuple]] = {}
    for p, t, h, w in zip(pid, when, d["신장"], d["체중"]):
        if pd.notna(p) and (pd.notna(h) or pd.notna(w)):
            out.setdefault(int(p), []).append((t, h, w))
    return out


# 촬영일과 가장 가까운 측정 기록을 고른다. 촬영일이 없으면 가장 최근 기록을 쓴다.
def pick_nearest(records: list[tuple], study_date) -> tuple:
    dated = [r for r in records if pd.notna(r[0])]
    if pd.notna(study_date) and dated:
        best = min(dated, key=lambda r: abs((r[0] - study_date).days))
        return best[1], best[2], abs((best[0] - study_date).days)
    if dated:
        best = max(dated, key=lambda r: r[0])
        return best[1], best[2], None
    return records[0][1], records[0][2], None


# 값이 생리학적 범위를 벗어나면 True (값은 그대로 두고 표시만 한다).
def implausible(h, w, age, bmi=None) -> bool:
    r = _ranges(age)
    for v, (lo, hi) in ((h, r["height"]), (w, r["weight"]),
                        (age, AGE_RANGE), (bmi, r["bmi"])):
        if pd.notna(v) and not (lo <= float(v) <= hi):
            return True
    return False


def build(site: str, cohort: dict, prog: dict) -> pd.DataFrame:
    dlo = load_dlo_hw(site)
    smi = load_smi_hw(site)
    dlo_age = load_dlo_age(site)
    rows = []
    for pid in tqdm(dicom_pids(site), desc=f"{site} 메타데이터"):
        # DICOM 헤더는 코호트에 없는 환자(성별/나이 보충)와 예후데이터를 쓰는 환자
        # (촬영일 기준 선택)만 읽는다 — 전수로 읽으면 USB 외장하드에서 6분쯤 걸린다.
        # 나이는 DICOM PatientAge(촬영 시점)를 우선한다. 코호트의 '내원시나이'는 CT가 아닌
        # 다른 내원 시점 기준이라 3년 이상 어긋나는 환자가 강남 1114명·신촌 1591명이다
        # (최대 11년). DICOM과 DLO_Results의 PatientAge는 표본에서 전원 일치했다.
        hdr = read_header(site, pid)

        # 나이 0은 값이 아니라 미입력이다(DICOM 태그 "000Y" 또는 공란). 키 178cm인 성인이
        # 0세로 찍히는 식이라 그대로 두면 신생아로 오독된다 — 결측으로 보고 다음 출처를 본다.
        def _age_ok(v):
            return v is not None and pd.notna(v) and float(v) > 0

        sex = hdr["sex"]
        age = next((v for v in (hdr["age"], dlo_age.get(pid),
                                cohort[pid][1] if pid in cohort else None)
                    if _age_ok(v)), None)
        if not sex and pid in cohort:
            sex = cohort[pid][0]

        h = w = None
        if pid in dlo and (pd.notna(dlo[pid][0]) or pd.notna(dlo[pid][1])):
            h, w = dlo[pid]
            # DLO 값이 입력 오류(키 0cm, 몸무게 0.6kg 등)면 SMI_kVp100의 정상값으로 바꾼다.
            if not hw_ok(h, w, age) and pid in smi and hw_ok(*smi[pid], age):
                h, w = smi[pid]
        elif pid in prog:
            h, w, _ = pick_nearest(prog[pid], hdr["study_date"])

        bmi = None
        if pd.notna(h) and pd.notna(w) and float(h) > 0:
            bmi = round(float(w) / (float(h) / 100.0) ** 2, 2)

        rows.append({
            "PatientID": pid,
            "PatientSex": sex, "PatientAge": age,
            "Height": h, "Weight": w, "BMI": bmi,
            "hw_implausible": int(implausible(h, w, age, bmi)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    cohort = load_cohort()
    prog = load_prognosis_hw()
    print(f"코호트 {len(cohort)}명 · 예후데이터 키몸무게 {len(prog)}명 로드\n")
    for site, slug in SITES.items():
        df = build(site, cohort, prog)
        out = os.path.join(DATA_DIR, site, "metadata", f"{slug}_patient_metadata.xlsx")
        df.to_excel(out, index=False)
        n = len(df)
        print(f"\n[{site}] {n}명 → {os.path.basename(out)}")
        for c in ("PatientSex", "PatientAge", "Height", "Weight"):
            miss = df[c].isna().sum()
            print(f"   {c:12s} 결측 {miss:5d} ({miss / n * 100:.0f}%)")
        print(f"   이상치 플래그: {int(df.hw_implausible.sum())}명")


if __name__ == "__main__":
    main()
