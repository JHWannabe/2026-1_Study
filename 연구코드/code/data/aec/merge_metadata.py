"""
{slug}_landmark.xlsx(landmarks/body_composition/aec_total/aec_landmark 4개 시트)와
{slug}_patient_metadata.xlsx(PatientSex/PatientAge/Height/Weight/BMI/hw_implausible)를
합쳐 {slug}_landmark_full.xlsx(시트 5개)와 조건 충족 환자만 남긴 _filtered.xlsx로 저장한다. 강남/신촌 둘 다 처리한다.

[왜 landmark 파일에 직접 시트를 추가하지 않는가]
landmark 파이프라인(check_landmarks.py)이 지금도 {slug}_landmark.xlsx에 계속 쓰고 있다.
그 파일을 직접 열어 patient_metadata 시트를 추가해도, 그 프로세스가 메모리에 올려둔
_write_all은 "landmarks/body_composition/aec_total/aec_landmark" 4개만 알고 있어서
다음 환자를 처리하자마자 모르는 시트를 그대로 지워버린다. 그래서 라이브 파일은 건드리지
않고 별도 파일로 만든다 — 지금, 그리고 파이프라인이 끝난 뒤에도 다시 실행하면 그 시점의
최신 진행 상황으로 새로 만들어진다.

check_landmarks.py를 import하지 않는다 — totalsegmentator/torch까지 불러오는 무거운
모듈이라, 경로 문자열과 시트 읽기만 필요한 이 스크립트엔 과하다.
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data"
SITE_SLUG = {"강남": "gangnam", "신촌": "sinchon"}
DISEASE_COLS = ["HTN", "DM", "CKD"]


# 원본 임상 정리본(raw/{site} 데이터 정리본*.xlsx)에서 환자별 HTN/DM/CKD(있음=1, 없음=0)를 읽는다. 파일이 바뀔 때만 다시 읽는다.
# DLO_Results의 질환 값도 이 정리본에서 온 것이라 같은 정의다(DLO 환자는 값이 완전히 일치).
_ROSTER_SHEET = {"강남": 0, "신촌": "신촌 전체 데이터 값만"}
_roster_cache: dict = {}


def _roster_disease(site: str):
    import glob
    paths = glob.glob(os.path.join(DATA_DIR, "raw", f"{site} 데이터 정리본*.xlsx"))
    if not paths:
        return None
    p = paths[0]; mt = os.path.getmtime(p)
    if _roster_cache.get(site, (None, None))[0] != (p, mt):
        d = pd.read_excel(p, sheet_name=_ROSTER_SHEET[site])
        d["PatientID"] = pd.to_numeric(d["연구등록번호"], errors="coerce")
        d = d.dropna(subset=["PatientID"]); d["PatientID"] = d["PatientID"].astype(int)
        for c in DISEASE_COLS:
            d[c] = d[c].map({"있음": 1, "없음": 0})
        _roster_cache[site] = ((p, mt), d.groupby("PatientID")[DISEASE_COLS].max())   # 중복 행은 "있음" 우선(충돌 0건 확인)
    return _roster_cache[site][1]


# raw/기저질환.xlsx에서 환자별 HTN/DM/CKD를 만든다: CT 촬영월 >= 해당 ICD-10 진단연월최소값이면 1(같은 달 포함), 아니면 0.
_ICD = {"HTN": r"^I1[0-5]", "DM": r"^E1[0-4]", "CKD": r"^N18"}
_HOSP = {"강남": "강남병원", "신촌": "신촌병원"}


def _dx_labels(site: str) -> pd.DataFrame:
    b = pd.read_excel(os.path.join(DATA_DIR, "raw", "기저질환.xlsx"))
    b.columns = ["hosp", "hosp_nm", "id", "sex", "sex_nm", "birth", "visit", "dx", "dx_nm", "icd", "icd_nm", "dxm", "diff"]
    b = b[b.hosp_nm == _HOSP[site]]
    dxm = pd.to_datetime(b.dxm.astype(str)).dt.to_period("M")
    # CT 촬영월 = DICOM StudyDate(extract_ct_study_dates.py가 raw/ct_study_dates.csv로 저장). 진료연월최소값은 첫 진료월이라 촬영월과 다르다.
    c = pd.read_csv(os.path.join(DATA_DIR, "raw", "ct_study_dates.csv"), dtype=str)
    c = c[c.site == site]
    ct = pd.Series(pd.to_datetime(c.study_dates.str.split(",").str[0], format="%Y%m%d").dt.to_period("M").values,
                   index=c.PatientID.astype(int).values)
    out = {}
    for k, pat in _ICD.items():
        m = b.icd.astype(str).str.contains(pat)
        first = dxm[m].groupby(b.id[m]).min()
        out[k] = (ct.reindex(first.index) >= first).astype(int)
    return pd.DataFrame(out)


# 정리본의 기저질환+악성질환 시트에서 환자별 진단 이력(코드/명칭/ICD-10/진단연월을 진단연월 순으로 "; "로 이어붙임)과
# 암진단연월(악성질환 최소값), CT촬영연월(raw/ct_study_dates.csv에서 가장 이른 촬영월(질환 라벨링과 동일))을 만든다.
_DX_COLS = ["진단코드", "진단코드(코드명)", "ICD-10 코드", "ICD-10 코드(코드명)"]
EXTRA_META_COLS = [*_DX_COLS, "진단연월", "CT촬영연월", "암진단연월"]


def _dx_extra(site: str) -> pd.DataFrame:
    import glob
    f = glob.glob(os.path.join(DATA_DIR, "raw", f"{site} 데이터 정리본*.xlsx"))[0]
    cols = ["연구등록번호", *_DX_COLS, "진단연월최소값"]
    mal = pd.read_excel(f, sheet_name="악성질환")[cols]
    b = pd.concat([pd.read_excel(f, sheet_name="기저질환")[cols], mal]).dropna(subset=["연구등록번호"])
    for d in (b, mal):
        d["id"] = d["연구등록번호"].astype(int)
        d["dxm"] = d["진단연월최소값"].astype(str).str[:7]
    g = b.sort_values(["id", "dxm"], kind="stable").groupby("id")
    join = lambda s: "; ".join(s.astype(str))
    out = pd.DataFrame({**{c: g[c].agg(join) for c in _DX_COLS}, "진단연월": g["dxm"].agg(join)})
    mal = mal.dropna(subset=["연구등록번호"])
    out["암진단연월"] = mal.groupby("id").dxm.min()
    c = pd.read_csv(os.path.join(DATA_DIR, "raw", "ct_study_dates.csv"), dtype=str)
    c = c[(c.site == site) & c.study_dates.notna()]
    ct = pd.to_datetime(c.study_dates.str.split(",").str[0], format="%Y%m%d").dt.strftime("%Y-%m")
    out = out.join(pd.Series(ct.values, index=c.PatientID.astype(int).values, name="CT촬영연월"), how="outer")
    return out[EXTRA_META_COLS]


# 조건(metadata 완비, seg_status==ok)에 HTN/DM/CKD 완비, 체성분 total 성공, landmark slice 단조증가까지 모두 만족하는 PatientID만 남긴다.
def _filter_sheets(sheets: dict) -> dict:
    m, lm = sheets["patient_metadata"], sheets.get("landmarks")
    ok = m[["Height", "Weight", "BMI"]].notna().all(axis=1)
    ok &= m["BMI"].between(13, 50)                            # BMI가 극단값(<13 또는 >50)인 환자는 제외한다
    if all(c in m.columns for c in DISEASE_COLS):
        ok &= m[DISEASE_COLS].notna().all(axis=1)             # 질환 정보를 구할 수 없는 환자는 filtered에서 뺀다
    ok &= m["PatientAge"] >= 20                               # 20세 미만은 제외
    if "hw_implausible" in m.columns:
        ok &= m["hw_implausible"] == 0                        # 키/몸무게가 비현실적인 환자는 제외
    ids = set(m.PatientID[ok])
    if lm is not None:
        ids &= set(lm.PatientID[lm.seg_status == "ok"])
        # landmark slice 번호가 T10→치골 순서로 엄격히 증가하지 않는 환자는 제외한다(결측은 건너뜀).
        # liver_dome은 T10과의 상하 위치가 해부학적으로 달라 검사하지 않는다.
        sc = [c for c in lm.columns if c.endswith("_slice") and c != "liver_dome_slice"]
        v = lm[sc].to_numpy(float)
        mono = [(np.diff(r[~np.isnan(r)]) > 0).all() for r in v]
        ids &= set(lm.PatientID[mono])
    return {k: d[d.PatientID.isin(ids)] for k, d in sheets.items()}


def _replace_retry(tmp: str, path: str, tries: int = 120, delay: float = 1.0) -> None:
    import time
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(delay)


def _write_xlsx(out_path: str, sheets: dict) -> None:
    tmp = out_path + ".tmp.xlsx"
    with pd.ExcelWriter(tmp, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    _replace_retry(tmp, out_path)


# landmark 시트(dict)와 metadata를 합쳐 *_landmark_full.xlsx와 *_landmark_filtered.xlsx를 저장한다.
# check_landmarks._write_all이 저장할 때마다 메모리의 시트로 호출한다(엑셀을 다시 읽지 않는다).
# filtered_extra로 받은 시트(*_total)는 filtered에만 넣는다.
def write_full_and_filtered(site: str, slug: str, sheets: dict, filtered_extra: dict | None = None) -> None:
    meta_path = os.path.join(DATA_DIR, site, "metadata", f"{slug}_patient_metadata.xlsx")
    if not os.path.exists(meta_path):
        print(f"[{site}] {meta_path} 없음 — 건너뜀")
        return
    meta = pd.read_excel(meta_path)
    # 질환 유무(HTN/DM/CKD, 1=있음 0=없음)는 DLO_Results에서 PatientID로 붙인다. 명단에 없는 환자는 결측.
    dlo_path = os.path.join(DATA_DIR, site, "metadata", f"{site}_DLO_Results.xlsx")
    if os.path.exists(dlo_path):
        dis = pd.read_excel(dlo_path)[["PatientID", *DISEASE_COLS]].drop_duplicates("PatientID")
        # metadata에 이미 채워 둔 값(fill_metadata_raw.py)은 DLO에 없는 환자에 한해 유지한다.
        old = meta[["PatientID", *DISEASE_COLS]] if set(DISEASE_COLS) <= set(meta.columns) else None
        meta = meta.drop(columns=DISEASE_COLS, errors="ignore").merge(dis, on="PatientID", how="left")
        if old is not None:
            meta[DISEASE_COLS] = meta[DISEASE_COLS].fillna(meta[["PatientID"]].merge(old, on="PatientID", how="left")[DISEASE_COLS])
        meta[DISEASE_COLS] = meta[DISEASE_COLS].astype("Int64")
    # 위에서도 비어 있는 칸만 원본 정리본으로 채운다(이미 있는 값은 바꾸지 않는다).
    ros = _roster_disease(site)
    for c in DISEASE_COLS:
        if ros is None or (c not in meta.columns and False):
            continue
        if c not in meta.columns:
            meta[c] = np.nan
        meta[c] = meta[c].astype("float").fillna(meta["PatientID"].map(ros[c])).astype("Int64")
    # 값이 있는 칸은 기저질환.xlsx 기준(CT 촬영월 >= 진단연월최소값이면 1)으로 다시 라벨링한다. 결측은 유지.
    dx = _dx_labels(site)
    for c in DISEASE_COLS:
        meta[c] = meta["PatientID"].map(dx[c]).fillna(0).astype("Int64").where(meta[c].notna())
    meta = meta.drop(columns=EXTRA_META_COLS, errors="ignore").join(_dx_extra(site), on="PatientID")   # 질환 칼럼 옆에 진단 이력 칼럼 추가
    full = {"patient_metadata": meta, **sheets, **(filtered_extra or {})}   # metadata가 맨 앞 시트, *_total도 full에 포함
    base = os.path.join(DATA_DIR, site, "landmark", f"{slug}_landmark")
    _write_xlsx(base + "_full.xlsx", full)
    filt = _filter_sheets({"patient_metadata": meta, **sheets})
    # *_total은 filtered 환자의 행만 남긴다.
    ids = set(filt["patient_metadata"].PatientID)
    # 체성분 total(VAT/SAT/TAMA) 계산이 실패(seg_status != ok)한 환자는 filtered에서 뺀다.
    for k, d in {**sheets, **(filtered_extra or {})}.items():
        if k.endswith("_total") and k != "aec_total" and "seg_status" in d.columns:
            ids -= set(d.PatientID[d.seg_status.astype(str) != "ok"])
    filt = {k: d[d.PatientID.isin(ids)] for k, d in filt.items()}
    filt.update({k: d[d.PatientID.isin(ids)] for k, d in (filtered_extra or {}).items()})
    m = filt["patient_metadata"]
    if m["hw_implausible"].nunique() <= 1:   # filtered에서는 비현실 값을 이미 걸러 단일 값(0)이라 칼럼을 뺀다
        filt["patient_metadata"] = m.drop(columns="hw_implausible")
    filt.pop("aec_total", None)   # filtered에는 aec_total 시트를 넣지 않는다
    _write_xlsx(base + "_filtered.xlsx", filt)


# 환자 1개 사이트의 landmark 파일 + metadata 파일을 합쳐 full/filtered로 저장한다.
def merge_one(site: str, slug: str) -> None:
    landmark_path = os.path.join(DATA_DIR, site, "landmark", f"{slug}_landmark.xlsx")
    if not os.path.exists(landmark_path):
        print(f"[{site}] {landmark_path} 없음 — 건너뜀")
        return
    sheets = pd.read_excel(landmark_path, sheet_name=None)   # {시트명: DataFrame}
    write_full_and_filtered(site, slug, sheets)
    print(f"[{site}] full/filtered 생성: landmark {len(sheets.get('landmarks', []))}명")


def main() -> None:
    for site, slug in SITE_SLUG.items():
        merge_one(site, slug)


def _self_test() -> None:
    """실제 파일 없이 merge_one의 뼈대(시트 보존 + patient_metadata 추가)만 검증."""
    import tempfile
    global DATA_DIR
    with tempfile.TemporaryDirectory() as d:
        site, slug = "_t", "t"
        os.makedirs(os.path.join(d, site, "landmark"))
        os.makedirs(os.path.join(d, site, "metadata"))
        with pd.ExcelWriter(os.path.join(d, site, "landmark", f"{slug}_landmark.xlsx")) as w:
            pd.DataFrame({"PatientID": [1, 2], "seg_status": ["ok", "ok"]}).to_excel(w, sheet_name="landmarks", index=False)
            pd.DataFrame({"PatientID": [1, 2], "aec_flat": [False, True]}).to_excel(w, sheet_name="aec_total", index=False)
            pd.DataFrame({"PatientID": [1], "anchor": ["L3_center"]}).to_excel(
                w, sheet_name="body_composition", index=False)
        pd.DataFrame({"PatientID": [1, 2], "Height": [170.0, 160.0], "Weight": [60.0, 50.0], "BMI": [20.0, 19.5]}).to_excel(
            os.path.join(d, site, "metadata", f"{slug}_patient_metadata.xlsx"), index=False)

        old_data_dir = DATA_DIR
        DATA_DIR = d
        try:
            merge_one(site, slug)
            out = pd.read_excel(os.path.join(d, site, "landmark", f"{slug}_landmark_full.xlsx"),
                                sheet_name=None)
            flt = pd.read_excel(os.path.join(d, site, "landmark", f"{slug}_landmark_filtered.xlsx"),
                                sheet_name=None)
        finally:
            DATA_DIR = old_data_dir

        assert list(out.keys())[0] == "patient_metadata", "patient_metadata가 맨 앞 시트여야 함"
        assert set(out.keys()) == {"landmarks", "body_composition", "aec_total", "patient_metadata"}
        assert list(flt["landmarks"]["PatientID"]) == [1, 2], "aec_flat은 filtered 조건이 아님"
        assert len(out["landmarks"]) == 2 and len(out["patient_metadata"]) == 2
        assert list(out["patient_metadata"]["Height"]) == [170.0, 160.0]
    print("self-test OK")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
