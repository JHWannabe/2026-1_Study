"""
{SITE}_DLO_Results.xlsx의 빈 칼럼(Manufacturer, Series_Desc, Height, Weight, BMI, HTN, DM, CKD)을 채운다.

- Height, Weight : radiation_data_260421_VBcohort.xlsx의 '임상관찰_가장가까운날짜만' /
  '예진_가장가까운날짜만' 시트를 모두 참고해, CT 촬영일(진료일)과의 날짜차이가
  더 작은 쪽의 실측값을 채택한다. (Height/Weight 각각 독립적으로 판단)
- BMI : Weight / (Height/100)^2, 소수 둘째자리 반올림
- HTN, DM, CKD : "{SITE} 데이터 정리본...xlsx"에서 연구등록번호로 매칭 (있음=1, 없음=0)
  강남은 기본 시트, 신촌은 '신촌 전체 데이터 값만' 시트를 사용한다.
- Manufacturer, Series_Desc : 기존에 OCR로 채워져 있던 Series_Desc 값은 해당 환자 폴더 아래
  여러 시리즈(PRE/HVP/PORTAL 등) 중 어떤 것이었는지 찾기 위한 힌트로만 쓰고, 실제 값은
  E:\\영상제공\\{SITE}\\{SITE}_axial\\{PatientID}\\ 아래 매칭된 시리즈 폴더의 DICOM에서
  ManufacturerModelName / SeriesDescription 태그를 직접 읽어 저장한다. (OCR 오독 방지,
  ground truth 확보. SeriesDescription의 연속 공백은 단일 공백으로 정리)
  - 신촌 : 힌트는 "{SITE}_DLO_Results_Unique.xlsx"의 raw 시트 Series_Desc에서 가져온다.
  - 강남 : 별도 OCR 백업 파일 없이, DLO_Results.xlsx 자신의 기존 Series_Desc 값을 힌트로 쓴다.
"""

import difflib
import os
import re

import numpy as np
import openpyxl
import pandas as pd
import pydicom

DATA_DIR = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data"
RAD_PATH = rf"{DATA_DIR}\raw\radiation_data_260421_VBcohort.xlsx"

SITE_CONFIG = {
    "강남": {
        "dlo_path": rf"{DATA_DIR}\강남\metadata\강남_DLO_Results.xlsx",
        "jb_path": rf"{DATA_DIR}\raw\강남 데이터 정리본 20260315 bmi 없는것들 있음.xlsx",
        "jb_sheet": 0,
        "mfr_seed_path": None,
        "axial_dir": r"E:\영상제공\강남\강남_axial",
    },
    "신촌": {
        "dlo_path": rf"{DATA_DIR}\신촌\metadata\신촌_DLO_Results.xlsx",
        "jb_path": rf"{DATA_DIR}\raw\신촌 데이터 정리본 20260316 bmi 없는것들 있음.xlsx",
        "jb_sheet": "신촌 전체 데이터 값만",
        "mfr_seed_path": None,
        "axial_dir": r"E:\영상제공\신촌\신촌_axial",
    },
}

HTN_DM_CKD_MAP = {"있음": 1, "없음": 0}

COL_IDX = {
    "PatientID": 1, "Manufacturer": 2, "Series_Desc": 3,
    "Height": 7, "Weight": 8, "BMI": 9, "HTN": 12, "DM": 13, "CKD": 14,
}


def build_height_weight(patient_ids: pd.Series) -> pd.DataFrame:
    imsang = pd.read_excel(RAD_PATH, sheet_name="임상관찰_가장가까운날짜만")
    yejin = pd.read_excel(RAD_PATH, sheet_name="예진_가장가까운날짜만")

    h_im = imsang[imsang["임상관찰코드(코드명)"] == "신체계측/신장(cm)"][
        ["연구등록번호", "측정값", "진료일로부터(일)_절대값"]
    ].rename(columns={"측정값": "Height_im", "진료일로부터(일)_절대값": "diff_h_im"})
    w_im = imsang[imsang["임상관찰코드(코드명)"] == "신체계측/체중(kg)"][
        ["연구등록번호", "측정값", "진료일로부터(일)_절대값"]
    ].rename(columns={"측정값": "Weight_im", "진료일로부터(일)_절대값": "diff_w_im"})
    ye = yejin[["연구등록번호", "신장", "체중", "진료일로부터(일)_절대값"]].rename(
        columns={"신장": "Height_ye", "체중": "Weight_ye", "진료일로부터(일)_절대값": "diff_ye"}
    )

    m = pd.DataFrame({"PatientID": patient_ids})
    m = m.merge(h_im, left_on="PatientID", right_on="연구등록번호", how="left").drop(columns="연구등록번호")
    m = m.merge(w_im, left_on="PatientID", right_on="연구등록번호", how="left").drop(columns="연구등록번호")
    m = m.merge(ye, left_on="PatientID", right_on="연구등록번호", how="left").drop(columns="연구등록번호")

    for c in ("Height_im", "Weight_im", "Height_ye", "Weight_ye"):
        m[c] = pd.to_numeric(m[c], errors="coerce")

    def pick(row, val_a, diff_a, val_b, diff_b):
        a_ok = pd.notna(row[val_a]) and pd.notna(row[diff_a])
        b_ok = pd.notna(row[val_b]) and pd.notna(row[diff_b])
        if a_ok and b_ok:
            return row[val_a] if row[diff_a] <= row[diff_b] else row[val_b]
        if a_ok:
            return row[val_a]
        if b_ok:
            return row[val_b]
        return np.nan

    m["Height"] = m.apply(lambda r: pick(r, "Height_im", "diff_h_im", "Height_ye", "diff_ye"), axis=1)
    m["Weight"] = m.apply(lambda r: pick(r, "Weight_im", "diff_w_im", "Weight_ye", "diff_ye"), axis=1)
    m["BMI"] = (m["Weight"] / (m["Height"] / 100) ** 2).round(2)
    return m[["PatientID", "Height", "Weight", "BMI"]]


def build_disease_flags(jb_path: str, jb_sheet) -> pd.DataFrame:
    jb = pd.read_excel(jb_path, sheet_name=jb_sheet)
    jb = jb.drop_duplicates(subset=["연구등록번호", "HTN", "DM", "CKD"])[["연구등록번호", "HTN", "DM", "CKD"]]
    for col in ("HTN", "DM", "CKD"):
        jb[col] = jb[col].map(HTN_DM_CKD_MAP)
    return jb.rename(columns={"연구등록번호": "PatientID"})


# ── Manufacturer / Series_Desc : DICOM에서 직접 추출 (신촌) ─────────────────────

def _norm_series_name(s) -> str:
    """폴더명/OCR 텍스트를 느슨하게 비교하기 위해 영숫자만 남기고 정규화한다."""
    s = str(s).strip().lower()
    s = re.sub(r"\.\.\.$", "", s)  # OCR이 잘라먹은 말줄임표 제거
    s = re.sub(r"[^0-9a-z가-힣]", "", s)
    return s


def find_series_folder(patient_dir: str, seed_series_desc) -> str | None:
    """OCR로 얻은 Series_Desc를 힌트 삼아 실제 시리즈 폴더명을 찾는다."""
    subs = [s for s in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, s))]
    if not subs:
        return None
    if pd.isna(seed_series_desc):
        return subs[0] if len(subs) == 1 else None

    target = _norm_series_name(seed_series_desc)
    for s in subs:
        if _norm_series_name(s) == target:
            return s
    for s in subs:
        ns = _norm_series_name(s)
        if ns.startswith(target) or target.startswith(ns):
            return s
    if len(subs) == 1:
        return subs[0]

    # OCR 특유의 숫자/문자 오독(0/O, 1/I/l 등)에 대비한 최종 fuzzy 매칭
    norm_subs = {_norm_series_name(s): s for s in subs}
    best = difflib.get_close_matches(target, list(norm_subs.keys()), n=1, cutoff=0.75)
    return norm_subs[best[0]] if best else None


def read_dicom_tags(series_dir: str) -> tuple[str | None, str | None]:
    files = sorted(f for f in os.listdir(series_dir) if f.lower().endswith(".dcm"))
    if not files:
        files = os.listdir(series_dir)
    if not files:
        return None, None
    ds = pydicom.dcmread(os.path.join(series_dir, files[0]), stop_before_pixels=True)
    mfr = ds.get("ManufacturerModelName")
    series_desc = ds.get("SeriesDescription")
    mfr = str(mfr) if mfr is not None else None
    series_desc = re.sub(r"\s+", " ", str(series_desc)).strip() if series_desc is not None else None
    return mfr, series_desc


def build_manufacturer_from_dicom(dlo: pd.DataFrame, mfr_seed_path: str | None, axial_dir: str) -> pd.DataFrame:
    if mfr_seed_path:
        # 신촌 : 별도 OCR 백업 파일(raw 시트)의 Series_Desc를 힌트로 사용
        seed = pd.read_excel(mfr_seed_path, sheet_name="raw")[["PatientID", "Series_Desc"]]
        seed = seed.rename(columns={"Series_Desc": "SeedSeries"})
        df = dlo[["PatientID"]].merge(seed, on="PatientID", how="left")
    else:
        # 강남 : DLO_Results.xlsx 자신의 기존 Series_Desc를 힌트로 사용
        df = dlo[["PatientID", "Series_Desc"]].rename(columns={"Series_Desc": "SeedSeries"})

    rows = []
    for _, r in df.iterrows():
        pid = int(r["PatientID"])
        patient_dir = os.path.join(axial_dir, str(pid))
        mfr = series_desc = None
        if os.path.isdir(patient_dir):
            folder = find_series_folder(patient_dir, r["SeedSeries"])
            if folder is not None:
                mfr, series_desc = read_dicom_tags(os.path.join(patient_dir, folder))
        rows.append((pid, mfr, series_desc))

    return pd.DataFrame(rows, columns=["PatientID", "Manufacturer", "Series_Desc"])


# ── 사이트별 실행 ───────────────────────────────────────────────────────────

def fill_site(site: str):
    cfg = SITE_CONFIG[site]
    dlo = pd.read_excel(cfg["dlo_path"])

    m = build_height_weight(dlo["PatientID"])
    m = m.merge(build_disease_flags(cfg["jb_path"], cfg["jb_sheet"]), on="PatientID", how="left")

    has_mfr = bool(cfg["axial_dir"])
    if has_mfr:
        m = m.merge(
            build_manufacturer_from_dicom(dlo, cfg["mfr_seed_path"], cfg["axial_dir"]),
            on="PatientID", how="left",
        )

    lookup = {}
    for _, r in m.iterrows():
        entry = {
            "Height": None if pd.isna(r["Height"]) else float(r["Height"]),
            "Weight": None if pd.isna(r["Weight"]) else float(r["Weight"]),
            "BMI": None if pd.isna(r["BMI"]) else float(r["BMI"]),
            "HTN": None if pd.isna(r["HTN"]) else int(r["HTN"]),
            "DM": None if pd.isna(r["DM"]) else int(r["DM"]),
            "CKD": None if pd.isna(r["CKD"]) else int(r["CKD"]),
        }
        if has_mfr:
            entry["Manufacturer"] = None if pd.isna(r["Manufacturer"]) else r["Manufacturer"]
            entry["Series_Desc"] = None if pd.isna(r["Series_Desc"]) else r["Series_Desc"]
        lookup[int(r["PatientID"])] = entry

    wb = openpyxl.load_workbook(cfg["dlo_path"])
    ws = wb.active

    fields = ["Height", "Weight", "BMI", "HTN", "DM", "CKD"]
    if has_mfr:
        fields = ["Manufacturer", "Series_Desc"] + fields

    filled = 0
    for row in range(2, ws.max_row + 1):
        pid = ws.cell(row=row, column=COL_IDX["PatientID"]).value
        if pid is None:
            continue
        data = lookup.get(int(pid))
        if data is None:
            continue
        for field in fields:
            val = data[field]
            if val is not None:
                ws.cell(row=row, column=COL_IDX[field], value=val)
        filled += 1

    wb.save(cfg["dlo_path"])
    print(f"[{site}] {filled}행 처리 완료 -> {cfg['dlo_path']}")


def main():
    for site in ("강남", "신촌"):
        fill_site(site)


if __name__ == "__main__":
    main()
