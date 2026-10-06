"""patient_metadata의 HTN/DM/CKD 컬럼 추가 + Height/Weight/BMI 결측을 raw 정리본으로 채운다.
 - 질환: DLO_Results 우선, 없으면 강남=raw HTN/DM/CKD 컬럼, 신촌='기저질환' 시트 ICD-10(진단 기록 있는 환자만; 없으면 결측)
 - 키/체중: raw '키'/'체중' 시트에서 진료일과 가장 가까운(절댓값 최소) 측정값
"""
import glob, os
import pandas as pd
from make_patient_metadata import DATA_DIR, implausible

ICD = {"HTN": r"^I1[0-5]", "DM": r"^E1[0-4]", "CKD": r"^N18"}
SITES = {"강남": "gangnam", "신촌": "sinchon"}


def nearest(sh: pd.DataFrame, col: str) -> pd.Series:
    s = sh.dropna(subset=["연구등록번호", col]).sort_values("절댓값", kind="stable")
    return s.drop_duplicates("연구등록번호").set_index(s.drop_duplicates("연구등록번호")["연구등록번호"].astype(int))[col]


for site, slug in SITES.items():
    raw = glob.glob(os.path.join(DATA_DIR, "raw", f"{site}*bmi*"))[0]
    x = pd.ExcelFile(raw)
    path = os.path.join(DATA_DIR, site, "metadata", f"{slug}_patient_metadata.xlsx")
    m = pd.read_excel(path).drop(columns=list(ICD), errors="ignore")

    # 질환
    dlo = pd.read_excel(os.path.join(DATA_DIR, site, "metadata", f"{site}_DLO_Results.xlsx"))
    dis = dlo[["PatientID", *ICD]].drop_duplicates("PatientID").set_index("PatientID")
    dx = x.parse("기저질환")
    for k, p in ICD.items():
        v = m.PatientID.map(dis[k])
        if site == "강남":
            r = x.parse(0).drop_duplicates("연구등록번호").set_index("연구등록번호")[k].map({"있음": 1, "없음": 0})
            v = v.fillna(m.PatientID.map(r))
        else:
            ids = set(dx[dx["ICD-10 코드"].astype(str).str.contains(p)]["연구등록번호"])
            has = m.PatientID.isin(dx["연구등록번호"])
            v = v.fillna(m.PatientID.isin(ids).astype(int).where(has))
        m[k] = v.astype("Int64")

    # 키/체중
    hs, ws = x.parse("키"), x.parse("체중")
    h_new = m.PatientID.map(nearest(hs, "측정값"))
    w_new = m.PatientID.map(nearest(ws, "체중" if "체중" in ws.columns else "측정값"))
    nh, nw = m.Height.isna().sum(), m.Weight.isna().sum()
    m["Height"] = m.Height.fillna(h_new)
    m["Weight"] = m.Weight.fillna(w_new)
    ok = m.Height.gt(0) & m.Weight.notna() & m.BMI.isna()
    m.loc[ok, "BMI"] = (m.Weight[ok] / (m.Height[ok] / 100) ** 2).round(2)
    m["hw_implausible"] = [int(implausible(h, w, a, b)) for h, w, a, b in zip(m.Height, m.Weight, m.PatientAge, m.BMI)]
    m.to_excel(path, index=False)
    print(site, "Height", nh, "->", m.Height.isna().sum(), "Weight", nw, "->", m.Weight.isna().sum(),
          "BMI 결측", m.BMI.isna().sum(), "질환 결측", m[list(ICD)].isna().sum().to_dict())
