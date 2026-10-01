"""
{slug}_landmark.xlsx(landmarks/body_composition/aec_total/aec_landmark 4개 시트)와
{slug}_patient_metadata.xlsx(PatientSex/PatientAge/Height/Weight/BMI/hw_implausible)를
합쳐 {slug}_landmark_full.xlsx(시트 5개)와 세 조건 충족 환자만 남긴 _filtered.xlsx로 저장한다. 강남/신촌 둘 다 처리한다.

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

import pandas as pd

DATA_DIR = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data"
SITE_SLUG = {"강남": "gangnam", "신촌": "sinchon"}
DISEASE_COLS = ["HTN", "DM", "CKD"]


# 세 조건(metadata 완비, seg_status==ok, aec_flat==False)을 모두 만족하는 PatientID만 남긴다.
def _filter_sheets(sheets: dict) -> dict:
    m, lm, at = sheets["patient_metadata"], sheets.get("landmarks"), sheets.get("aec_total")
    ids = set(m.PatientID[m[["Height", "Weight", "BMI"]].notna().all(axis=1)])
    if lm is not None:
        ids &= set(lm.PatientID[lm.seg_status == "ok"])
    if at is not None:
        ids &= set(at.PatientID[at.aec_flat.astype(str).str.lower() == "false"])
    return {k: d[d.PatientID.isin(ids)] for k, d in sheets.items()}


def _write_xlsx(out_path: str, sheets: dict) -> None:
    tmp = out_path + ".tmp.xlsx"
    with pd.ExcelWriter(tmp, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    os.replace(tmp, out_path)


# landmark 시트(dict)와 metadata를 합쳐 *_landmark_full.xlsx와 *_landmark_filtered.xlsx를 저장한다.
# check_landmarks._write_all이 저장할 때마다 메모리의 시트로 호출한다(엑셀을 다시 읽지 않는다).
def write_full_and_filtered(site: str, slug: str, sheets: dict) -> None:
    meta_path = os.path.join(DATA_DIR, site, "metadata", f"{slug}_patient_metadata.xlsx")
    if not os.path.exists(meta_path):
        print(f"[{site}] {meta_path} 없음 — 건너뜀")
        return
    meta = pd.read_excel(meta_path)
    # 질환 유무(HTN/DM/CKD, 1=있음 0=없음)는 DLO_Results에서 PatientID로 붙인다. 명단에 없는 환자는 결측.
    dlo_path = os.path.join(DATA_DIR, site, "metadata", f"{site}_DLO_Results.xlsx")
    if os.path.exists(dlo_path):
        dis = pd.read_excel(dlo_path)[["PatientID", *DISEASE_COLS]].drop_duplicates("PatientID")
        meta = meta.drop(columns=DISEASE_COLS, errors="ignore").merge(dis, on="PatientID", how="left")
        meta[DISEASE_COLS] = meta[DISEASE_COLS].astype("Int64")
    full = {"patient_metadata": meta, **sheets}   # metadata가 맨 앞 시트
    base = os.path.join(DATA_DIR, site, "landmark", f"{slug}_landmark")
    _write_xlsx(base + "_full.xlsx", full)
    _write_xlsx(base + "_filtered.xlsx", _filter_sheets(full))


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
        assert list(flt["landmarks"]["PatientID"]) == [1], "aec_flat==False인 환자만 남아야 함"
        assert len(out["landmarks"]) == 2 and len(out["patient_metadata"]) == 2
        assert list(out["patient_metadata"]["Height"]) == [170.0, 160.0]
    print("self-test OK")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
