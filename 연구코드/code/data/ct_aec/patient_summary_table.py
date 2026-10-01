"""
"통합 문서.xlsx"의 각 시트에서 patientID(=연구등록번호)당 핵심 데이터를 추출하여
patientID 1개당 1행으로 정리한 요약 테이블을 만든다.

시트별 처리 방식 (환자당 여러 행이 존재하는 시트는 요약/집계함):
- 대상자명단        : CT_검사명 (첫 행 기준)
- 1-1 임상관찰      : 임상관찰코드(신장/체중/BMI)를 열로 피벗, CT 촬영일과 가장 가까운 기록 사용(일시차2 최소)
- 1-2 예진정보      : 신장, 체중 (직접 컬럼, CT 촬영일과 가장 가까운 기록, 1-1 값의 보완용)
- 1-3 간호정보조사  : 흡연력/음주력 (서식일자 기준 최신값)
- 2. 과거력         : 진단 개수 + 고유 진단명 목록(세미콜론 구분)
- 4-1 입원기록      : 입원 횟수 + 고유 진단명 목록
- 4-2 응급실기록    : 응급실 방문 횟수 + 고유 진단명 목록
- 4-3 사망          : 사망 여부, 사망의 종류
- 4-4 Last f_u date : 전체 환자 커버 -> 성별(M/F)/나이 추출용으로만 사용
- 5-1~5-6 f_u       : 질환별 진단 여부(0/1) (DM/HTN/DL/골다공증/심근경색/뇌졸중)
- 6-1 LAB추적       : 검사종목별(13종) 수치결과를 열로 피벗
"""

import os
import pandas as pd
from check_patientid_registration import BASE_DIR, EXCEL_PATH, get_folder_patient_ids

OUTPUT_XLSX = os.path.join(BASE_DIR, "patientID_summary_table.xlsx")
REG = "연구등록번호"

CONDITION_SHEETS = {
    "5-1. DM_f_u": ("당뇨병", "DM 최초진단일"),
    "5-2. HTN_f_u": ("고혈압", "HTN 최초진단일"),
    "5-3. DL_f_u": ("이상지질혈증", "Dyslipidemia 최초진단일"),
    "5-4. 골다공증_f_u": ("골다공증", "Osteoporosis 최초진단일"),
    "5-5. 심근경색_f_u": ("심근경색", "심근경색 최초진단일"),
    "5-6. 뇌졸중_f_u": ("뇌졸중", "뇌졸중 최초진단일"),
}


def read_sheet(sheet_name):
    print(f"  로드: {sheet_name}")
    return pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)


def to_patient_id(df):
    df[REG] = pd.to_numeric(df[REG], errors="coerce")
    return df.dropna(subset=[REG]).assign(**{REG: lambda d: d[REG].astype(int)})


def main():
    print(f"폴더 스캔 중: {BASE_DIR}")
    patient_ids = sorted(get_folder_patient_ids(BASE_DIR))
    print(f"patientID 개수: {len(patient_ids)}")

    df = pd.DataFrame({"patientID": patient_ids})

    print("시트 처리 중...")

    # 대상자명단: CT_검사명
    s = to_patient_id(read_sheet("대상자명단"))
    s = s.drop_duplicates(subset=REG, keep="first")
    s = s.rename(columns={REG: "patientID", "처방코드(코드명)": "CT_검사명"})
    df = df.merge(s[["patientID", "CT_검사명"]], on="patientID", how="left")

    # 4-4 Last f_u date: 성별/나이(전체 환자 커버)
    s44 = to_patient_id(read_sheet("4-4. Last f_u date"))
    s44 = s44.drop_duplicates(subset=REG, keep="first")
    s44 = s44[[REG, "성별", "처방시나이"]]
    s44 = s44.rename(columns={REG: "patientID", "처방시나이": "나이"})
    df = df.merge(s44, on="patientID", how="left")

    # 1-1 임상관찰: 신장/체중/BMI 피벗 (CT 촬영일과 가장 가까운 기록 = 일시차2 최솟값)
    s11 = to_patient_id(read_sheet("1-1.키몸무게BMI_임상관찰"))
    s11 = s11.sort_values("일시차2").drop_duplicates(subset=[REG, "임상관찰코드(코드명)"], keep="first")
    piv11 = s11.pivot(index=REG, columns="임상관찰코드(코드명)", values="측정값")
    piv11 = piv11.rename(columns={
        "신체계측/신장(cm)": "신장_1",
        "신체계측/체중(kg)": "체중_1",
        "신체계측/BMI": "BMI",
    }).reset_index().rename(columns={REG: "patientID"})
    keep_cols = ["patientID"] + [c for c in ["신장_1", "체중_1", "BMI"] if c in piv11.columns]
    df = df.merge(piv11[keep_cols], on="patientID", how="left")

    # 1-2 예진정보: 신장/체중 (CT 촬영일과 가장 가까운 기록, 1-1 보완용)
    s12 = to_patient_id(read_sheet("1-2. 키몸무게BMI_예진정보"))
    s12 = s12.sort_values("일시차2").drop_duplicates(subset=REG, keep="first")
    s12 = s12.rename(columns={REG: "patientID", "신장": "신장_2", "체중": "체중_2"})
    df = df.merge(s12[["patientID", "신장_2", "체중_2"]], on="patientID", how="left")

    df["신장"] = df["신장_2"].combine_first(df.get("신장_1"))
    df["체중"] = df["체중_2"].combine_first(df.get("체중_1"))
    df = df.drop(columns=[c for c in ["신장_1", "체중_1", "신장_2", "체중_2"] if c in df.columns])

    # 1-3 간호정보조사: 흡연력/음주력 (최신값)
    s13 = to_patient_id(read_sheet("1-3.흡연,음주_간호정보조사"))
    s13 = s13[s13["Attr코드(코드명)"].isin(["흡연력(간호 정보 조사2.0)", "음주력(간호 정보 조사2.0)"])]
    s13 = s13.sort_values("서식일자").drop_duplicates(subset=[REG, "Attr코드(코드명)"], keep="last")
    piv13 = s13.pivot(index=REG, columns="Attr코드(코드명)", values="항목값")
    piv13 = piv13.rename(columns={
        "흡연력(간호 정보 조사2.0)": "흡연력",
        "음주력(간호 정보 조사2.0)": "음주력",
    }).reset_index().rename(columns={REG: "patientID"})
    keep_cols = ["patientID"] + [c for c in ["흡연력", "음주력"] if c in piv13.columns]
    df = df.merge(piv13[keep_cols], on="patientID", how="left")

    # 2. 과거력: 진단 개수 + 고유 진단명 목록
    s2 = to_patient_id(read_sheet("2. 과거력"))
    agg2 = s2.groupby(REG)["진단코드(코드명)"].agg(
        과거력_진단개수="nunique",
        과거력_진단목록=lambda x: "; ".join(sorted(set(x.dropna()))),
    ).reset_index().rename(columns={REG: "patientID"})
    df = df.merge(agg2, on="patientID", how="left")

    # 4-1 입원기록: 입원 횟수 + 고유 진단명 목록
    s41 = to_patient_id(read_sheet("4-1. 입원기록_f_u"))
    agg41 = s41.groupby(REG).agg(
        입원횟수=("연구내원번호_1", "nunique"),
        입원진단목록=("진단코드(코드명)", lambda x: "; ".join(sorted(set(x.dropna())))),
    ).reset_index().rename(columns={REG: "patientID"})
    df = df.merge(agg41, on="patientID", how="left")

    # 4-2 응급실기록: 방문 횟수 + 고유 진단명 목록
    s42 = to_patient_id(read_sheet("4-2. 응급실기록_f_u"))
    agg42 = s42.groupby(REG).agg(
        응급실방문횟수=("연구내원번호_1", "nunique"),
        응급실진단목록=("진단코드(코드명)", lambda x: "; ".join(sorted(set(x.dropna())))),
    ).reset_index().rename(columns={REG: "patientID"})
    df = df.merge(agg42, on="patientID", how="left")

    # 4-3 사망: 사망여부, 사망의 종류 (사망일시는 제외)
    s43 = to_patient_id(read_sheet("4-3.사망일시, 사망원인_f_u"))
    s43 = s43[s43["Attr코드(코드명)"].isin(["사망일시", "사망의 종류"])]
    s43 = s43.drop_duplicates(subset=[REG, "Attr코드(코드명)"], keep="first")
    piv43 = s43.pivot(index=REG, columns="Attr코드(코드명)", values="항목값")
    piv43 = piv43.rename(columns={"사망의 종류": "사망의_종류"}).reset_index().rename(columns={REG: "patientID"})
    piv43["사망여부"] = piv43["사망일시"].notna().astype(int) if "사망일시" in piv43.columns else 0
    keep_cols = ["patientID", "사망여부"] + (["사망의_종류"] if "사망의_종류" in piv43.columns else [])
    df = df.merge(piv43[keep_cols], on="patientID", how="left")
    df["사망여부"] = df["사망여부"].fillna(0).astype(int)

    # 5-1~5-6 만성질환 f_u: 진단 여부(0/1)만 (최초진단일은 제외)
    for sheet_name, (label, date_col) in CONDITION_SHEETS.items():
        s = to_patient_id(read_sheet(sheet_name))
        agg = s.groupby(REG)[date_col].min().reset_index()
        agg = agg.rename(columns={REG: "patientID", date_col: "_최초진단일"})
        agg[f"{label}_여부"] = agg["_최초진단일"].notna().astype(int)
        df = df.merge(agg[["patientID", f"{label}_여부"]], on="patientID", how="left")
        df[f"{label}_여부"] = df[f"{label}_여부"].fillna(0).astype(int)

    # 6-1 LAB추적: 검사종목별 수치결과 피벗
    s61 = to_patient_id(read_sheet("6-1. LAB추적_일시차최소"))
    s61 = s61.drop_duplicates(subset=[REG, "처방코드_1(코드명)"], keep="first")
    piv61 = s61.pivot(index=REG, columns="처방코드_1(코드명)", values="수치결과")
    piv61 = piv61.reset_index().rename(columns={REG: "patientID"})
    df = df.merge(piv61, on="patientID", how="left")

    df.to_excel(OUTPUT_XLSX, index=False, sheet_name="summary")
    print(f"\n저장 완료: {OUTPUT_XLSX} ({len(df)}행 x {len(df.columns)}열)")


if __name__ == "__main__":
    main()
