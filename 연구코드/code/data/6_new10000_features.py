"""
[목적]
새 데이터 10000명 코호트를 gangnam/sinchon_final_dataset.xlsx와 같은 구조(metadata 시트)로
저장한다. 체성분(VAT/SAT/TAMA/SMI 등) 분할 데이터는 아직 없어 metadata에 포함하지 않는다
(TotalSegmentator 기반 추출은 별도 진행 — 9_merge_body_composition.py가 나중에 병합).

(예전에는 aec_cropped_novatsat.xlsx의 metadata_cleaned 시트를 썼지만, 그 파일을 더 이상
쓰지 않기로 해 원본 예후데이터.xlsx에서 직접 만든다.)

[처리 흐름]
  1. 성별/나이/신장/체중: 예후데이터.xlsx의 "1-2. 키몸무게BMI_예진정보" 시트(연구등록번호당
     1행으로 이미 정리됨)를 기준으로 쓰고, 그 시트에 없는 환자만 "1-1.키몸무게BMI_임상관찰"
     시트(임상관찰코드(코드명)가 "신체계측/신장(cm)", "신체계측/체중(kg)"인 행을 pivot)로
     보완한다. BMI = 체중 / (신장/100)^2, 소수 둘째 자리 반올림 (5_merged_features.py의
     gangnam 계산식과 동일).
  2. 질환 플래그(DM/HTN/DL/Osteoporosis/MI/Stroke): 예후데이터.xlsx는 질환별로 별도
     f_u(follow-up) 시트에 진단 이력만 있고 "있음/없음" 컬럼은 없다 — 해당 f_u 시트에
     PatientID가 한 번이라도 있으면 1, 없으면 0으로 매긴다(진단 시점은 구분하지 않음 —
     gangnam/신촌의 fill_dlo_results.py가 쓰는 "있음=1, 없음=0" 방식과 동일한 원칙).
  3. best_series.xlsx(7_select_best_series.py 산출물)에서 Series_Desc/Manufacturer/kVp를
     병합한다(kVp는 gangnam_final_dataset.xlsx의 CANONICAL_COLUMN_ORDER와 동일하게
     Manufacturer 다음에 둔다). best_series와 위 anthropometrics는 대상 환자 집합이
     서로 다를 수 있어 PatientID 교집합으로 맞춘다.
  4. BMI IQR 이상치 제거: 5_merged_features.py의 gangnam STEP 1-③과 동일한 기준
     (Q1-1.5*IQR ~ Q3+1.5*IQR 범위 밖이면 제외)을 BMI 컬럼에만 적용한다. 결측(NaN)은
     이상치로 세지 않고 그대로 둔다.

[출력: new10000_final_dataset.xlsx]
  metadata 시트: PatientID, Series_Desc, Manufacturer, kVp, PatientAge, PatientSex, Height,
    Weight, BMI, DM, HTN, DL, Osteoporosis, MI, Stroke
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "새 데이터 10000명"

PROGNOSIS_PATH = DATA_DIR / "예후데이터.xlsx"
BEST_SERIES_PATH = DATA_DIR / "best_series.xlsx"
OUT_PATH = DATA_DIR / "new10000_final_dataset.xlsx"

DISEASE_SHEETS = {
    "DM": "5-1. DM_f_u", "HTN": "5-2. HTN_f_u", "DL": "5-3. DL_f_u",
    "Osteoporosis": "5-4. 골다공증_f_u", "MI": "5-5. 심근경색_f_u", "Stroke": "5-6. 뇌졸중_f_u",
}


def build_anthropometrics() -> pd.DataFrame:
    d12 = pd.read_excel(PROGNOSIS_PATH, sheet_name="1-2. 키몸무게BMI_예진정보",
                         usecols=["연구등록번호", "성별", "처방시나이", "신장", "체중"])
    d12 = d12.rename(columns={"연구등록번호": "PatientID", "성별": "PatientSex",
                               "처방시나이": "PatientAge", "신장": "Height", "체중": "Weight"})
    d12 = d12.drop_duplicates(subset="PatientID").set_index("PatientID")

    d11 = pd.read_excel(PROGNOSIS_PATH, sheet_name="1-1.키몸무게BMI_임상관찰",
                         usecols=["연구등록번호", "성별", "처방시나이", "임상관찰코드(코드명)", "측정값"])
    piv = d11.pivot_table(index="연구등록번호", columns="임상관찰코드(코드명)", values="측정값", aggfunc="first")
    d11_flat = d11.drop_duplicates(subset="연구등록번호").set_index("연구등록번호")[["성별", "처방시나이"]]
    d11_flat = d11_flat.rename(columns={"성별": "PatientSex", "처방시나이": "PatientAge"})
    d11_flat["Height"] = piv.get("신체계측/신장(cm)")
    d11_flat["Weight"] = piv.get("신체계측/체중(kg)")
    d11_flat.index.name = "PatientID"

    only_in_11 = d11_flat.loc[d11_flat.index.difference(d12.index)]
    merged = pd.concat([d12, only_in_11])
    for col in ("PatientSex", "PatientAge", "Height", "Weight"):
        merged[col] = merged[col].combine_first(d11_flat[col])

    merged["Height"] = pd.to_numeric(merged["Height"], errors="coerce")
    merged["Weight"] = pd.to_numeric(merged["Weight"], errors="coerce")
    merged["BMI"] = (merged["Weight"] / (merged["Height"] / 100) ** 2).round(2)
    return merged.reset_index()


def drop_bmi_outliers(df_meta: pd.DataFrame) -> pd.DataFrame:
    """5_merged_features.py의 gangnam STEP 1-③과 같은 IQR 방식(Q1-1.5*IQR ~ Q3+1.5*IQR
    범위 밖 제외)이나, BMI 컬럼에만 적용한다. NaN은 이상치로 세지 않고 그대로 남긴다."""
    q1, q3 = df_meta["BMI"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier = df_meta["BMI"].notna() & ~df_meta["BMI"].between(lower, upper)
    print(f"  BMI IQR 이상치 {outlier.sum()}명 제외 (범위: [{lower:.1f}, {upper:.1f}])")
    if outlier.any():
        df_meta = df_meta[~outlier].reset_index(drop=True)
    return df_meta


def build_disease_flags(patient_ids: set[int]) -> pd.DataFrame:
    df = pd.DataFrame({"PatientID": sorted(patient_ids)}).set_index("PatientID")
    for col, sheet in DISEASE_SHEETS.items():
        ids = set(pd.read_excel(PROGNOSIS_PATH, sheet_name=sheet, usecols=["연구등록번호"])
                   ["연구등록번호"].dropna().astype(int))
        df[col] = df.index.isin(ids).astype(int)
    return df.reset_index()


def main():
    best = pd.read_excel(BEST_SERIES_PATH, usecols=["PatientID", "series_desc", "manufacturer", "kVp"])
    anthro = build_anthropometrics()

    common_ids = set(anthro["PatientID"]) & set(best["PatientID"])
    print(f"anthropometrics {len(anthro)}명 / best_series {len(best)}명 → 교집합 {len(common_ids)}명")

    df_meta = anthro[anthro["PatientID"].isin(common_ids)].merge(
        best.rename(columns={"series_desc": "Series_Desc", "manufacturer": "Manufacturer"}),
        on="PatientID", how="left")
    df_meta = df_meta.merge(build_disease_flags(common_ids), on="PatientID", how="left")

    front = ["PatientID", "Series_Desc", "Manufacturer", "kVp", "PatientAge", "PatientSex", "Height", "Weight", "BMI"]
    df_meta = df_meta[front + list(DISEASE_SHEETS.keys())]

    df_meta = drop_bmi_outliers(df_meta)

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        df_meta.to_excel(writer, sheet_name="metadata", index=False)
        print(f"  [metadata] {len(df_meta)}행 x {len(df_meta.columns)}열")

    print(f"저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
