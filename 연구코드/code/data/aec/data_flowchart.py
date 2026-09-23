"""
논문용 data flowchart(STROBE 스타일) 작성을 위해, 강남/신촌 각각의 단계별 대상 수·제외 수를
집계해 엑셀로 저장한다.

[단계 정의]
  1. 초기 대상            : {site}_DLO_Results.xlsx의 전체 PatientID
  2. DICOM 시리즈 매칭 제외 : z_bounds.xlsx에서 seg_status가 no_dicom/no_series인 환자
     (E:\\영상제공 폴더 자체가 없거나, Series_Desc와 일치하는 시리즈 폴더를 찾지 못한 경우)
  3. AEC 신호 제외         : {site}_aec_total.xlsx에 없는 환자 (신호가 전부 NaN / CV<0.05 /
     range<10mA / R²≥0.95로 거의 직선인 경우 — liver~pubis 구간 크롭과는 별개 기준)
  4. Segmentation 결과     : z_bounds.xlsx seg_status 분포 (ok / pubis_missing / liver_missing /
     both_missing / invalid_bounds / invalid_volume_match / nifti_fail / error / failed)
     → seg_status=='ok'인 환자만 liver~pubis crop 및 VAT/SAT 산출 대상이 된다.
  5. 체성분(VAT/SAT) 결과  : {site}_body_composition.xlsx seg_status 분포
     (ok인 환자만 DLO_Results.xlsx의 VAT(내장지방)_SUM / SAT(피하지방)_SUM에 값이 채워짐)
  6. 최종 ML 데이터셋({gangnam|sinchon}_final_dataset.xlsx, 5_merged_features.py 산출)
     : {gangnam|sinchon}_final_dataset_stats.json 기준 — Height/Weight 결측, 20세 미만,
       Height/Weight/BMI IQR 이상치, AEC Crop 변동성 없음을 제외한 최종 교집합 행 수

주의: 2번(DICOM 매칭)과 3번(AEC 신호)은 서로 독립적인 제외 기준이라 순서대로 걸러지는
  하나의 깔때기가 아니다. 각 기준별로 "전체 중 제외된 수"를 따로 집계한다.
  extract_liver_pubis_aec.py 실행이 진행 중이면 이 스크립트를 다시 실행해 최신 수치로 갱신한다.

[출력] 연구코드/data/data_flowchart.xlsx (사이트별 시트 + summary 시트)
"""

import json
import os

import pandas as pd

DATA_DIR = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data"
SITES    = ["강남", "신촌"]
SITE_EN  = {"강남": "gangnam", "신촌": "sinchon"}
OUT_PATH = rf"{DATA_DIR}\data_flowchart.xlsx"


def site_paths(site: str) -> dict:
    return {
        "dlo_path":     rf"{DATA_DIR}\{site}\metadata\{site}_DLO_Results.xlsx",
        "aec_total":    rf"{DATA_DIR}\{site}\aec\{site}_aec_total.xlsx",
        "z_bounds":     rf"{DATA_DIR}\{site}\aec\{site}_z_bounds.xlsx",
        "body_comp":    rf"{DATA_DIR}\{site}\aec\{site}_body_composition.xlsx",
        "final_stats":  rf"{DATA_DIR}\{site}\{SITE_EN[site]}_final_dataset_stats.json",
    }


def build_site_flow(site: str) -> list[dict]:
    paths = site_paths(site)
    rows: list[dict] = []

    dlo_n = pd.read_excel(paths["dlo_path"])["PatientID"].nunique()
    rows.append({"단계": "1. 초기 대상 (DLO_Results 전체)", "수": dlo_n})

    zb = pd.read_excel(paths["z_bounds"]) if os.path.exists(paths["z_bounds"]) else pd.DataFrame()
    attempted_n = zb["PatientID"].nunique() if not zb.empty else 0
    dicom_excl = int(zb["seg_status"].isin(["no_dicom", "no_series"]).sum()) if not zb.empty else None
    rows.append({"단계": "  ├ 시도됨(파이프라인 실행 완료)", "수": attempted_n,
                "비고": "미완료 시 아래 수치는 잠정치 — 재실행 후 갱신 필요" if attempted_n < dlo_n else ""})
    rows.append({"단계": "  └ 제외: DICOM 시리즈 미발견/미매칭 (no_dicom/no_series)", "수": dicom_excl})

    aec_n = pd.read_excel(paths["aec_total"], sheet_name="aec_total")["PatientID"].nunique() \
        if os.path.exists(paths["aec_total"]) else None
    aec_excl = (attempted_n - aec_n) if (aec_n is not None and attempted_n) else None
    rows.append({"단계": "2. 제외: AEC 신호 기준 미달 (CV<0.05 / range<10mA / R²≥0.95)",
                "수": aec_excl, "비고": "AEC 분석 대상에서만 제외 — segmentation과는 무관"})
    rows.append({"단계": "   AEC 분석 가능 (aec_total 포함)", "수": aec_n})

    if not zb.empty:
        vc = zb["seg_status"].value_counts()
        rows.append({"단계": "3. Segmentation 결과 (liver+hip_left+hip_right, TotalSegmentator)", "수": None})
        for status, cnt in vc.items():
            label = {
                "ok": "   ├ ok (liver~pubis 경계 확정 → crop 대상)",
                "pubis_missing": "   ├ 제외: pubis(hip) 미검출",
                "liver_missing": "   ├ 제외: liver 미검출",
                "both_missing": "   ├ 제외: liver+hip 모두 미검출",
                "invalid_bounds": "   ├ 제외: 경계 순서 이상(invalid_bounds)",
                "invalid_volume_match": "   ├ 제외: 볼륨-슬라이스 수 불일치",
                "nifti_fail": "   ├ 제외: DICOM→NIfTI 변환 실패",
                "no_position_data": "   ├ 제외: 위치정보(ImagePositionPatient) 없음",
                "no_series": "   ├ 제외: 시리즈 미매칭",
                "no_dicom": "   ├ 제외: DICOM 폴더 없음",
                "failed": "   ├ 제외: 처리 실패(failed)",
            }.get(str(status), f"   ├ 제외: {status}")
            rows.append({"단계": label, "수": int(cnt)})
        ok_n = int(vc.get("ok", 0))
        rows.append({"단계": "   └ liver~pubis crop / AEC crop 최종 대상", "수": ok_n})
    else:
        ok_n = 0

    bc = pd.read_excel(paths["body_comp"], sheet_name="body_composition") if os.path.exists(paths["body_comp"]) else pd.DataFrame()
    if not bc.empty:
        vc = bc["seg_status"].value_counts()
        rows.append({"단계": "4. 체성분(VAT/SAT, tissue_4_types) 결과", "수": None,
                    "비고": f"3단계 ok {ok_n}명 중 시도"})
        for status, cnt in vc.items():
            label = "   ├ ok (VAT/SAT 산출 완료)" if status == "ok" else f"   ├ 제외: {status}"
            rows.append({"단계": label, "수": int(cnt)})
        final_n = int(vc.get("ok", 0))
        rows.append({"단계": "   └ VAT/SAT 최종 산출 (DLO_Results에 반영)", "수": final_n})
    else:
        rows.append({"단계": "4. 체성분(VAT/SAT) — 아직 미실행", "수": 0})

    if os.path.exists(paths["final_stats"]):
        with open(paths["final_stats"], encoding="utf-8") as f:
            fs = json.load(f)
        rows.append({"단계": "5. 최종 ML 데이터셋 필터링 (final_dataset.xlsx)", "수": fs.get("total"),
                    "비고": "메타데이터(DLO_Results) 전체 기준"})
        rows.append({"단계": "  ├ 제외: Height/Weight 결측", "수": fs.get("no_height_weight")})
        rows.append({"단계": "  ├ 제외: 20세 미만", "수": fs.get("under_20")})
        rows.append({"단계": "  ├ 제외: Height/Weight/BMI IQR 이상치 (합계, 중복 포함 가능)",
                    "수": fs.get("clinical_outlier_total")})
        rows.append({"단계": "  │    Height 이상치", "수": fs.get("height_outlier")})
        rows.append({"단계": "  │    Weight 이상치", "수": fs.get("weight_outlier")})
        rows.append({"단계": "  │    BMI 이상치", "수": fs.get("bmi_outlier")})
        rows.append({"단계": "  ├ 제외: 기타 칼럼 결측치 (예: PatientAge 등)", "수": fs.get("remaining_na")})
        rows.append({"단계": "  ├ 메타데이터 필터링 후", "수": fs.get("after_metadata_filter")})
        rows.append({"단계": "  ├ 제외: AEC Crop 변동성 없음 (CV<0.05/range<10mA/R²≥0.95)",
                    "수": fs.get("aec_no_variability"),
                    "비고": f"aec_cropped 전체 {fs.get('aec_cropped_total')}명 중"})
        rows.append({"단계": "  ├ (PatientID, Manufacturer) 공통 교집합", "수": fs.get("common_pairs")})
        rows.append({"단계": "  └ 최종 ML 데이터셋 (final_dataset.xlsx)", "수": fs.get("final_rows")})
    else:
        rows.append({"단계": "5. 최종 ML 데이터셋(final_dataset.xlsx) — 아직 미생성", "수": 0,
                    "비고": "5_merged_features.py 실행 필요"})

    return rows


def main():
    sheets: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for site in SITES:
        rows = build_site_flow(site)
        df = pd.DataFrame(rows, columns=["단계", "수", "비고"])
        sheets[site] = df

        dlo_n = df.loc[df["단계"].str.contains("초기 대상"), "수"].iloc[0]
        ok_n = df[df["단계"].str.contains("liver~pubis crop / AEC crop 최종 대상")]["수"]
        vat_n = df[df["단계"].str.contains("VAT/SAT 최종 산출")]["수"]
        final_n = df[df["단계"].str.contains("최종 ML 데이터셋 \\(final_dataset")]["수"]
        summary_rows.append({
            "사이트": site,
            "초기 대상": dlo_n,
            "liver~pubis crop 최종": int(ok_n.iloc[0]) if len(ok_n) else 0,
            "VAT/SAT 최종": int(vat_n.iloc[0]) if len(vat_n) else 0,
            "최종 ML 데이터셋": int(final_n.iloc[0]) if len(final_n) and pd.notna(final_n.iloc[0]) else 0,
        })

    sheets["summary"] = pd.DataFrame(summary_rows)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        sheets["summary"].to_excel(writer, sheet_name="summary", index=False)
        for site in SITES:
            sheets[site].to_excel(writer, sheet_name=site, index=False)

    print(f"저장 완료: {OUT_PATH}")
    print(sheets["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
