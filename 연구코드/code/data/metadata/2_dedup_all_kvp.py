"""
강남_DLO_Results_Unique.xlsx - 'raw' 시트(전체 kVp)에서
(PatientID, Manufacturer) 기준 중복을 Portal 우선순위 규칙으로 자동 제거하여
'all_kVp_dedup' 시트로 저장한다.

기존 kVp_100 필터는 조영 단계(Pre/Arterial/Portal/Delay 등) 중 우연히 kVp=100인
스캔만 남기는 방식이었다. 이 스크립트는 kVp와 무관하게 조영 단계 우선순위로
환자당 대표 스캔 1개를 선택하여, kVp=100이 아닌 스캔만 있던 환자도 포함한다.

우선순위 규칙은 2_unique.py의 GUI 자동선택 로직과 동일하다.
"""

import pandas as pd
import openpyxl
from openpyxl.styles import Font
from pathlib import Path

SITE = "신촌"
EXCEL_PATH = Path(rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\metadata\{SITE}_DLO_Results_Unique.xlsx")
SRC_SHEET  = "raw"
OUT_SHEET  = "all_kVp_dedup"

PRIORITY_KEYWORDS = ["portal", "por", "with", "post", "delay", "contrast", "arterial", "artery"]


def pick_representative(grp: pd.DataFrame) -> int:
    for kw in PRIORITY_KEYWORDS:
        m = grp[grp["Series_Desc"].astype(str).str.lower().str.contains(kw, na=False)]
        if len(m):
            return int(m.index[0])
    return int(grp.index[0])


def load_hyperlinks(sheet_name: str) -> dict[int, dict]:
    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws = wb[sheet_name]
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    src_col = headers.index("SRC_Report")
    links = {}
    for row_idx, row in enumerate(ws.iter_rows(min_row=2)):
        cell = row[src_col]
        entry: dict = {}
        if cell.hyperlink:
            entry["hyperlink"] = cell.hyperlink
        if cell.value is not None and "HYPERLINK" in str(cell.value).upper():
            entry["formula"] = str(cell.value)
        if entry:
            links[row_idx] = entry
    wb.close()
    return links


def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SRC_SHEET).reset_index(drop=True)
    print(f"[{SRC_SHEET}] 전체 {len(df)}행, PatientID {df['PatientID'].nunique()}명")

    dup_mask = df.duplicated(["PatientID", "Manufacturer"], keep=False)
    keep_idx = set(df[~dup_mask].index)
    n_groups = 0
    for _, grp in df[dup_mask].groupby(["PatientID", "Manufacturer"]):
        keep_idx.add(pick_representative(grp))
        n_groups += 1

    orig_indices = sorted(keep_idx)
    result_df = df.loc[orig_indices].reset_index(drop=True)

    print(f"중복 그룹 {n_groups}개 → 대표 1행씩 선택")
    print(f"결과: {len(result_df)}행 (kVp 분포)\n{result_df['kVp'].value_counts()}")

    hyperlinks = load_hyperlinks(SRC_SHEET)

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        result_df.to_excel(writer, sheet_name=OUT_SHEET, index=False)

    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws = wb[OUT_SHEET]
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    src_col = headers.index("SRC_Report") + 1
    for new_row, orig_idx in enumerate(orig_indices, start=2):
        entry = hyperlinks.get(orig_idx)
        if entry:
            cell = ws.cell(row=new_row, column=src_col)
            if "hyperlink" in entry:
                cell.hyperlink = entry["hyperlink"]
            if "formula" in entry:
                cell.value = entry["formula"]
            cell.font = Font(color="0563C1", underline="single")
    wb.save(EXCEL_PATH)

    print(f"\n저장 완료: '{OUT_SHEET}' 시트 → {EXCEL_PATH}")


if __name__ == "__main__":
    main()
