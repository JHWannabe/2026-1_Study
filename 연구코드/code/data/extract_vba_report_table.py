"""
Pooled dataset(cleaned_dataset 시트)의 327명 각각에 대해 VBA_Report(PNG) 이미지를 찾아
17개 레벨(T1~T10, T11, T12, L1~L5) x 4개 지표(Anterior/Middle/Posterior Height, HLR)
= 68개 숫자를 OCR로 추출하고, Pooled dataset 파일의 새로운 시트('vba_report_values')에 저장한다.

매칭 경로:
    cleaned_dataset.Research_study_id
    -> (institution별) 강남_DLO_Results.xlsx / 신촌_DLO_Results.xlsx 의 PatientID
    -> VBA_Report 컬럼의 HYPERLINK 수식에서 PNG 상대경로 추출
    -> 실제 PNG 파일을 열어 표 영역을 OCR

주의:
- VBA_Report 표는 모든 리포트에서 동일한 픽셀 위치에 고정 렌더링되는 템플릿이므로,
  표 영역을 크롭한 뒤 좌표 기반으로 각 셀을 행(레벨)/열(지표)에 배정한다.
- DLO_Results.xlsx에 동일 PatientID가 여러 행 있는 경우 첫 번째 행(첫 촬영)을 사용한다.

사용법:
    python extract_vba_report_table.py
"""

import os
import re

import openpyxl
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

POOLED_PATH = "../../data/Pooled dataset 20260717.xlsx"
RESULT_SHEET_NAMES = {
    "Anterior": "vba_anterior_height",
    "Middle": "vba_middle_height",
    "Posterior": "vba_posterior_height",
    "HLR": "vba_height_loss_ratio",
}

INSTITUTIONS = {
    "Kangnam": {
        "dlo_xlsx": r"E:\영상제공\강남\강남_결과\강남_DLO_Results.xlsx",
        "base_dir": r"E:\영상제공\강남\강남_결과",
    },
    "Sinchon": {
        "dlo_xlsx": r"E:\영상제공\신촌\신촌_결과\신촌_DLO_Results.xlsx",
        "base_dir": r"E:\영상제공\신촌\신촌_결과",
    },
}

LEVELS = [
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10",
    "T11", "T12", "L1", "L2", "L3", "L4", "L5",
]
METRICS = ["Anterior", "Middle", "Posterior", "HLR"]

# Osteo_VBA_1.png (1827x2579 고정 템플릿) 기준 표 영역 좌표
TABLE_CROP = (1000, 1100, 1827, 2350)  # left, top, right, bottom (L5 텍스트 잘림 방지 여유 포함)
ROW0_TOP = 51      # 크롭 기준 T1 행 텍스트의 top 좌표
ROW_STEP = 71.0    # 행간 간격(px)
COL_RANGES = {
    "Anterior": (180, 300),
    "Middle": (330, 450),
    "Posterior": (480, 600),
    "HLR": (630, 760),
}

HYPERLINK_RE = re.compile(r'HYPERLINK\("([^"]+)"')


def load_vba_report_paths(dlo_xlsx_path: str, base_dir: str) -> dict:
    """DLO_Results.xlsx에서 PatientID -> VBA_Report PNG 절대경로 매핑을 만든다."""
    wb = openpyxl.load_workbook(dlo_xlsx_path)
    ws = wb["Sheet1"]
    header = [c.value for c in ws[4]]
    idx_pid = header.index("PatientID")
    idx_vba = header.index("VBA_Report")

    mapping = {}
    for row in ws.iter_rows(min_row=5, max_row=ws.max_row):
        pid_val = row[idx_pid].value
        if pid_val is None:
            continue
        try:
            pid = int(pid_val)
        except (TypeError, ValueError):
            continue
        if pid in mapping:
            continue  # 첫 번째 행만 사용

        vba_val = row[idx_vba].value
        if not isinstance(vba_val, str):
            continue
        m = HYPERLINK_RE.search(vba_val)
        if not m:
            continue

        rel_path = m.group(1)
        if rel_path.startswith("./"):
            rel_path = rel_path[2:]
        mapping[pid] = os.path.normpath(os.path.join(base_dir, rel_path))

    wb.close()
    return mapping


def extract_table_from_image(image_path: str) -> dict:
    """VBA report PNG에서 17 x 4 표를 OCR로 읽어 {level: {metric: str}} 형태로 반환."""
    im = Image.open(image_path)
    table_img = im.crop(TABLE_CROP)
    # psm 6(기본 자동 분할)은 특정 이미지에서 "Empty page"로 통째로 실패하거나
    # 빨간/주황 강조 행에서 소수점을 놓치는 경우가 있어, sparse-text 모드(11)를 사용한다.
    data = pytesseract.image_to_data(table_img, output_type=pytesseract.Output.DICT, config="--psm 11")

    grid = {level: {metric: None for metric in METRICS} for level in LEVELS}

    for text, top, left in zip(data["text"], data["top"], data["left"]):
        text = text.strip()
        if not text:
            continue

        row_idx = round((top - ROW0_TOP) / ROW_STEP)
        if row_idx < 0 or row_idx >= len(LEVELS):
            continue

        metric = None
        for m_name, (x0, x1) in COL_RANGES.items():
            if x0 <= left <= x1:
                metric = m_name
                break
        if metric is None:
            continue

        level = LEVELS[row_idx]
        existing = grid[level][metric]
        if existing is None or len(text) > len(existing):
            grid[level][metric] = text

    return grid


def load_patients(pooled_path: str) -> list:
    """cleaned_dataset 시트에서 (Research_study_id, Real_ID, institution) 목록을 만든다."""
    wb = openpyxl.load_workbook(pooled_path, read_only=True, data_only=True)
    ws = wb["cleaned_dataset"]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    idx_study_id = header.index("Research_study_id")
    idx_real_id = header.index("Real_ID")
    idx_institution = header.index("institution")

    patients = []
    for row in rows:
        study_id = row[idx_study_id]
        if study_id is None:
            continue
        real_id = row[idx_real_id]
        patients.append((int(study_id), real_id, row[idx_institution]))

    wb.close()
    return patients


def write_result_sheets(pooled_path: str, results: list):
    """지표(Anterior/Middle/Posterior/HLR)별로 별도 시트에 결과를 저장한다."""
    wb = openpyxl.load_workbook(pooled_path)

    LEGACY_SHEET_NAME = "vba_report_values"
    if LEGACY_SHEET_NAME in wb.sheetnames:
        del wb[LEGACY_SHEET_NAME]

    for metric, sheet_name in RESULT_SHEET_NAMES.items():
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
        ws = wb.create_sheet(sheet_name)
        ws.append(["Research_study_id", "institution"] + LEVELS)

        for study_id, institution, grid in results:
            row_vals = [study_id, institution]
            if grid is None:
                row_vals += [None] * len(LEVELS)
            else:
                row_vals += [grid[lvl][metric] for lvl in LEVELS]
            ws.append(row_vals)

    wb.save(pooled_path)
    wb.close()


def main():
    vba_paths_by_institution = {
        inst: load_vba_report_paths(cfg["dlo_xlsx"], cfg["base_dir"])
        for inst, cfg in INSTITUTIONS.items()
    }

    patients = load_patients(POOLED_PATH)

    results = []
    missing_mapping = []
    missing_file = []

    for study_id, institution in patients:
        path_map = vba_paths_by_institution.get(institution, {})
        image_path = path_map.get(study_id)

        if image_path is None:
            missing_mapping.append((study_id, institution))
            results.append((study_id, institution, None))
            continue

        if not os.path.exists(image_path):
            missing_file.append((study_id, institution, image_path))
            results.append((study_id, institution, None))
            continue

        grid = extract_table_from_image(image_path)
        results.append((study_id, institution, grid))

    write_result_sheets(POOLED_PATH, results)

    print(f"총 {len(results)}명 처리")
    print(f"PatientID 매핑 실패: {len(missing_mapping)}건")
    print(f"이미지 파일 없음: {len(missing_file)}건")
    if missing_mapping:
        print("  매핑 실패 예시:", missing_mapping[:5])
    if missing_file:
        print("  파일 없음 예시:", missing_file[:5])
    sheet_list = ", ".join(RESULT_SHEET_NAMES.values())
    print(f"결과가 다음 시트에 저장되었습니다 ({sheet_list}): {POOLED_PATH}")


if __name__ == "__main__":
    main()
