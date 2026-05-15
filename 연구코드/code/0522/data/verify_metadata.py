"""
강남_metadata.xlsx 데이터 정합성 검증 스크립트

실행 순서: build_dataset.py 전에 먼저 실행해 원본 metadata.xlsx 값이 맞는지 확인한다.

검증 항목:
  A. DICOM 필드 — Age, Sex, kVp, n_slices, z_range_mm, SeriesDescription, ManufacturerModelName
     출처: D:/영상제공/강남/강남_axial DCM 파일
  B. 신장 / 체중 / BMI
     출처: radiation_data_260421_VBcohort.xlsx (예진 측정, CT 기준 ±DATE_THRESHOLD_DAYS 이내)
  C. TAMA
     출처: 강남_DLO_Results.xlsx (PatientID + CT 날짜 매칭)
  D. IMATA  (이미지 OCR)
     출처: DLO_Results SRC_Report 링크 → Osteo_Sarco_1.png L3 level

결과:
  - 콘솔 섹션별 불일치 출력
  - 검증 보고서 Excel 저장 (META_DIR / 강남_verification_report.xlsx)

FIX_AND_SAVE = True 시 수정 값으로 metadata.xlsx를 덮어쓴다 (기본 False).
"""
import os, re, sys
import numpy as np
import pandas as pd
import pydicom
import pytesseract
import cv2
from PIL import Image
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

SITE = "강남"

META_DIR  = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}"
META_PATH = os.path.join(META_DIR, f"{SITE}_metadata.xlsx")
OUT_REPORT= os.path.join(META_DIR, f"{SITE}_verification_report.xlsx")

DICOM_BASE = rf"D:/영상제공/{SITE}/{SITE}_axial"
RAD_PATH   = r"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\radiation_data_260421_VBcohort.xlsx"
DLO_PATH   = rf"D:\영상제공\{SITE}\{SITE}_결과\{SITE}_DLO_Results.xlsx"
DLO_BASE   = rf"D:\영상제공\{SITE}\{SITE}_결과"

SITE_CODE          = "20"     # radiation_data 강남병원 지역병원코드
DATE_THRESHOLD_DAYS = 180     # 신장/체중: CT 기준 절대 일수 이내 측정값만 사용
DICOM_SAMPLE_N     = 0        # 0 = 전체 환자 DICOM 검증 (시간 오래 걸림)
IMATA_SAMPLE_N     = 0        # 0 = 전체 환자 OCR
FIX_AND_SAVE       = False    # True = 검증된 값으로 metadata.xlsx 업데이트

# 허용 오차
TOL_AGE   = 0
TOL_KVP   = 0.5
TOL_NSLIC = 0
TOL_ZRANG = 1.0  # mm
TOL_HT    = 1.0  # cm
TOL_WT    = 1.0  # kg
TOL_BMI   = 0.5
TOL_TAMA  = 5    # mm²
TOL_IMATA = 3    # mm²

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ══════════════════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def _close(a, b, tol):
    """두 값이 tol 이내인지 확인. NaN 포함 처리."""
    try:
        if pd.isna(a) and pd.isna(b):
            return True
        if pd.isna(a) or pd.isna(b):
            return False
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return str(a).strip() == str(b).strip()


def _eq_str(a, b):
    return str(a).strip().lower() == str(b).strip().lower()


def build_folder_map(dicom_base):
    """PatientID → (folder_path, scan_date YYYYMMDD) 매핑."""
    fm, dm = {}, {}
    for name in os.listdir(dicom_base):
        parts = name.split("_")
        if len(parts) >= 3:
            pid       = parts[1]
            date_str  = parts[2]
            folder    = os.path.join(dicom_base, name)
            fm[pid]   = folder
            if len(date_str) == 8 and date_str.isdigit():
                dm[pid] = date_str
    return fm, dm


def extract_dicom_fields(patient_folder):
    """DICOM 폴더에서 메타데이터 + AEC 슬라이스 재추출."""
    subfolders = os.listdir(patient_folder)
    if not subfolders:
        return None
    dcm_dir   = os.path.join(patient_folder, subfolders[0])
    dcm_files = [f for f in os.listdir(dcm_dir) if not f.startswith(".")]
    if not dcm_files:
        return None

    slice_data, meta, header_read = [], {}, False
    for fname in dcm_files:
        try:
            dcm = pydicom.dcmread(os.path.join(dcm_dir, fname), stop_before_pixels=True)
            if not header_read:
                age_raw = str(getattr(dcm, "PatientAge", ""))
                age_num = "".join(filter(str.isdigit, age_raw))
                meta["PatientAge"]            = int(age_num) if age_num else np.nan
                meta["PatientSex"]            = str(getattr(dcm, "PatientSex", ""))
                meta["kVp"]                   = float(getattr(dcm, "KVP", np.nan))
                meta["SeriesDescription"]     = str(getattr(dcm, "SeriesDescription", ""))
                meta["ManufacturerModelName"] = str(getattr(dcm, "ManufacturerModelName", ""))
                header_read = True
            if hasattr(dcm, "ImagePositionPatient"):
                z = float(dcm.ImagePositionPatient[2])
            elif hasattr(dcm, "SliceLocation"):
                z = float(dcm.SliceLocation)
            else:
                continue
            slice_data.append(z)
        except Exception:
            continue

    if not slice_data:
        return None
    meta["n_slices"]   = len(slice_data)
    meta["z_range_mm"] = round(abs(max(slice_data) - min(slice_data)), 2) if len(slice_data) >= 2 else np.nan
    return meta


def parse_dlo_img_path(formula_str, dlo_base):
    """SRC_Report HYPERLINK 수식 → (이미지 절대경로, 날짜 YYYYMMDD)."""
    m = re.search(r'HYPERLINK\("([^"]+)"', str(formula_str))
    if not m:
        return None, None
    rel   = m.group(1).replace("\\", "/").lstrip("./")
    full  = os.path.join(dlo_base, rel.replace("/", os.sep))
    folder_name = os.path.basename(os.path.dirname(full))
    date_str = folder_name[:8] if len(folder_name) >= 8 and folder_name[:8].isdigit() else None
    return full, date_str


def extract_l3_from_sarco(img_path):
    """Osteo_Sarco_1.png 하단 테이블 OCR → L3 TAMA/LAMA/NAMA/IMATA/Myosteatosis 반환."""
    if not img_path or not os.path.exists(img_path):
        return {}
    try:
        img = Image.open(img_path)
        w, h = img.size
        # 테이블 영역: 이미지 하단 72~93%
        crop = img.crop((0, int(h * 0.72), w, int(h * 0.93)))
        arr  = np.array(crop)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        return _parse_muscle_table(text)
    except Exception as e:
        return {"_ocr_error": str(e)}


def _parse_muscle_table(text):
    """OCR 텍스트 → L3 level 값 dict.
    테이블 열 순서: T12 L1 L2 L3 L4 (L3 = index 3, 0-based)."""
    results = {}
    for line in text.split("\n"):
        line = line.strip()
        # 근육 면적 행: 'TAMA 112 127 139 167 167 ...'
        m = re.match(r"(TAMA|LAMA|NAMA|IMATA)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if m:
            vals = [int(m.group(i)) for i in range(2, 6)]  # T12,L1,L2,L3
            results[m.group(1)] = vals[3]                   # L3
        # Myosteatosis 행: '36.6 32.1 36.4 28.5 ...'
        m2 = re.match(
            r"Myosteatosis.*?(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)",
            line
        )
        if m2:
            results["Myosteatosis_L3"] = float(m2.group(4))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

print("=== 강남 metadata 검증 시작 ===\n")
print(f"metadata 로드: {META_PATH}")
meta_df = pd.read_excel(META_PATH, dtype={"PatientID": str})
all_pids = meta_df["PatientID"].tolist()
print(f"  환자 수: {len(meta_df)}")

print(f"\nDICOM 폴더 매핑 중: {DICOM_BASE}")
folder_map, scan_date_map = build_folder_map(DICOM_BASE)
print(f"  DICOM 폴더 수: {len(folder_map)}")
no_dcm = set(all_pids) - set(folder_map.keys())
if no_dcm:
    print(f"  경고: DICOM 없는 환자 {len(no_dcm)}명")

# radiation_data 로드 (예진 전체 시트)
print(f"\nradiation_data 로드: {RAD_PATH}")
rad_df = pd.read_excel(RAD_PATH, sheet_name="신장체중(예진)_전체", dtype=str)
rad_df = rad_df[rad_df["지역병원코드"] == SITE_CODE].copy()
rad_df["진료일로부터(일)_절대값"] = pd.to_numeric(
    rad_df["진료일로부터(일)_절대값"], errors="coerce"
)
rad_df["신장"] = pd.to_numeric(rad_df["신장"], errors="coerce")
rad_df["체중"] = pd.to_numeric(rad_df["체중"], errors="coerce")
rad_df["연구등록번호"] = rad_df["연구등록번호"].astype(str)
print(f"  강남병원 예진 레코드 수: {len(rad_df)}")

# DLO_Results 로드 (header row 3)
print(f"\nDLO_Results 로드: {DLO_PATH}")
dlo_raw = pd.read_excel(DLO_PATH, header=None, skiprows=4, dtype=str)
dlo_raw.columns = [
    "_idx", "NO", "PatientID", "PatientAge", "PatientSex",
    "VBA_Status", "QCT_Status", "SRC",
    "kVp", "mAs", "VBA_Report", "QCT_Report_1", "QCT_Report_2",
    "SRC_Report", "TAMA", "BMI",
    "_a", "_b", "_c",
]
dlo_df = dlo_raw[dlo_raw["PatientID"].notna()].copy()
dlo_df["PatientID"] = dlo_df["PatientID"].astype(str)
dlo_df["TAMA"]      = pd.to_numeric(dlo_df["TAMA"], errors="coerce")

# DLO SRC_Report에서 이미지 경로 + 날짜 추출
dlo_df["img_path"], dlo_df["img_date"] = zip(
    *dlo_df["SRC_Report"].map(lambda f: parse_dlo_img_path(f, DLO_BASE))
)
print(f"  DLO 레코드 수: {len(dlo_df)}, 유니크 환자: {dlo_df['PatientID'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
# A. DICOM 필드 검증
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("A. DICOM 필드 검증")
print("="*60)

dicom_check_pids = all_pids
if DICOM_SAMPLE_N > 0:
    import random; random.seed(42)
    dicom_check_pids = random.sample(
        [p for p in all_pids if p in folder_map],
        min(DICOM_SAMPLE_N, len([p for p in all_pids if p in folder_map]))
    )
    print(f"  샘플 {len(dicom_check_pids)}명 검증 (DICOM_SAMPLE_N={DICOM_SAMPLE_N})")
else:
    print(f"  전체 {len(dicom_check_pids)}명 검증")

DICOM_FIELDS = [
    ("PatientAge",            "PatientAge",            TOL_AGE,   "int"),
    ("PatientSex",            "PatientSex",            None,      "str"),
    ("kVp",                   "kVp",                   TOL_KVP,   "float"),
    ("n_slices",              "n_slices",               TOL_NSLIC, "int"),
    ("z_range_mm",            "z_range_mm",             TOL_ZRANG, "float"),
    ("SeriesDescription",     "SeriesDescription",      None,      "str"),
    ("ManufacturerModelName", "ManufacturerModelName",  None,      "str"),
]

dicom_mismatch_rows = []
dicom_ok = 0

for pid in tqdm(dicom_check_pids, desc="DICOM 검증"):
    if pid not in folder_map:
        continue
    dcm_fields = extract_dicom_fields(folder_map[pid])
    if dcm_fields is None:
        continue

    m_row = meta_df[meta_df["PatientID"] == pid].iloc[0]
    for meta_col, dcm_key, tol, dtype in DICOM_FIELDS:
        if meta_col not in meta_df.columns:
            continue
        v_meta = m_row.get(meta_col, np.nan)
        v_dcm  = dcm_fields.get(dcm_key, np.nan)
        ok = _close(v_meta, v_dcm, tol) if tol is not None else _eq_str(v_meta, v_dcm)
        if not ok:
            dicom_mismatch_rows.append({
                "PatientID": pid, "field": meta_col,
                "metadata": v_meta, "dicom": v_dcm,
            })
    dicom_ok += 1

dicom_mismatch_df = pd.DataFrame(dicom_mismatch_rows)
n_mismatch_patients = dicom_mismatch_df["PatientID"].nunique() if not dicom_mismatch_df.empty else 0
print(f"\n  검증 완료: {dicom_ok}명")
print(f"  불일치: {n_mismatch_patients}명 ({len(dicom_mismatch_df)}개 필드)")
if not dicom_mismatch_df.empty:
    print(dicom_mismatch_df.head(20).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# B. 신장 / 체중 / BMI 검증
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print(f"B. 신장/체중/BMI 검증  (CT 기준 ±{DATE_THRESHOLD_DAYS}일 이내)")
print("="*60)

anthr_rows = []

for pid in tqdm(all_pids, desc="신장/체중 검증"):
    m_row = meta_df[meta_df["PatientID"] == pid].iloc[0]

    # radiation_data에서 해당 환자 레코드 추출 + 날짜 필터
    r_sub = rad_df[rad_df["연구등록번호"] == pid].copy()
    r_sub = r_sub[r_sub["진료일로부터(일)_절대값"] <= DATE_THRESHOLD_DAYS]

    if r_sub.empty:
        anthr_rows.append({
            "PatientID": pid, "status": "NO_RAD_DATA",
            "meta_ht": m_row.get("신장", np.nan),
            "meta_wt": m_row.get("체중", np.nan),
            "meta_bmi": m_row.get("BMI", np.nan),
            "rad_ht": np.nan, "rad_wt": np.nan,
            "rad_bmi_calc": np.nan,
            "ht_ok": np.nan, "wt_ok": np.nan, "bmi_ok": np.nan,
        })
        continue

    # 절대값이 최소인 행 선택
    best = r_sub.loc[r_sub["진료일로부터(일)_절대값"].idxmin()]
    ht   = best["신장"]
    wt   = best["체중"]
    bmi_calc = round(wt / (ht / 100) ** 2, 2) if (ht and ht > 0) else np.nan

    ht_ok  = _close(m_row.get("신장", np.nan), ht, TOL_HT)
    wt_ok  = _close(m_row.get("체중", np.nan), wt, TOL_WT)
    bmi_ok = _close(m_row.get("BMI",  np.nan), bmi_calc, TOL_BMI)

    anthr_rows.append({
        "PatientID": pid, "status": "OK" if (ht_ok and wt_ok and bmi_ok) else "MISMATCH",
        "meta_ht": m_row.get("신장", np.nan), "meta_wt": m_row.get("체중", np.nan),
        "meta_bmi": m_row.get("BMI", np.nan),
        "rad_ht": ht, "rad_wt": wt, "rad_bmi_calc": bmi_calc,
        "ht_ok": ht_ok, "wt_ok": wt_ok, "bmi_ok": bmi_ok,
        "days_diff": int(best["진료일로부터(일)_절대값"]),
    })

anthr_df = pd.DataFrame(anthr_rows)
n_no_data = (anthr_df["status"] == "NO_RAD_DATA").sum()
n_mismatch = (anthr_df["status"] == "MISMATCH").sum()
print(f"\n  검증 완료: {len(anthr_df)}명")
print(f"  radiation_data 없음: {n_no_data}명")
print(f"  불일치: {n_mismatch}명")
if n_mismatch > 0:
    mm = anthr_df[anthr_df["status"] == "MISMATCH"]
    print(mm[["PatientID","meta_ht","rad_ht","ht_ok",
              "meta_wt","rad_wt","wt_ok",
              "meta_bmi","rad_bmi_calc","bmi_ok","days_diff"]].head(20).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# C. TAMA 검증
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("C. TAMA 검증  (DLO_Results, PatientID + CT날짜 매칭)")
print("="*60)

tama_rows = []

for pid in tqdm(all_pids, desc="TAMA 검증"):
    m_row   = meta_df[meta_df["PatientID"] == pid].iloc[0]
    ct_date = scan_date_map.get(pid)          # YYYYMMDD
    d_sub   = dlo_df[dlo_df["PatientID"] == pid]

    if d_sub.empty:
        tama_rows.append({
            "PatientID": pid, "status": "NO_DLO",
            "meta_tama": m_row.get("TAMA", np.nan), "dlo_tama": np.nan,
            "ct_date": ct_date, "img_date": None, "ok": np.nan,
        })
        continue

    # CT 날짜로 매칭
    if ct_date:
        d_date = d_sub[d_sub["img_date"] == ct_date]
        best_row = d_date.iloc[0] if not d_date.empty else d_sub.iloc[0]
    else:
        best_row = d_sub.iloc[0]

    dlo_tama = best_row["TAMA"]
    ok = _close(m_row.get("TAMA", np.nan), dlo_tama, TOL_TAMA)
    tama_rows.append({
        "PatientID": pid, "status": "OK" if ok else "MISMATCH",
        "meta_tama": m_row.get("TAMA", np.nan), "dlo_tama": dlo_tama,
        "ct_date": ct_date, "img_date": best_row.get("img_date"),
        "ok": ok,
    })

tama_df = pd.DataFrame(tama_rows)
n_no_dlo  = (tama_df["status"] == "NO_DLO").sum()
n_mm_tama = (tama_df["status"] == "MISMATCH").sum()
print(f"\n  검증 완료: {len(tama_df)}명")
print(f"  DLO 없음: {n_no_dlo}명")
print(f"  불일치: {n_mm_tama}명")
if n_mm_tama > 0:
    mm = tama_df[tama_df["status"] == "MISMATCH"]
    print(mm[["PatientID","meta_tama","dlo_tama","ct_date","img_date","ok"]].head(20).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# D. IMATA OCR 검증
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("D. IMATA OCR 검증  (Osteo_Sarco_1.png L3 level)")
print("="*60)

# TAMA 검증에서 매칭된 img_path 재활용
tama_path_map = {}
for pid in all_pids:
    ct_date = scan_date_map.get(pid)
    d_sub   = dlo_df[dlo_df["PatientID"] == pid]
    if d_sub.empty:
        continue
    if ct_date:
        d_date = d_sub[d_sub["img_date"] == ct_date]
        best_row = d_date.iloc[0] if not d_date.empty else d_sub.iloc[0]
    else:
        best_row = d_sub.iloc[0]
    img = best_row.get("img_path")
    if img and "Osteo_Sarco_1" in str(img):
        tama_path_map[pid] = img

imata_check_pids = list(tama_path_map.keys())
if IMATA_SAMPLE_N > 0:
    import random; random.seed(42)
    imata_check_pids = random.sample(imata_check_pids, min(IMATA_SAMPLE_N, len(imata_check_pids)))

print(f"  OCR 대상: {len(imata_check_pids)}명")

imata_rows = []
for pid in tqdm(imata_check_pids, desc="IMATA OCR"):
    m_row    = meta_df[meta_df["PatientID"] == pid].iloc[0]
    img_path = tama_path_map[pid]
    ocr      = extract_l3_from_sarco(img_path)

    ocr_tama  = ocr.get("TAMA",  np.nan)
    ocr_imata = ocr.get("IMATA", np.nan)
    ocr_err   = ocr.get("_ocr_error", None)

    tama_ok  = _close(m_row.get("TAMA",  np.nan), ocr_tama,  TOL_TAMA)
    imata_ok = _close(m_row.get("IMATA", np.nan), ocr_imata, TOL_IMATA)

    imata_rows.append({
        "PatientID": pid,
        "meta_tama": m_row.get("TAMA",  np.nan), "ocr_tama":  ocr_tama,  "tama_ok":  tama_ok,
        "meta_imata":m_row.get("IMATA", np.nan), "ocr_imata": ocr_imata, "imata_ok": imata_ok,
        "ocr_error": ocr_err, "img_path": img_path,
    })

imata_df = pd.DataFrame(imata_rows) if imata_rows else pd.DataFrame()
if not imata_df.empty:
    n_imata_mm = (~imata_df["imata_ok"]).sum()
    n_tama_mm2 = (~imata_df["tama_ok"]).sum()
    print(f"\n  OCR 완료: {len(imata_df)}명")
    print(f"  TAMA 불일치(OCR 기준): {n_tama_mm2}명")
    print(f"  IMATA 불일치(OCR 기준): {n_imata_mm}명")
    mm = imata_df[~imata_df["imata_ok"]]
    if not mm.empty:
        print(mm[["PatientID","meta_imata","ocr_imata","imata_ok"]].head(20).to_string(index=False))
else:
    print("  OCR 결과 없음")


# ══════════════════════════════════════════════════════════════════════════════
# 검증 보고서 저장
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n검증 보고서 저장: {OUT_REPORT}")
with pd.ExcelWriter(OUT_REPORT, engine="openpyxl") as writer:
    if not dicom_mismatch_df.empty:
        dicom_mismatch_df.to_excel(writer, sheet_name="A_DICOM_mismatch", index=False)
    anthr_df.to_excel(writer, sheet_name="B_anthropometry",  index=False)
    tama_df.to_excel( writer, sheet_name="C_TAMA",           index=False)
    if not imata_df.empty:
        imata_df.drop(columns=["img_path"], errors="ignore").to_excel(
            writer, sheet_name="D_IMATA_OCR", index=False
        )

print("[검증 완료]\n")


# ══════════════════════════════════════════════════════════════════════════════
# (선택) FIX_AND_SAVE: OCR/소스 값으로 metadata.xlsx 업데이트
# ══════════════════════════════════════════════════════════════════════════════

if FIX_AND_SAVE:
    print("FIX_AND_SAVE=True: metadata.xlsx 업데이트 중...")
    meta_fix = meta_df.copy()
    meta_fix = meta_fix.set_index("PatientID")

    # B. 신장/체중/BMI 업데이트
    for _, row in anthr_df[anthr_df["status"] == "MISMATCH"].iterrows():
        pid = row["PatientID"]
        if pid in meta_fix.index:
            if not pd.isna(row["rad_ht"]):
                meta_fix.at[pid, "신장"] = row["rad_ht"]
            if not pd.isna(row["rad_wt"]):
                meta_fix.at[pid, "체중"] = row["rad_wt"]
            if not pd.isna(row["rad_bmi_calc"]):
                meta_fix.at[pid, "BMI"] = row["rad_bmi_calc"]

    # C. TAMA 업데이트
    for _, row in tama_df[tama_df["status"] == "MISMATCH"].iterrows():
        pid = row["PatientID"]
        if pid in meta_fix.index and not pd.isna(row["dlo_tama"]):
            meta_fix.at[pid, "TAMA"] = row["dlo_tama"]

    # D. IMATA 업데이트 (OCR)
    if not imata_df.empty:
        for _, row in imata_df[~imata_df["imata_ok"]].iterrows():
            pid = row["PatientID"]
            if pid in meta_fix.index and not pd.isna(row["ocr_imata"]):
                meta_fix.at[pid, "IMATA"] = row["ocr_imata"]

    meta_fix.reset_index().to_excel(META_PATH, index=False)
    print(f"  업데이트 완료: {META_PATH}")
