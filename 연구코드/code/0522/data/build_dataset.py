"""
DICOM 데이터 추출부터 머신러닝용 데이터셋 생성까지의 전체 파이프라인.

Step 1 — DICOM 추출 (체크포인트 지원):
  DICOM 폴더에서 환자별 메타데이터와 슬라이스별 mA(AEC) 값을 추출한다.
  출력: {SITE}_dicom_meta.xlsx, {SITE}_aec_raw.xlsx
  ※ CLINICAL_SRC({SITE}_metadata.xlsx)는 절대 덮어쓰지 않는다.

Step 2 — 병합 및 품질 필터:
  Step 1 결과에 필터를 적용하고 AEC 곡선을 고정 길이로 보간한다.
  필터: metadata 결측치 제거 / AEC 분산=0 제거 / 희소 기기 제거(<5%) / kVp==100 유지
  출력: {SITE}_merged_features.xlsx (sheets: metadata, aec_raw, aec_64, aec_128, aec_256)

전제:
  - D:/영상제공/{SITE}/{SITE}_axial 에 DICOM 폴더 존재
    (구조: {idx}_{PatientID}_{date}_CT / {series} / *.dcm)
  - CLINICAL_SRC({SITE}_metadata.xlsx) 에 임상 컬럼(TAMA, BMI, SMI 등) 포함
"""
import os
import json
import random
import pydicom
import pandas as pd
import numpy as np
from tqdm import tqdm

SITE         = "강남"
DICOM_BASE   = rf"D:/영상제공/{SITE}/{SITE}_axial"

# 원본 임상 데이터 — 절대 덮어쓰지 않음 (읽기 전용)
CLINICAL_SRC = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_metadata.xlsx"

# Step 1 출력 (원본과 분리된 별도 파일)
META_OUT     = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_dicom_meta.xlsx"
AECRAW_PATH  = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_aec_raw.xlsx"

# Step 2 출력
OUT_PATH     = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_merged_features.xlsx"

CHECKPOINT   = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\.pipeline_checkpoint.json"
BATCH_SIZE   = 100

# True → Step 1 건너뛰고 기존 META_OUT / AECRAW_PATH 파일 사용
# (이미 추출이 완료된 경우 Step 2만 재실행할 때 사용)
SKIP_STEP1   = False

# DICOM에 없는 임상 측정값 — CLINICAL_SRC에서 병합해 보존
CLINICAL_COLS = ["TAMA", "IMATA", "신장", "체중", "BMI", "SMI"]

# ── PatientID → DICOM 폴더 매핑 ({idx}_{PatientID}_{date}_CT) ─────────────────
folder_map = {}
no_map     = {}   # PatientID → No (폴더명 맨 앞 숫자)
for folder_name in os.listdir(DICOM_BASE):
    parts = folder_name.split("_")
    if len(parts) >= 2:
        folder_map[parts[1]] = os.path.join(DICOM_BASE, folder_name)
        no_map[parts[1]]     = int(parts[0])

print(f"총 DICOM 폴더 수: {len(folder_map)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 함수
# ─────────────────────────────────────────────────────────────────────────────

def extract_patient(pid_str):
    """DICOM 폴더에서 스캔 메타데이터와 슬라이스별 mA 값을 추출한다."""
    if pid_str not in folder_map:
        return None, None

    patient_folder = folder_map[pid_str]
    subfolders = os.listdir(patient_folder)
    if not subfolders:
        return None, None

    dcm_dir   = os.path.join(patient_folder, subfolders[0])
    dcm_files = [f for f in os.listdir(dcm_dir) if not f.startswith(".")]
    if not dcm_files:
        return None, None

    slice_data  = []
    meta_dict   = {}
    header_read = False

    for fname in dcm_files:
        try:
            dcm = pydicom.dcmread(os.path.join(dcm_dir, fname), stop_before_pixels=True)

            if not header_read:
                age_raw = str(getattr(dcm, "PatientAge", ""))
                age_num = "".join(filter(str.isdigit, age_raw))
                meta_dict["PatientAge"]            = int(age_num) if age_num else np.nan
                meta_dict["PatientSex"]            = str(getattr(dcm, "PatientSex", ""))
                meta_dict["kVp"]                   = float(getattr(dcm, "KVP", np.nan))
                meta_dict["SeriesDescription"]     = str(getattr(dcm, "SeriesDescription", ""))
                meta_dict["ManufacturerModelName"] = str(getattr(dcm, "ManufacturerModelName", ""))
                header_read = True

            if hasattr(dcm, "ImagePositionPatient"):
                z = float(dcm.ImagePositionPatient[2])
            elif hasattr(dcm, "SliceLocation"):
                z = float(dcm.SliceLocation)
            else:
                continue

            mA = float(getattr(dcm, "XRayTubeCurrent", np.nan))
            slice_data.append((z, mA))
        except Exception:
            continue

    if not slice_data:
        return None, None

    z_first_orig = slice_data[0][0]
    z_last_orig  = slice_data[-1][0]

    slice_data.sort(key=lambda x: x[0])
    z_vals  = [s[0] for s in slice_data]
    mA_vals = [s[1] for s in slice_data]

    meta_dict["n_slices"]   = len(slice_data)
    meta_dict["z_range_mm"] = abs(max(z_vals) - min(z_vals)) if len(z_vals) >= 2 else np.nan
    meta_dict["z_min"]      = min(z_vals)
    meta_dict["z_max"]      = max(z_vals)
    meta_dict["z_flipped"]  = z_first_orig > z_last_orig

    return meta_dict, mA_vals


def save_checkpoint(processed, meta_rows, aec_rows):
    def clean(v):
        return None if isinstance(v, float) and np.isnan(v) else v

    data = {
        "processed": processed,
        "meta_rows": [{k: clean(v) for k, v in row.items()} for row in meta_rows],
        "aec_rows":  [[clean(v) for v in row] for row in aec_rows],
    }
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return 0, [], []
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta_rows = [{k: (np.nan if v is None else v) for k, v in row.items()} for row in data["meta_rows"]]
    aec_rows  = [[(np.nan if v is None else v) for v in row] for row in data["aec_rows"]]
    return data["processed"], meta_rows, aec_rows


def write_intermediate(patient_ids_all, meta_rows, aec_rows, clinical_df):
    n_done = len(meta_rows)

    meta_df = pd.DataFrame(meta_rows)
    meta_df.insert(0, "PatientID", patient_ids_all[:n_done])
    meta_df["PatientID"] = meta_df["PatientID"].astype(str)
    
    # [수정] No 컬럼 매핑 후 맨 앞으로 배치되도록 보장
    meta_df.insert(0, "No", meta_df["PatientID"].map(no_map))

    avail_clinical = [c for c in CLINICAL_COLS if c in clinical_df.columns]
    clin = clinical_df[["PatientID"] + avail_clinical].copy()
    clin["PatientID"] = clin["PatientID"].astype(str)
    meta_df = meta_df.merge(clin, on="PatientID", how="left")

    col_order = ["No", "PatientID", "PatientAge", "PatientSex", "kVp",
                 "n_slices", "z_range_mm", "z_min", "z_max", "z_flipped",
                 "SeriesDescription", "ManufacturerModelName"] + CLINICAL_COLS
    col_order = list(dict.fromkeys(c for c in col_order if c in meta_df.columns))
    remaining = [c for c in meta_df.columns if c not in col_order]
    meta_df   = meta_df[col_order + remaining]
    meta_df[meta_df.select_dtypes(include="float").columns] = \
        meta_df.select_dtypes(include="float").round(2)
    meta_df.to_excel(META_OUT, index=False)

    max_slices = max(len(r) for r in aec_rows) if aec_rows else 0
    slice_cols = [f"slice_{i+1}" for i in range(max_slices)]
    aec_data   = []
    for pid, vals in zip(patient_ids_all[:n_done], aec_rows):
        aec_data.append([pid] + vals + [np.nan] * (max_slices - len(vals)))
    aec_df = pd.DataFrame(aec_data, columns=["PatientID"] + slice_cols)
    
    # [수정] No 컬럼 맨 앞으로 배치되도록 보장
    aec_df.insert(0, "No", aec_df["PatientID"].map(no_map))
    aec_df.insert(2, "n_slices", [row.get("n_slices", np.nan) for row in meta_rows])
    aec_df[aec_df.select_dtypes(include="float").columns] = \
        aec_df.select_dtypes(include="float").round(2)
    aec_df.to_excel(AECRAW_PATH, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 검증 함수
# ─────────────────────────────────────────────────────────────────────────────

def verify_dicom(meta_df, aec_df, n_samples=5, seed=42):
    aec_slice_cols = [c for c in aec_df.columns if c.startswith("slice_")]
    eligible = [pid for pid in meta_df["PatientID"].astype(str).tolist() if pid in folder_map]
    if not eligible:
        print("[검증] DICOM 폴더가 있는 환자가 없습니다.")
        return

    rng     = random.Random(seed)
    samples = rng.sample(eligible, min(n_samples, len(eligible)))

    print(f"\n[검증] {len(samples)}명 샘플 DICOM 원본 대조 시작")
    ok_count = 0

    for pid_str in samples:
        meta_dict, mA_vals = extract_patient(pid_str)
        m_row = meta_df[meta_df["PatientID"].astype(str) == pid_str]
        a_row = aec_df[aec_df["PatientID"].astype(str) == pid_str]

        if meta_dict is None or m_row.empty or a_row.empty:
            print(f"  [FAIL] {pid_str}: 데이터 누락 또는 재추출 실패")
            continue

        issues = []
        kv_excel = float(m_row["kVp"].values[0])
        kv_dcm   = meta_dict["kVp"]
        if not (np.isnan(kv_excel) and np.isnan(kv_dcm)):
            if np.isnan(kv_excel) != np.isnan(kv_dcm) or abs(kv_excel - kv_dcm) > 0.5:
                issues.append(f"kVp: excel={kv_excel} dcm={kv_dcm}")

        ns_excel_raw = m_row["n_slices"].values[0]
        ns_excel = int(ns_excel_raw) if not pd.isna(ns_excel_raw) else None
        ns_dcm   = meta_dict["n_slices"]
        if ns_excel != ns_dcm:
            issues.append(f"n_slices: excel={ns_excel} dcm={ns_dcm}")

        if "z_range_mm" in m_row.columns:
            zr_excel = float(m_row["z_range_mm"].values[0])
            zr_dcm   = meta_dict["z_range_mm"]
            if not (np.isnan(zr_excel) or np.isnan(zr_dcm)) and abs(zr_excel - zr_dcm) > 1.0:
                issues.append(f"z_range_mm: excel={zr_excel:.1f} dcm={zr_dcm:.1f}")

        aec_vals_excel = a_row.iloc[0][aec_slice_cols].dropna().values.astype(float)
        mA_arr_dcm     = np.array(mA_vals, dtype=float)
        if len(aec_vals_excel) != len(mA_arr_dcm):
            issues.append(f"mA 길이: excel={len(aec_vals_excel)} dcm={len(mA_arr_dcm)}")
        elif len(mA_arr_dcm) > 0:
            max_diff = np.nanmax(np.abs(aec_vals_excel - mA_arr_dcm))
            if max_diff > 0.5:
                issues.append(f"mA 최대 차이: {max_diff:.2f} mA")

        if issues:
            print(f"  [WARN] {pid_str}: {' | '.join(issues)}")
        else:
            mA_med = np.nanmedian(mA_vals) if mA_vals else float("nan")
            print(f"  [ OK ] {pid_str}: kVp={meta_dict['kVp']}, n_slices={meta_dict['n_slices']}, z_range={meta_dict.get('z_range_mm', float('nan')):.1f}mm, mA_median={mA_med:.1f}")
            ok_count += 1

    print(f"[검증 완료] {ok_count}/{len(samples)} 정상\n")


def check_z_direction_consistency(meta_df, z_iqr_factor=3.0):
    if "z_min" not in meta_df.columns or "z_max" not in meta_df.columns:
        print("[방향 검증] z_min/z_max 컬럼 없음 — 검증 불가")
        return

    valid = meta_df.dropna(subset=["z_min", "z_max"])
    n_total = len(valid)
    if n_total == 0:
        print("[방향 검증] 유효한 z 데이터 없음")
        return

    print(f"\n[방향 일관성 검증]  N={n_total}")
    if "z_flipped" in meta_df.columns:
        n_flipped = valid["z_flipped"].astype(float).sum()
        print(f"  원본 DICOM 내림차순(cranial→caudal) 저장 비율: {int(n_flipped)}/{"N_total"} ({100*n_flipped/n_total:.1f}%)")

    for col in ("z_min", "z_max"):
        q1, q3 = valid[col].quantile([0.25, 0.75])
        iqr    = q3 - q1
        lo     = q1 - z_iqr_factor * iqr
        hi     = q3 + z_iqr_factor * iqr
        outliers = valid[~valid[col].between(lo, hi)][["PatientID", col]]
        print(f"  {col}: median={valid[col].median():.1f}mm, IQR=[{q1:.1f}, {q3:.1f}], 허용범위=[{lo:.1f}, {hi:.1f}]")
        if not outliers.empty:
            print(f"         이상치 {len(outliers)}명: " + ", ".join(f"{r['PatientID']}({r[col]:.1f})" for _, r in outliers.head(10).iterrows()))


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: DICOM 추출
# ══════════════════════════════════════════════════════════════════════════════

clinical_df = pd.read_excel(CLINICAL_SRC)
all_pids    = clinical_df["PatientID"].astype(str).tolist()
total       = len(all_pids)

missing_dicom = set(all_pids) - set(folder_map.keys())
if missing_dicom:
    print(f"경고: DICOM 폴더 없는 PatientID {len(missing_dicom)}개: {list(missing_dicom)[:5]}")

if SKIP_STEP1:
    if not os.path.exists(META_OUT) or not os.path.exists(AECRAW_PATH):
        raise FileNotFoundError(f"SKIP_STEP1=True이지만 Step 1 출력 파일이 없습니다.")
    print(f"[Step 1 SKIP] 기존 파일 사용\n  {META_OUT}\n  {AECRAW_PATH}")
else:
    start_idx, meta_rows, aec_rows = load_checkpoint()
    if start_idx > 0:
        print(f"체크포인트 감지: {start_idx}/{total}번째부터 재시작")

    for i in tqdm(range(start_idx, total), desc="DICOM 추출", initial=start_idx, total=total):
        pid = all_pids[i]
        meta_dict, mA_vals = extract_patient(pid)

        if meta_dict is None:
            meta_dict = {
                "PatientAge": np.nan, "PatientSex": np.nan, "kVp": np.nan,
                "n_slices": np.nan, "z_range_mm": np.nan,
                "z_min": np.nan, "z_max": np.nan, "z_flipped": np.nan,
                "SeriesDescription": np.nan, "ManufacturerModelName": np.nan,
            }
        if mA_vals is None:
            mA_vals = []

        meta_rows.append(meta_dict)
        aec_rows.append(mA_vals)

        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
            save_checkpoint(i + 1, meta_rows, aec_rows)
            write_intermediate(all_pids, meta_rows, aec_rows, clinical_df)
            tqdm.write(f"  [{i + 1}/{total}] 저장 완료")

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

meta_df = pd.read_excel(META_OUT)
aec_df  = pd.read_excel(AECRAW_PATH)

print(f"\n[Step 1 완료]\n  metadata : {meta_df.shape}\n  aec_raw  : {aec_df.shape}")
verify_dicom(meta_df, aec_df, n_samples=5)
check_z_direction_consistency(meta_df)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 병합 및 필터 → merged_features.xlsx
# ══════════════════════════════════════════════════════════════════════════════

MIN_MFR_RATIO = 0.05
slice_cols = [c for c in aec_df.columns if c.startswith("slice_")]

# 1) metadata 결측치 제거
merged     = pd.merge(meta_df, aec_df[["PatientID"]], on="PatientID", how="inner")
meta_clean = merged.dropna()
n1 = len(meta_clean)

# 2) AEC 분산 없는 환자 제거
aec_std    = aec_df.set_index("PatientID")[slice_cols].std(axis=1)
no_var_ids = set(aec_std[aec_std == 0].index)
meta_clean = meta_clean[~meta_clean["PatientID"].isin(no_var_ids)].reset_index(drop=True)
n2 = len(meta_clean)

# 3) 소수 기기 모델 제거
model_ratio = meta_clean["ManufacturerModelName"].value_counts(normalize=True)
rare_models = set(model_ratio[model_ratio < MIN_MFR_RATIO].index)
meta_clean  = meta_clean[~meta_clean["ManufacturerModelName"].isin(rare_models)].reset_index(drop=True)
n3 = len(meta_clean)

# 4) kVp == 100만 유지
meta_clean = meta_clean[meta_clean["kVp"] == 100].reset_index(drop=True)
n4 = len(meta_clean)

# 5) 18세 미만 제거
meta_clean = meta_clean[meta_clean["PatientAge"] >= 18].reset_index(drop=True)
n5 = len(meta_clean)

# 6) 임상 컬럼 IQR 이상치 제거
iqr_cols = ["PatientAge", "BMI", "SMI"] + (["TAMA"] if "TAMA" in meta_clean.columns else [])
for col in iqr_cols:
    q1, q3 = meta_clean[col].quantile([0.25, 0.75])
    iqr    = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    meta_clean = meta_clean[meta_clean[col].between(lo, hi)].reset_index(drop=True)
n6 = len(meta_clean)

# 7) AEC 행 평균 IQR 이상치 제거
aec_for_iqr = aec_df[aec_df["PatientID"].isin(set(meta_clean["PatientID"]))].set_index("PatientID")[slice_cols]
row_mean    = aec_for_iqr.mean(axis=1)
q1, q3     = row_mean.quantile([0.25, 0.75])
iqr        = q3 - q1
lo, hi     = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outlier_aec = set(row_mean[~row_mean.between(lo, hi)].index)
meta_clean  = meta_clean[~meta_clean["PatientID"].isin(outlier_aec)].reset_index(drop=True)
n7 = len(meta_clean)

valid_ids = set(meta_clean["PatientID"])

# ── Sheet 1: metadata 컬럼 정렬 보장 ─────────────────────────────────────────
sheet_meta = meta_clean.reset_index(drop=True)
if "No" in sheet_meta.columns:
    sheet_meta = sheet_meta.drop(columns=["No"])
sheet_meta.insert(0, "No", sheet_meta["PatientID"].map(no_map))

# No와 PatientID를 무조건 제일 앞으로 고정
_meta_cols = ["No", "PatientID"] + [c for c in sheet_meta.columns if c not in ["No", "PatientID"]]
sheet_meta = sheet_meta[_meta_cols]


# ── Sheet 2: aec_raw 컬럼 정렬 보장 ──────────────────────────────────────────
sheet_aecraw = (
    meta_clean[["PatientID"]]
    .merge(aec_df[aec_df["PatientID"].isin(valid_ids)], on="PatientID", how="left")
    .reset_index(drop=True)
)
if "No" in sheet_aecraw.columns:
    sheet_aecraw = sheet_aecraw.drop(columns=["No"])
sheet_aecraw.insert(0, "No", sheet_aecraw["PatientID"].map(no_map))

_aec_slice_cols = [c for c in sheet_aecraw.columns if c.startswith("slice_")]
if "n_slices" not in sheet_aecraw.columns:
    sheet_aecraw["n_slices"] = sheet_aecraw[_aec_slice_cols].notna().sum(axis=1)

# 순서 고정: No, PatientID, n_slices, 그 외 순서
_raw_cols = ["No", "PatientID", "n_slices"] + [c for c in sheet_aecraw.columns if c not in ["No", "PatientID", "n_slices"]]
sheet_aecraw = sheet_aecraw[_raw_cols]


# ── Sheet 3~5: 보간 AEC (64 / 128 / 256 포인트) 컬럼 정렬 보장 ─────────────────
def make_interp_sheet(aecraw_df, pid_list, s_cols, n_points, z_bounds=None, meta_idx=None):
    x_new = np.linspace(0, 1, n_points)
    cols  = ["No", "PatientID"] + [f"z_{i+1}" for i in range(n_points)]
    rows  = []
    for pid in pid_list:
        vals = aecraw_df[aecraw_df["PatientID"] == pid].iloc[0][s_cols].dropna().values.astype(float)

        if (z_bounds is not None and meta_idx is not None
                and pid in z_bounds.index and pid in meta_idx.index):
            z_lo  = z_bounds.at[pid, "z_pubis_bottom"]
            z_hi  = z_bounds.at[pid, "z_t12_upper"]
            z_min = meta_idx.at[pid, "z_min"]
            z_rng = meta_idx.at[pid, "z_range_mm"]
            n = len(vals)
            if pd.notna(z_lo) and pd.notna(z_hi) and pd.notna(z_min) and pd.notna(z_rng) and n > 1:
                z_pos  = np.linspace(z_min, z_min + z_rng, n)
                lo_idx = max(0, int(np.searchsorted(z_pos, z_lo)))
                hi_idx = min(n, int(np.searchsorted(z_pos, z_hi, side="right")))
                if hi_idx - lo_idx > 1:
                    vals = vals[lo_idx:hi_idx]

        x_orig = np.linspace(0, 1, len(vals))
        rows.append([no_map.get(str(pid)), pid] + np.interp(x_new, x_orig, vals).tolist())
    return pd.DataFrame(rows, columns=cols)


Z_BOUNDS_PATH = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}\{SITE}_z_bounds.xlsx"
if os.path.exists(Z_BOUNDS_PATH):
    _zb = pd.read_excel(Z_BOUNDS_PATH).query("seg_status == 'ok'")
    _zb["PatientID"] = _zb["PatientID"].astype(int)
    z_bounds_df = _zb.set_index("PatientID")[["z_t12_upper", "z_pubis_bottom"]]
    print(f"[z_bounds] {len(z_bounds_df)}명 로드")
else:
    z_bounds_df = None
    print("[z_bounds] 파일 없음 → 전체 구간 보간")

_meta_idx = meta_clean.set_index("PatientID")[["z_min", "z_range_mm"]]

pid_list     = meta_clean["PatientID"].tolist()
sheet_aec64  = make_interp_sheet(aec_df, pid_list, slice_cols, 64,  z_bounds_df, _meta_idx)
sheet_aec128 = make_interp_sheet(aec_df, pid_list, slice_cols, 128, z_bounds_df, _meta_idx)
sheet_aec256 = make_interp_sheet(aec_df, pid_list, slice_cols, 256, z_bounds_df, _meta_idx)


# ── 최종 저장 ─────────────────────────────────────────────────────────────────
def round_floats(df, decimals=2):
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(decimals)
    return df

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    round_floats(sheet_meta).to_excel(writer,    sheet_name="metadata", index=False)
    round_floats(sheet_aecraw).to_excel(writer,  sheet_name="aec_raw",  index=False)
    round_floats(sheet_aec64).to_excel(writer,   sheet_name="aec_64",   index=False)
    round_floats(sheet_aec128).to_excel(writer,  sheet_name="aec_128",  index=False)
    round_floats(sheet_aec256).to_excel(writer,  sheet_name="aec_256",  index=False)

print(f"\n[Step 2 완료]  Saved: {OUT_PATH}")