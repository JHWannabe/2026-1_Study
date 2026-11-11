r"""예후데이터 5y_filtered 6개 질환의 고유 환자(키/몸무게/BMI 완비)에 대해 gangnam_landmark_filtered.xlsx와
같은 시트(patient_metadata/landmarks/aec_total/VAT_total/SAT_total/TAMA_total)를 만든다.

DICOM은 D:\데이터서비스팀 요청(이홍선)\{seq}_{환자ID}_{YYYYMM01}_CT\*.dcm 에 시리즈가 섞여 있어, 환자별로
best series(7_select_best_series.py 규칙) 파일만 임시 폴더에 복사해 check_landmarks.process_patient를 그대로 쓴다.
재시작 가능(체크포인트). 끝까지 돌면 data/예후데이터/5y_landmark/prognosis_5y_landmark_filtered.xlsx(시트 여러 개)를 만든다.
인자 --finalize 이면 세그멘테이션 없이 지금까지의 결과로 질환별 파일만 다시 만든다."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")  # 동시 실행 프로세스끼리 CPU 경합을 줄이려 프로세스당 스레드 제한(torch import 전)
os.environ.setdefault("MKL_NUM_THREADS", "3")
import functools
import json
import os
import shutil
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_landmarks as cl  # noqa: E402
import merge_metadata as mm  # noqa: E402

warnings.filterwarnings("ignore", message="Workbook contains no default style")  # 병원 원본 엑셀에 기본 스타일이 없어 openpyxl이 매번 내는 무해한 경고

DATA = Path(r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data\예후데이터")
OUT_DIR = DATA / "5y_landmark"
OUT = str(OUT_DIR / "prognosis_landmark.xlsx")
CKPT = str(OUT_DIR / "prognosis_landmark_checkpoint.pkl")
DICOM_BASE = r"D:\데이터서비스팀 요청(이홍선)"
FINALIZE_EVERY = 5  # 환자 수. 저장(5명 배치)마다 filtered도 갱신
DISEASES = ["dm", "htn", "dl", "osteoporosis", "mi", "stroke"]

# 키/체중/BMI 완비 환자만 대상으로 한다(그렇지 않으면 filtered에서 어차피 빠진다).
cl._compute_bc_rows_orig = cl._compute_bc_rows
cl._compute_bc_rows = lambda pid, row, img, n_k, tmp_dir, report_path=None, vol=None: (
    [] if vol is None and report_path is None else cl._compute_bc_rows_orig(pid, row, img, n_k, tmp_dir, report_path=report_path, vol=vol))  # ponytail: 시트에 안 쓰는 anchor 체성분 재세그멘테이션 생략


NPY_DIR = os.path.join(cl.NPY_ROOT, "예후데이터")
_PROG_NPY: dict = {}   # 가짜 series_dir -> (json 메타, npy 경로). cl._find_npy가 이 표를 먼저 본다.
_find_npy_orig = cl._find_npy
cl._find_npy = lambda series_dir: _PROG_NPY.get(series_dir) or _find_npy_orig(series_dir)  # series_folder에 '/'가 있는 시리즈도 경로 분해 없이 찾는다


def prog_npy_dir(pid: int):
    """{NPY_DIR}\\{pid} 의 npy/json이 cl._load_volume_npy로 쓸 수 있으면 가짜 series_dir을 등록해 돌려준다. 못 쓰면 None(→ DICOM)."""
    d = os.path.join(NPY_DIR, str(pid))
    try:
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json") and os.path.exists(os.path.join(d, fn[:-5] + ".npy")):
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    m = json.load(f)
                ps = m["per_slice"]
                if "0020000E" in ps or "00200037" in ps or "00200013" not in ps or "00200032" not in ps:   # 00200037: 방향이 슬라이스마다 다른 합쳐진 시리즈
                    continue   # cl._load_volume_npy가 DICOM 경로로 넘기는 시리즈
                key = f"npy://{pid}"
                _PROG_NPY[key] = (m, os.path.join(d, fn[:-5] + ".npy"))
                return key
    except (OSError, ValueError, KeyError):
        pass
    return None


def drop_error_rows(status_prefix: str = "error") -> list[int]:
    """결과 파일의 landmarks/aec_total에서 seg_status가 status_prefix로 시작하는 환자를 모든 시트에서 지우고 그 PatientID를 돌려준다.
    다른 실행이 같은 파일에 쓰고 있으므로 같은 잠금 안에서 읽고 쓴다(cl._write_all은 중복이면 기존 행을 유지하므로, 지워야 새 결과가 들어간다)."""
    with cl._xlsx_lock(OUT):
        sh = cl._read_sheets(OUT)
        lm, at = sh.get(cl.SHEETS["landmarks"]), sh.get(cl.SHEETS["aec_total"])
        if lm is None or at is None:
            raise RuntimeError("결과 파일을 읽지 못해 중단(빈 파일로 덮어쓰지 않기 위해)")
        is_err = lambda d: d.seg_status.astype(str).str.startswith(status_prefix)
        bad = set(lm.PatientID[is_err(lm)]) | set(at.PatientID[is_err(at)])
        if bad:
            tmp = OUT + ".tmp.xlsx"
            with pd.ExcelWriter(tmp, engine="openpyxl") as w:
                for name, df in sh.items():   # 시트 순서 유지
                    if "PatientID" in df.columns:   # PatientID가 없는 시트(요약 등)는 그대로 둔다
                        df = df[~df.PatientID.isin(bad)]
                    df.to_excel(w, sheet_name=name, index=False)
            cl._replace_retry(tmp, OUT)
    return sorted(int(p) for p in bad)


def load_meta() -> pd.DataFrame:
    """질환별 5y_filtered의 patient_metadata를 PatientID 기준으로 합친 환자 마스터."""
    ms = [pd.read_excel(DATA / "5y_filtered" / f"{d}_5y_filtered.xlsx") for d in DISEASES]
    return pd.concat(ms).drop_duplicates("PatientID").reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def load_noev() -> pd.DataFrame:
    """6개 질환 진단이 한 건도 없는 환자(Patient_Master 기준)의 metadata. 이벤트 일수는 0."""
    x = pd.ExcelFile(DATA / "longitudinal_cohort_cleaned_before_roster.xlsx")
    M, E = x.parse("Patient_Master"), x.parse("Events_Long")
    m = M[~M.patient_id.isin(set(E[E.outcome.isin(DISEASES)].patient_id))]
    return pd.DataFrame({
        "PatientID": m.patient_id, "PatientSex": m.sex, "PatientAge": m.age_at_ct,
        "Height": m.height_cm_analysis, "Weight": m.weight_kg_analysis, "BMI": m.bmi_analysis,
        "hw_implausible": 0, "event_days_from_ct": 0, "onset_within_5y": 0}).reset_index(drop=True)


def dir_map(master: pd.DataFrame) -> dict:
    """환자ID -> CT 폴더. 반복 CT 환자는 폴더 연월이 인덱스 CT 연월과 같은 폴더를 우선한다."""
    ym = dict(zip(master.patient_id, pd.to_datetime(master.index_ct_date).dt.strftime("%Y%m")))
    m = {}
    for n in sorted(os.listdir(DICOM_BASE)):
        p = n.split("_")
        if n.endswith("_CT") and len(p) >= 4 and p[1].isdigit():
            pid = int(p[1])
            if pid not in m or p[2][:6] == ym.get(pid):
                m[pid] = os.path.join(DICOM_BASE, n)
    return m


def pick_stack(recs: list) -> list[str]:
    """같은 시리즈 이름으로 묶인 파일이 여러 스택이면(PACS 내보내기에서 방향·두께·간격이 다른 axial/MPR/스카우트가 한 시리즈로 합쳐진 경우, 예: PIVIEW5)
    스택 하나만 고른다. recs = [(InstanceNumber, z, IOP(반올림 튜플), SliceThickness, 파일경로)].
    방향·두께가 한 종류뿐이면(보통의 시리즈) 나누지 않고 전부 돌려준다."""
    if len({r[2] for r in recs}) == 1 and len({r[3] for r in recs}) == 1:
        return [r[4] for r in recs]
    recs = sorted(recs, key=lambda r: r[0])
    stacks, cur = [], []
    for r in recs:
        if cur:
            a, same = cur[-1], True
            same = a[2] == r[2] and a[3] == r[3] and abs(r[1] - a[1]) > 1e-3        # 방향·두께가 같고 z가 실제로 이동
            if same and len(cur) > 1:
                same = abs((r[1] - a[1]) - (cur[1][1] - cur[0][1])) <= 0.5          # 슬라이스 간격이 균일
            if not same:
                stacks.append(cur)
                cur = []
        cur.append(r)
    stacks.append(cur)

    def key(st):   # axial 우선 → 길이 300mm 이상 → z 범위가 큰 쪽 → 슬라이스 수
        iop = st[0][2]
        axial = len(iop) == 6 and abs(iop[0] * iop[4] - iop[1] * iop[3]) > 0.9
        zr = max(r[1] for r in st) - min(r[1] for r in st)
        return (len(st) >= 20 and axial, zr >= 300, zr, len(st))

    best = max(stacks, key=key)
    if not key(best)[0]:   # axial이면서 20장 이상인 스택이 없으면(예: 한 방향의 MPR 묶음) 나누지 않고 전부 돌려준다(이전 동작)
        return [r[4] for r in recs]
    return [r[4] for r in best]


def best_series_files(patient_dir: str) -> list[str] | None:
    """환자 폴더에서 best series(phase 우선순위, 커버리지 필터)의 파일 경로 목록을 고른다."""
    sb = cl.sb
    groups = sb.scan_patient_series(patient_dir)
    if not groups:
        return None
    rows = []
    for sd, g in groups.items():
        z = g["z_values"]
        ph = sb.classify_phase(sd)
        rows.append({"PatientID": 0, "series_desc": sd, "phase_rank": sb.PHASE_RANK[ph],
                     "z_range_mm": round(max(z) - min(z), 2) if len(z) >= 2 else 0.0,
                     "slice_thickness_mm": g["slice_thickness_mm"] if g["slice_thickness_mm"] is not None else 99.0})
    best = sb.pick_best(pd.DataFrame(rows)).iloc[0]["series_desc"]
    import pydicom
    recs = []
    for f in os.listdir(patient_dir):
        fp = os.path.join(patient_dir, f)
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
        except Exception:
            continue
        if sb._norm_series_name(getattr(ds, "SeriesDescription", "")) == best:
            ipp, iop = getattr(ds, "ImagePositionPatient", None), getattr(ds, "ImageOrientationPatient", None)
            recs.append((int(getattr(ds, "InstanceNumber", 0) or 0), float(ipp[2]) if ipp is not None else 0.0,
                         tuple(round(float(x), 2) for x in iop) if iop is not None else (),
                         str(getattr(ds, "SliceThickness", "")), fp))
    return pick_stack(recs)


COV_COLS = ["index_ct_date", "last_followup_date_raw", "analysis_censor_date", "followup_days_capped",
            "death_date", "death_days_from_ct", "death_event_after30d", "smoking_status", "drinking_status", "history_code_count",
            *(f"history_{h}" for h in ("dm", "htn", "dl", "ckd", "osteoporosis", "mi", "stroke", "heart_failure", "malignancy")),
            *(f"lab_{a}" for a in ("albumin", "alt", "ast", "creatinine", "crp", "glucose", "hba1c_ngsp", "hdl_cholesterol",
                                   "ldl_cholesterol", "total_protein", "triglyceride", "wbc")),
            "lab_test_count", "admission_count", "ed_count"]


_LAB = {"albumin": "알부민", "alt": "ALT", "ast": "AST", "creatinine": "크레아티닌", "crp": "CRP", "glucose": "혈당", "hba1c_ngsp": "HbA1c(NGSP)",
        "hdl_cholesterol": "HDL", "ldl_cholesterol": "LDL", "total_protein": "총단백", "triglyceride": "중성지방", "wbc": "WBC"}
_HIST = {"dm": "당뇨", "htn": "고혈압", "dl": "이상지질혈증", "ckd": "만성콩팥병", "osteoporosis": "골다공증", "mi": "심근경색", "stroke": "뇌졸중", "heart_failure": "심부전", "malignancy": "악성종양"}
_DESC = {
    "PatientID": ("환자 ID(연구등록번호)", "공통"), "PatientSex": ("성별", "5y_filtered"), "PatientAge": ("CT 시점 나이", "5y_filtered"),
    "Height": ("키(cm)", "5y_filtered"), "Weight": ("체중(kg)", "5y_filtered"), "BMI": ("BMI(kg/m²)", "5y_filtered"),
    "index_ct_date": ("기준 CT 촬영일(YYYY-MM-DD)", "Patient_Master"), "last_followup_date_raw": ("마지막 추적관찰일(원자료, 4. Last f_u date)", "Patient_Master"),
    "analysis_censor_date": ("분석용 censor일(마지막 추적일을 data_cutoff 2026-04-18로 제한)", "Patient_Master"),
    "followup_days_capped": ("CT~analysis_censor_date 추적일수. 사망 분석엔 쓰지 말 것(death_days_from_ct 사용)", "Patient_Master"),
    "death_date": ("사망일시(3-3)", "Patient_Master"), "death_days_from_ct": ("CT~사망 일수(사망자만)", "Patient_Master"),
    "death_event_after30d": ("CT 30일 이후 사망 여부(0/1)", "Patient_Master"),
    "smoking_status": ("흡연 상태(1-3, CT 당일 이전 기록만 → 결측 많음)", "Patient_Master"), "drinking_status": ("음주 상태(1-3, CT 당일 이전 기록만 → 결측 많음)", "Patient_Master"),
    "history_code_count": ("과거력(2.) 고유 ICD-10 코드 수", "Patient_Master"),
    "history_icd10_codes": ("과거력(2.) 고유 ICD-10 코드 목록(;로 연결)", "2. 과거력"),
    "lab_test_count": ("CT 전후 1개월 검사실 수치(6.) 건수", "Patient_Master"), "admission_count": ("입원(3-1) 건수", "Patient_Master"), "ed_count": ("응급실(3-2) 방문 건수", "Patient_Master"),
    **{f"{d}_event_days_from_ct": (f"CT~{d.upper()} 최초 진단 일수(결측=진단 없음/위험집단 아님, 진단일 0일은 1)", "5y_filtered") for d in DISEASES},
    **{f"{d}_first_icd10": (f"{d.upper()} 최초 진단 ICD-10(같은 날 복수는 ;)", "Events_Long") for d in DISEASES},
    **{f"history_{k}": (f"{v} 과거력 여부(0/1)", "Patient_Master") for k, v in _HIST.items()},
    **{f"lab_{k}": (f"{v} (CT 전후 1개월 최근값, 타당범위 필터 없음)", "Patient_Master") for k, v in _LAB.items()},
}


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """patient_metadata 컬럼별 설명·출처·비결측 수."""
    return pd.DataFrame({"column": df.columns, "description": [_DESC.get(c, ("", ""))[0] for c in df.columns],
                         "dtype": [str(t) for t in df.dtypes],
                         "n_nonnull": df.notna().sum().values, "pct_nonnull": (df.notna().mean().values * 100).round(1)})


def icd10_table(base: pd.DataFrame) -> pd.DataFrame:
    """patient_metadata에 나온 ICD-10 코드 목록: 코드명(원자료의 'ICD-10 코드(코드명)') + 과거력 보유 환자 수 + 질환별 최초 진단 환자 수."""
    h = pd.read_excel(next((DATA / "이홍선(2025401459)").glob("2. *.xlsx")), usecols=["ICD-10 코드", "ICD-10 코드(코드명)"]).dropna().drop_duplicates("ICD-10 코드")
    E = pd.read_excel(DATA / "longitudinal_cohort_cleaned_before_roster.xlsx", sheet_name="Events_Long", usecols=["icd10_code", "icd10_name"]).dropna().drop_duplicates("icd10_code")
    names = {**dict(zip(E.icd10_code, E.icd10_name)), **dict(zip(h["ICD-10 코드"], h["ICD-10 코드(코드명)"]))}
    cnt = lambda col: base[col].dropna().str.split(";").explode().value_counts()
    t = pd.DataFrame({"n_history": cnt("history_icd10_codes"), **{f"n_first_{d}": cnt(f"{d}_first_icd10") for d in DISEASES}}).fillna(0).astype(int)
    t.insert(0, "name", t.index.map(names))
    return t.rename_axis("icd10_code").reset_index().sort_values(["n_history", "icd10_code"], ascending=[False, True])


def load_covariates() -> pd.DataFrame:
    """이홍선 원자료를 정리한 longitudinal_cohort_cleaned_before_roster.xlsx(Patient_Master)에서 보정변수만 PatientID 기준으로 뽑는다.
    검사실 수치는 타당범위 필터가 없고(정리본_재검증 §3-6), 사망 분석엔 followup_days_capped 대신 death_days_from_ct를 쓴다."""
    f = DATA / "longitudinal_cohort_cleaned_before_roster.xlsx"
    M = pd.read_excel(f, sheet_name="Patient_Master").rename(columns={"patient_id": "PatientID"})
    dup = {"sex", "age_at_ct", "height_cm_analysis", "weight_kg_analysis", "bmi_analysis"}   # PatientSex/PatientAge/Height/Weight/BMI와 같은 값
    rest = [c for c in M.columns if c not in {"PatientID", *COV_COLS, *dup}]   # 정리본의 나머지 컬럼(30일 landmark, MACE, disease censor, 추적 플래그, CT 오더 등)을 전부 넣는다
    _DESC.update({c: ("Patient_Master 컬럼(정의는 longitudinal_cohort_cleaned_before_roster.xlsx의 Data_Dictionary)", "Patient_Master") for c in rest})
    M = M[["PatientID", *COV_COLS, *rest]]
    M["analysis_censor_date"] = pd.to_datetime(M.analysis_censor_date).dt.strftime("%Y-%m-%d")  # 시각 제거
    h =pd.read_excel(next((DATA / "이홍선(2025401459)").glob("2. *.xlsx"))).dropna(subset=["ICD-10 코드"])  # 과거력(CT 이전)
    M = M.merge(h.groupby("연구등록번호")["ICD-10 코드"].agg(lambda s: ";".join(sorted(set(s)))).rename("history_icd10_codes"),
                left_on="PatientID", right_index=True, how="left")
    E = pd.read_excel(f, sheet_name="Events_Long").sort_values("event_days_from_ct")  # 질환별 첫 진단 ICD-10(같은 날 여러 코드는 ;로 연결)
    E = E[E.event_days_from_ct == E.groupby(["outcome", "patient_id"]).event_days_from_ct.transform("min")]
    e = E.groupby(["outcome", "patient_id"]).icd10_code.agg(lambda s: ";".join(sorted(set(s.dropna())))).unstack(0)
    M = M.merge(e[[d for d in DISEASES if d in e]].add_suffix("_first_icd10"), left_on="PatientID", right_index=True, how="left")
    return M.merge(load_raw_details(f), left_on="PatientID", right_index=True, how="left")


def load_raw_details(f) -> pd.DataFrame:
    """원자료 상세 기록(사망 원인, 입원·응급 진단, 과거력 일자, 흡연·음주 전체 시점, CT와 시간차가 큰 키·체중)을 환자당 1행으로 줄인다. LAB 전체 수치는 제외."""
    raw = DATA / "이홍선(2025401459)"
    rd = lambda pat, **kw: pd.read_excel(next(raw.glob(pat)), **kw).rename(columns={"연구등록번호": "PatientID"})
    uniq = lambda s: ";".join(sorted({str(x) for x in s.dropna() if str(x) != ""}))   # 중복 제거 후 ;로 연결
    d = lambda s: pd.to_datetime(s).dt.strftime("%Y-%m-%d")
    out, desc = {}, {}

    D = pd.read_excel(f, sheet_name="Death_Detail")   # 사망 상세: 종류와 원인(가·나·다)
    D = D[D.attr_name != "사망일시"].drop_duplicates(["patient_id", "attr_name"]).pivot(index="patient_id", columns="attr_name", values="item_value")
    names = {"사망의 종류": "death_type", "(가)의 원인(중간선행사인)": "death_cause_a", "(나)의 원인(선행사인)": "death_cause_b",
             "(다)의 원인": "death_cause_c", "(가) 내지 (라)와 관계없는 기타의 신체상황": "death_other_condition"}
    for k, v in names.items():
        out[v] = D[k] if k in D else pd.Series(dtype=object)
        desc[v] = (f"사망 상세(3-3 사망원인): {k}", "3-3 사망")

    for tag, pat, ind, outd, dept, label in (("admission", "3-1*", "병동입원일시", "퇴실일시", "입원진료과코드(코드명)", "입원(3-1)"),
                                             ("ed", "3-2*", "도착일자", "퇴실일자", None, "응급실(3-2)")):
        X = rd(pat)
        g = X.groupby("PatientID")
        out[f"{tag}_first_date"] = d(g[ind].min()); desc[f"{tag}_first_date"] = (f"{label} 첫 일자", "원자료")
        out[f"{tag}_last_date"] = d(g[ind].max()); desc[f"{tag}_last_date"] = (f"{label} 마지막 일자", "원자료")
        out[f"{tag}_icd10_codes"] = g["ICD-10 코드"].agg(uniq); desc[f"{tag}_icd10_codes"] = (f"{label} 진단 ICD-10 목록(;)", "원자료")
        P = X[X["주상병여부"].astype(str).isin(["1", "Y", "Yes"]) | X["주상병여부(코드명)"].astype(str).eq("Yes")]
        out[f"{tag}_primary_icd10_codes"] = P.groupby("PatientID")["ICD-10 코드"].agg(uniq); desc[f"{tag}_primary_icd10_codes"] = (f"{label} 주상병 ICD-10 목록(;)", "원자료")
        if dept:
            out[f"{tag}_dept_list"] = g[dept].agg(uniq); desc[f"{tag}_dept_list"] = (f"{label} 입원 진료과 목록(;)", "원자료")

    H = rd("2.*").dropna(subset=["ICD-10 코드"])   # 과거력 최초진단일
    H["과거력최초진단일"] = pd.to_datetime(H["과거력최초진단일"], errors="coerce")
    out["history_first_dx_date"] = d(H.groupby("PatientID")["과거력최초진단일"].min()); desc["history_first_dx_date"] = ("과거력(2.) 가장 이른 최초진단일", "2. 과거력")
    H["code_date"] = H["ICD-10 코드"] + ":" + d(H["과거력최초진단일"]).fillna("")
    out["history_icd10_dated"] = H.groupby("PatientID")["code_date"].agg(uniq); desc["history_icd10_dated"] = ("과거력 ICD-10별 최초진단일(코드:YYYY-MM-DD, ;로 연결)", "2. 과거력")

    L = rd("1-3*")   # 흡연·음주: CT 이전 기록만 쓴 smoking_status와 달리 CT 이후 서식까지 포함한 가장 최근 값
    L["서식일자"] = pd.to_datetime(L["서식일자"], errors="coerce")
    for key, name, lab in (("흡연력", "smoking", "흡연"), ("음주력(간호", "drinking", "음주"), ("음주력(종류", "drinking_detail", "음주 상세(종류·양·횟수/wk)")):
        S = L[L["Attr코드(코드명)"].astype(str).str.startswith(key)].sort_values("서식일자").groupby("PatientID").last()
        out[f"{name}_latest_any"] = S["항목값"]; desc[f"{name}_latest_any"] = (f"{lab}: CT 이후 포함 가장 최근 기록 값(1-3)", "1-3 흡연·음주")
        if name != "drinking_detail":
            out[f"{name}_latest_any_date"] = d(S["서식일자"]); desc[f"{name}_latest_any_date"] = (f"{lab} 가장 최근 기록 일자", "1-3 흡연·음주")

    K = rd("1-1*")   # 키·체중·BMI: 365일 제한 없이 CT와 일시차가 가장 작은 기록
    K["gap"] = K["일시차2"].abs()
    for code, name, lab in (("신장", "height_nearest_any_cm", "키(cm)"), ("체중", "weight_nearest_any_kg", "체중(kg)"), ("BMI", "bmi_nearest_any", "BMI")):
        S = K[K["임상관찰코드(코드명)"].astype(str).str.contains(code)].sort_values("gap").groupby("PatientID").first()
        out[name] = pd.to_numeric(S["측정값"], errors="coerce"); desc[name] = (f"{lab}: 시점 제한 없이 CT와 가장 가까운 기록(1-1)", "1-1 임상관찰")
        out[name + "_gap_days"] = S["일시차2"]; desc[name + "_gap_days"] = (f"{lab} 기록의 CT와의 일시차(일, 부호는 원자료)", "1-1 임상관찰")

    R = rd("1-2*")   # 예진정보의 키·체중
    R["gap"] = R["일시차2"].abs()
    for col, name, lab in (("신장", "height_previsit_nearest_cm", "키(cm)"), ("체중", "weight_previsit_nearest_kg", "체중(kg)")):
        S = R.dropna(subset=[col]).sort_values("gap").groupby("PatientID").first()
        out[name] = pd.to_numeric(S[col], errors="coerce"); desc[name] = (f"{lab}: 예진정보(1-2)에서 CT와 가장 가까운 기록", "1-2 예진정보")
        out[name + "_gap_days"] = S["일시차2"]; desc[name + "_gap_days"] = (f"{lab} 예진정보 기록의 CT와의 일시차(일)", "1-2 예진정보")
    V = rd("6.*")   # LAB: 값 자체는 Patient_Master(lab_*)에 있으므로 그 값이 CT에서 며칠 떨어진 검사인지만 항목별로 넣는다
    V["abs_gap"] = V["검사일시차"].abs()
    codes = {"B105002": "wbc", "B257001": "ast", "B258001": "alt", "C220001": "total_protein", "C221002": "albumin", "C224301": "crp",
             "C242001": "hdl_cholesterol", "C243001": "ldl_cholesterol", "C244302": "triglyceride", "C371101": "glucose",
             "C375002": "creatinine", "L004101200": "hba1c_ngsp"}
    for code, a_ in codes.items():
        S = V[V["처방코드_1"] == code].sort_values("abs_gap").groupby("PatientID").first()
        out[f"lab_{a_}_gap_days"] = S["검사일시차"]; desc[f"lab_{a_}_gap_days"] = (f"{_LAB[a_]} 검사의 CT와의 일시차(일, 원자료 검사일시차는 부호 없는 절댓값 0~30)", "6. LAB추적")
    _DESC.update(desc)
    return pd.DataFrame(out)


def finalize(meta: pd.DataFrame, tag: str = None) -> None:
    """지금까지의 결과로 한 파일(prognosis_5y_landmark_filtered.xlsx)을 만든다.
    시트: patient_metadata(환자 1행, 질환별 {질환}_event_days_from_ct) + landmarks/aec_total/VAT/SAT/TAMA_total(공통, PatientID로 연결).
    filtered 조건(metadata 완비, seg ok, aec 비flat, hw_implausible=0)은 mm._filter_sheets와 동일."""
    sheets = {k: v for k, v in pd.read_excel(OUT, sheet_name=None).items() if k not in cl.EXTRA_SHEETS}  # 메타 시트는 아래에서 새로 만든다
    metas = {}
    for d in DISEASES:
        pm = pd.read_excel(DATA / "5y_filtered" / f"{d}_5y_filtered.xlsx")
        ok = mm._filter_sheets({"patient_metadata": pm, **{k: sheets[k] for k in ("landmarks", "aec_total")}})
        metas[d] = ok["patient_metadata"]
        print(d, "환자", len(pm), "-> filtered", len(metas[d]), "5년이내", int(metas[d].onset_within_5y.sum()))
    nv = mm._filter_sheets({"patient_metadata": load_noev(), **{k: sheets[k] for k in ("landmarks", "aec_total")}})["patient_metadata"]
    print("질환진단 없음 환자", len(load_noev()), "-> filtered", len(nv))
    has_tot = set(sheets["VAT_total"].PatientID)  # 체성분 total까지 계산된 환자만 filtered에 넣는다
    metas = {d: m[m.PatientID.isin(has_tot)] for d, m in metas.items()}
    nv = nv[nv.PatientID.isin(has_tot)]
    ids = set().union(*(set(m.PatientID) for m in metas.values()), set(nv.PatientID))
    # 질환별 metadata를 환자 1행으로 통합: 공통 컬럼 + {질환}_event_days_from_ct (바이너리 컬럼 제외, 해당 질환 위험집단이 아니거나 진단 없으면 결측)
    base = pd.concat(m.drop(columns=["event_days_from_ct", "onset_within_5y", "hw_implausible"]) for m in [*metas.values(), nv]).drop_duplicates("PatientID")
    for d, m in metas.items():
        base[f"{d}_event_days_from_ct"] = base.PatientID.map(m.set_index("PatientID").event_days_from_ct.clip(lower=1))  # 진단일 0일은 1로(결측=진단 없음/위험집단 아님)
    base = base.merge(load_covariates(), on="PatientID", how="left")  # 보정변수(추적관찰·사망·흡연/음주·과거력·검사실·입원/응급)
    base = base.reset_index(drop=True)
    out = {"metadata_summary": column_summary(base), "icd10_codes": icd10_table(base), "patient_metadata": base}  # 맨 앞 시트: patient_metadata 컬럼 요약, 그다음 ICD-10 코드표
    out.update({k: v[v.PatientID.isin(ids)] for k, v in sheets.items()})
    mm._write_xlsx(str(OUT_DIR / "prognosis_5y_landmark_filtered.xlsx"), out)
    if tag:   # 단계별 스냅샷(1차=part1 완료+part0 중간, 2차=전체 완료)
        shutil.copy(OUT_DIR / "prognosis_5y_landmark_filtered.xlsx", OUT_DIR / f"prognosis_5y_landmark_filtered_{tag}.xlsx")
    with cl._xlsx_lock(OUT):  # 전체 결과 파일에도 같은 메타데이터 시트 3개를 맨 앞에 넣는다(이후 저장에서도 보존: cl.EXTRA_SHEETS)
        sh = cl._read_sheets(OUT)
        sh = {**{k: out[k] for k in cl.EXTRA_SHEETS}, **{k: v for k, v in sh.items() if k not in cl.EXTRA_SHEETS}}
        tmp = OUT + ".tmp.xlsx"
        with pd.ExcelWriter(tmp, engine="openpyxl") as w:
            for k, v in sh.items():
                v.to_excel(w, sheet_name=k, index=False)
        cl._replace_retry(tmp, OUT)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    if "--drop-no-dicom" in sys.argv:   # DICOM 폴더가 없는 환자(seg_status=no_dicom)의 행을 결과 파일에서 지우고 끝낸다
        print("no_dicom 환자 삭제:", drop_error_rows("no_dicom"))
        return
    M = pd.read_excel(DATA / "longitudinal_cohort_cleaned_before_roster.xlsx", sheet_name="Patient_Master")
    meta = load_meta()
    aec_only = "--aec-only" in sys.argv  # 병렬 보조 실행: total 불필요 환자의 landmarks/aec_total만 저장(filtered는 본 실행이 갱신)
    retry = "--retry-errors" in sys.argv  # 결과 파일에서 seg_status가 error인 환자만 지우고 다시 처리(자체 체크포인트, 다른 실행과 동시에 가능)
    part_k, part_n = (map(int, sys.argv[sys.argv.index("--part") + 1].split("/")) if "--part" in sys.argv else (0, 1))  # 병렬 분할 실행: --part 0/2, --part 1/2 (pending을 번갈아 나눠 가진다)
    ckpt = CKPT.replace("_checkpoint", "_aeconly_checkpoint" if aec_only else "_retry_checkpoint") if (aec_only or retry) else CKPT
    if part_n > 1:
        ckpt = CKPT.replace("_checkpoint", f"_part{part_k}_checkpoint")
    if "--finalize" in sys.argv:
        return finalize(meta)
    ok = meta[meta[["Height", "Weight", "BMI"]].notna().all(axis=1)]
    nv = load_noev()
    meta_ok = set(ok.PatientID) | set(nv.PatientID[nv[["Height", "Weight", "BMI"]].notna().all(axis=1)])  # 질환진단 없는 환자도 total 대상
    dmap = dir_map(M)
    attempted = cl._load_checkpoint(ckpt)
    if retry:
        retry_ids = drop_error_rows()
        if "--ids" in sys.argv:   # 행이 없는(처리 중 끊긴) 환자를 직접 지정: --ids 5886098,1234567 (이미 행이 있는 환자는 지워지지 않아 새 결과가 반영되지 않는다)
            retry_ids = sorted(set(retry_ids) | {int(x) for x in sys.argv[sys.argv.index("--ids") + 1].split(",")})
        print(f"error 환자 {len(retry_ids)}명 삭제 후 재처리: {retry_ids}", flush=True)
        attempted = set()
    elif os.path.exists(OUT):
        sh = pd.read_excel(OUT, sheet_name=["landmarks", "aec_total", "VAT_total"])
        attempted |= set(sh["landmarks"].PatientID.astype(int))
        # total 없이 처리된 대상 환자(seg ok, aec 비flat)는 다시 돌린다
        flat = sh["aec_total"].set_index("PatientID").aec_flat.astype(str).str.lower().isin(["false", "0", "0.0"])
        segok = set(sh["landmarks"].PatientID[sh["landmarks"].seg_status == "ok"])
        redo = (attempted & segok) - set(sh["VAT_total"].PatientID)   # 전원 total: 키/체중·aec flat 조건 없이 total 없는 seg ok 환자 전부
        attempted -= redo if not aec_only else set()
        print(f"total 누락 재처리 {len(redo)}명", flush=True)
    all_ids = set(M.patient_id.astype(int)) | set(meta.PatientID)  # landmark/AEC는 마스터 전원, 체성분 total은 키/체중 완비자만
    if retry:
        all_ids = set(retry_ids)
    pending = sorted((p for p in all_ids if p not in attempted), key=lambda p: (p not in meta_ok, p))  # total 필요한 환자 먼저
    if aec_only:
        pending = [p for p in pending if p not in meta_ok]
    pending = pending[part_k::part_n]
    print(f"전체 {len(all_ids)}명 | 키/체중/BMI 완비 {len(meta_ok)}명 | 완료 {len(attempted)}명 | 처리 예정 {len(pending)}명", flush=True)

    fin = not aec_only and part_k == 0   # filtered 파일은 한 실행만 갱신(임시파일 충돌 방지)
    if fin:
        first = not (OUT_DIR / "prognosis_5y_landmark_filtered_1차.xlsx").exists()
        finalize(meta, tag="1차" if first else None)  # 시작할 때도 filtered를 먼저 갱신(처음 시작이면 1차 스냅샷도 저장)
    want = (lambda *a: False) if aec_only else (lambda pid, st, flat: st == "ok")   # 전원 total(키/체중·aec flat 조건 제거)
    rows, aec_t, aec_m, tot = [], [], [], {s: [] for s in cl.TOTAL_SHEETS}
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        for pid in tqdm(pending, desc="5y 코호트 랜드마크"):
            pdir = dmap.get(pid)
            nkey = prog_npy_dir(pid) if ((aec_only or retry) and "--dicom" not in sys.argv) else None   # --aec-only/--retry-errors는 npy/json이 있으면 DICOM 스캔·복사 없이 그걸 쓴다(--dicom이면 쓰지 않고 원본 DICOM으로)
            files = [] if nkey else (best_series_files(pdir) if pdir else None)
            if nkey:
                try:
                    row, _, at, am = cl.process_patient(pid, nkey, tmp, report_path=None, want_total=want)
                    for sh, r in (row.pop("_total", None) or {}).items():
                        tot[sh].append(r)
                except Exception as e:
                    tqdm.write(f"  PID {pid} traceback: {traceback.format_exc()}")
                    row = cl._empty_landmark_row(pid, None, None, f"error:{type(e).__name__}:{e}")
                    at, am = cl._aec_rows(pid, row, None)
                _PROG_NPY.pop(nkey, None)
            elif not files:
                row = cl._empty_landmark_row(pid, None, None, "no_dicom" if not pdir else "no_series")
                at, am = cl._aec_rows(pid, row, None)
            else:
                sdir = os.path.join(tmp, f"s{pid}")
                os.makedirs(sdir)
                for f in files:
                    shutil.copy(f, sdir)
                try:
                    row, _, at, am = cl.process_patient(pid, sdir, tmp, report_path=None, want_total=want)
                    for sh, r in (row.pop("_total", None) or {}).items():
                        tot[sh].append(r)
                except Exception as e:
                    tqdm.write(f"  PID {pid} traceback: {traceback.format_exc()}")
                    row = cl._empty_landmark_row(pid, None, None, f"error:{type(e).__name__}:{e}")
                    at, am = cl._aec_rows(pid, row, None)
                shutil.rmtree(sdir, ignore_errors=True)
            rows.append(row); aec_t.append(at); aec_m.append(am)
            tqdm.write(f"  PID {pid}: {row['seg_status']}")
            attempted.add(pid); n += 1
            if n % cl.BATCH_SIZE == 0 or n == len(pending):
                cl._write_all(OUT, rows, [], aec_t, aec_m, total=tot)
                cl._save_checkpoint(ckpt, attempted)
                if fin and n % FINALIZE_EVERY == 0:  # 중간 저장: 질환별 filtered 파일도 같이 갱신
                    try:
                        finalize(meta)
                    except Exception as e:
                        tqdm.write(f"  finalize 실패 ({type(e).__name__}: {e})")
                rows, aec_t, aec_m, tot = [], [], [], {s: [] for s in cl.TOTAL_SHEETS}
    if fin:
        finalize(meta, tag="2차")
    print("완료")


if __name__ == "__main__":
    main()
