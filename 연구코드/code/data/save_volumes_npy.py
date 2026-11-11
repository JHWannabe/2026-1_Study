r"""
CT 볼륨을 HU(int16) numpy(.npy) + DICOM 헤더(.json)로 저장해 이후 DICOM 재파싱을 없앤다. 모드는 셋이다.

[기본] 강남/신촌 최종 데이터셋 환자의 모든 series를 각각 저장 (Pre/Portal 등 비교용)
  python save_volumes_npy.py [신촌|강남 ...]   (인자 없으면 둘 다)
  대상: {site}_final_dataset.xlsx의 PatientID
  출력: {NPY_ROOT}\{slug}\{pid}\{pid}_{series 폴더 첫 단어}.npy  (n_slices, H, W) int16 HU, InstanceNumber 오름차순
        (aec_1 = 가장 작은 InstanceNumber와 같은 방향 — k 인덱스/instance로 landmark와 바로 맞출 수 있음)
        같은 위치의 .json  spacing_zyx_mm, inst_first/last, z_first/last, series_folder +
        header(첫 슬라이스 DICOM 태그 전부, private 포함), per_slice(슬라이스마다 달라지는 태그 값 목록), keywords
        data/{site}/npy_status.csv 실패/스킵 사유 (재실행 시 이미 있는 .npy는 건너뜀)
  np.load(path, mmap_mode="r")로 열면 일부 슬라이스만 즉시 읽는다.

[--missing] xlsx와 무관하게 드라이브의 환자 폴더를 직접 훑어 npy가 없는 환자를 찾아 변환한다
  python save_volumes_npy.py --missing --scan   대상만 집계해 출력/저장 (변환 안 함)
  python save_volumes_npy.py --missing --run    scan 결과의 환자를 전부 변환 (이미 있는 series는 건너뜀)
  누락 = 환자 폴더의 series 폴더 수 > 완성된 .npy 수 (환자 폴더 + 공유 폴더, .tmp.npy 제외).
    none: .npy가 하나도 없는 환자,  partial: 일부 series만 변환된 환자
  결과: data/{site}/npy_missing_scan.csv (scan), data/{site}/npy_missing_status.csv (run)

[--prognosis] 예후데이터 환자 중 npy가 없는 환자의 best series 1개만 저장한다
  python save_volumes_npy.py --prognosis          (전체 저장)
  python save_volumes_npy.py --prognosis --dry 5  (앞 5명 선정 결과만 출력, 저장 안 함)
  대상: longitudinal_cohort_cleaned_before_roster.xlsx Patient_Master 환자 중 DICOM 폴더가 있고 npy가 없는 환자
  입력: D:\데이터서비스팀 요청(이홍선)\{No}_{PatientID}_{date}_CT\*.dcm  (시리즈 폴더 구분 없이 한 폴더)
  선정: 7_select_best_series.py와 같은 기준 (SeriesInstanceUID 단위). 단 ImageType이 ORIGINAL axial인 시리즈만 후보이고,
    없으면 저장하지 않고 status=no_original_axial로 기록한다 (MPR/파생·localizer 영상 제외).
    1) range_ok(z_range_mm >= MIN_RANGE_MM) 우선  2) phase: Portal > Contrast > Delay > Arterial > Pre/NonContrast > Unknown > Scout/Tracking
    3) z_range_mm 큰 쪽  4) slice thickness 얇은 쪽
  출력: {NPY_ROOT}\예후데이터\{PatientID}\{PatientID}_{시리즈 첫 단어}.npy / .json  (이미 있으면 건너뜀)
        data/예후데이터/prognosis_npy_best_status.csv  PatientID, series_desc, phase, status
"""

import functools
import glob
import importlib.util
import json
import os
import re
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")  # 병원 UID 길이/형식 경고(VR UI) 숨김
SITES = ["신촌", "강남"]
DATA_DIR = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data"
FINAL = {"강남": "gangnam_final_dataset.xlsx", "신촌": "sinchon_final_dataset.xlsx"}
NPY_ROOT = r"\\heart-lab.tail38ea6b.ts.net\home\jhjun\npy_json"   # 새 npy/json 저장 위치: {NPY_ROOT}\{slug}\{pid}\{pid}_{tag}.npy
SLUG = {"강남": "gangnam", "신촌": "sinchon"}
MIN_FREE_GB = 30   # 저장 드라이브 여유가 이보다 적으면 새 변환을 멈춘다(디스크가 가득 차 파일이 깨지는 것 방지)
N_WORKERS = 4  # E: 드라이브 I/O 한계에 맞춘 값
PROG_WORKERS = 3   # --prognosis: 메모리(페이징 파일) 부족 에러가 있었어서 줄였다
MASTER_XLSX = rf"{DATA_DIR}\예후데이터\longitudinal_cohort_cleaned_before_roster.xlsx"
PROG_STATUS_CSV = rf"{DATA_DIR}\예후데이터\prognosis_npy_best_status.csv"


# 모든 DICOM 태그(픽셀 제외, private 포함)를 저장한다: header=첫 슬라이스 전체,
# per_slice=슬라이스마다 값이 달라지는 태그만 {tag: [슬라이스별 값]} (InstanceNumber 순서), keywords=tag→이름.
def _dicom_tags(slices: list) -> dict:
    hdrs = [{k: v for k, v in ds.to_json_dict().items() if k != "7FE00010"} for _, ds in slices]
    per_slice = {k: [h.get(k) for h in hdrs] for k in hdrs[0]
                 if any(h.get(k) != hdrs[0][k] for h in hdrs[1:])}
    kw = {k: pydicom.datadict.keyword_for_tag(int(k, 16)) for k in hdrs[0]}
    return {"header": hdrs[0], "per_slice": per_slice, "keywords": kw}


# 저장할 드라이브의 여유 공간이 MIN_FREE_GB 미만이면 True.
def _disk_low(stem: str) -> bool:
    d = os.path.dirname(stem)
    while not os.path.exists(d):
        d = os.path.dirname(d)
    return shutil.disk_usage(d).free < MIN_FREE_GB * 1e9


# series 폴더 1개의 DICOM을 읽어 stem.npy / stem.json으로 저장한다. 상태 문자열을 반환.
def save_series(series_dir: str, stem: str, old_stem: str = "") -> str:
    if os.path.exists(stem + ".npy") or (old_stem and os.path.exists(old_stem + ".npy")):
        return "skip"          # old_stem: 공유 폴더로 옮기기 전, 환자 폴더에 이미 만들어 둔 파일
    if _disk_low(stem):
        return "disk_low"
    slices = []
    for fname in os.listdir(series_dir):
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, fname))
            slices.append((int(ds.InstanceNumber), ds))
        except Exception:
            continue
    return save_slices(slices, stem, os.path.basename(series_dir))


# (InstanceNumber, dataset) 목록을 HU int16 npy + json으로 저장한다. 상태 문자열을 반환.
def save_slices(slices: list, stem: str, series_label: str) -> str:
    if not slices:
        return "no_slices"
    if _disk_low(stem):
        return "disk_low"
    def _decodable(ds):    # 픽셀 데이터가 없는 손상 슬라이스(강남 3201710/3639829/8965018)는 빼고 저장한다
        try:
            ds.pixel_array
            return True
        except Exception:
            return False
    slices = sorted((t for t in slices if _decodable(t[1])), key=lambda t: t[0])
    if not slices:
        return "no_slices"

    if len({ds.pixel_array.shape for _, ds in slices}) != 1:
        return "inconsistent_shape"
    vol = np.stack([
        ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1))
        + float(getattr(ds, "RescaleIntercept", 0))
        for _, ds in slices
    ])
    vol = np.clip(np.rint(vol), -32768, 32767).astype(np.int16)

    first = slices[0][1]
    z = [float(ds.ImagePositionPatient[2]) for _, ds in slices]
    # PixelSpacing 태그가 없는 시리즈(신촌 9892046)는 ReconstructionDiameter(FOV)/픽셀 수로 구한다.
    px = ([float(v) for v in first.PixelSpacing] if "PixelSpacing" in first
          else [float(first.ReconstructionDiameter) / first.Rows, float(first.ReconstructionDiameter) / first.Columns])
    dz = abs(z[-1] - z[0]) / (len(z) - 1) if len(z) > 1 else float(getattr(first, "SliceThickness", 0))
    meta = {"spacing_zyx_mm": [dz, px[0], px[1]], "inst_first": slices[0][0], "inst_last": slices[-1][0],
            "z_first": z[0], "z_last": z[-1], "series_folder": series_label}
    meta.update(_dicom_tags(slices))

    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    np.save(stem + ".tmp.npy", vol)               # 중간에 끊겨도 반쪽짜리 .npy가 남지 않게 임시파일 후 rename
    os.replace(stem + ".tmp.npy", stem + ".npy")
    return "ok"


# 환자 1명: 환자 폴더의 모든 series 폴더를 각각 {pid}_{series 첫 단어}.npy로 저장한다.
# 첫 단어가 겹치는 series가 있으면(대소문자 무시) 그 환자는 폴더명 전체(공백→_)를 쓴다.
def save_patient(args: tuple) -> list[tuple]:
    site, pid = args
    patient_dir = rf"E:\영상제공\{site}\{site}_axial\{pid}"
    if not os.path.isdir(patient_dir):
        return [(pid, "", "no_patient_dir")]
    subs = sorted(d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d)))
    firsts = [(d.split() or ["series"])[0].lower() for d in subs]
    dup = len(set(firsts)) != len(firsts)
    out = []
    for d, f in zip(subs, firsts):
        tag = "_".join(d.split()) if dup else d.split()[0]
        try:
            st = save_series(os.path.join(patient_dir, d), rf"{NPY_ROOT}\{SLUG[site]}\{pid}\{pid}_{tag}",
                             rf"{patient_dir}\{pid}_{tag}")
        except Exception as e:                       # 한 시리즈 오류로 전체 실행이 멈추지 않게 기록만 하고 넘어간다
            st = f"error:{type(e).__name__}:{str(e)[:120]}"
        out.append((pid, d, st))
    return out or [(pid, "", "no_series_folder")]


def run_site(site: str):
    df = pd.read_excel(rf"{DATA_DIR}\{site}\{FINAL[site]}", usecols=["PatientID"])
    jobs = sorted(((site, int(p)) for p in df["PatientID"]), key=lambda j: -j[1])   # PatientID 역순
    with ProcessPoolExecutor(N_WORKERS) as ex:
        res = [row for rows in tqdm(ex.map(save_patient, jobs), total=len(jobs), desc=site) for row in rows]
    st = pd.DataFrame(res, columns=["PatientID", "series_folder", "status"])
    st.to_csv(rf"{DATA_DIR}\{site}\npy_status.csv", index=False)
    print(site, st["status"].value_counts().to_dict())


# ---- --missing: 드라이브를 훑어 npy가 없는 환자를 찾아 변환 ----

# 환자 폴더 1개의 series 폴더 수와 완성된 npy 수를 센다.
def _scan_one(args: tuple) -> tuple:
    site, pid = args
    d = rf"E:\영상제공\{site}\{site}_axial\{pid}"
    names = os.listdir(d)
    n_series = sum(os.path.isdir(os.path.join(d, n)) for n in names)
    sh = rf"{NPY_ROOT}\{SLUG[site]}\{pid}"        # 변환 결과는 공유 폴더에 있다(환자 폴더에 남은 옛 파일도 함께 센다)
    names += os.listdir(sh) if os.path.isdir(sh) else []
    n_npy = sum(n.endswith(".npy") and not n.endswith(".tmp.npy") for n in names)
    return pid, n_series, n_npy


def scan_missing(site: str) -> pd.DataFrame:
    base = rf"E:\영상제공\{site}\{site}_axial"
    pids = [n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n))]
    with ProcessPoolExecutor(N_WORKERS) as ex:
        rows = list(tqdm(ex.map(_scan_one, [(site, p) for p in pids], chunksize=20), total=len(pids), desc=f"scan {site}"))
    df = pd.DataFrame(rows, columns=["PatientID", "n_series", "n_npy"])
    df["kind"] = ["none" if n == 0 else "partial" if n < s else "done" for s, n in zip(df.n_series, df.n_npy)]
    df.loc[df.n_series == 0, "kind"] = "no_series"
    df.to_csv(rf"{DATA_DIR}\{site}\npy_missing_scan.csv", index=False)
    print(site, df["kind"].value_counts().to_dict(),
          "| 변환 필요 series:", int((df.n_series - df.n_npy)[df.kind.isin(["none", "partial"])].sum()))
    return df


def run_missing(site: str):
    df = pd.read_csv(rf"{DATA_DIR}\{site}\npy_missing_scan.csv")
    jobs = sorted(((site, int(p)) for p in df.loc[df.kind.isin(["none", "partial"]), "PatientID"]), key=lambda j: -j[1])  # PatientID 역순
    with ProcessPoolExecutor(N_WORKERS) as ex:
        res = [r for rows in tqdm(ex.map(save_patient, jobs), total=len(jobs), desc=site) for r in rows]
    st = pd.DataFrame(res, columns=["PatientID", "series_folder", "status"])
    st.to_csv(rf"{DATA_DIR}\{site}\npy_missing_status.csv", index=False)
    print(site, st["status"].value_counts().to_dict())


# ---- --prognosis: 예후데이터 환자의 best series 1개 ----

# 파일명이 숫자로 시작해 일반 import가 안 되므로 경로로 불러와 phase 분류/우선순위를 재사용한다.
@functools.lru_cache(maxsize=1)
def _sel():
    spec = importlib.util.spec_from_file_location("select_best", os.path.join(HERE, "7_select_best_series.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 환자 폴더 헤더만 읽어 (SeriesInstanceUID → {desc, files, z, thickness}) 로 모은다.
def _scan_groups(patient_dir: str) -> dict:
    groups: dict = {}
    for fname in os.listdir(patient_dir):
        fp = os.path.join(patient_dir, fname)
        if not fname.lower().endswith(".dcm") or not os.path.isfile(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
        except Exception:
            continue
        g = groups.setdefault(str(getattr(ds, "SeriesInstanceUID", "")), {
            "desc": _sel()._norm_series_name(getattr(ds, "SeriesDescription", "")),
            "files": [], "z": [], "thick": None})
        g["files"].append(fp)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            g["z"].append(float(ipp[2]))
        if g["thick"] is None and getattr(ds, "SliceThickness", None) is not None:
            g["thick"] = float(ds.SliceThickness)
    return groups


# 시리즈의 첫 파일 헤더로 "원본 axial" 여부를 판단한다. 파생(DERIVED)·localizer 영상(MPR 재구성 등)이거나
# 영상 방향이 axial이 아니면 False. 헤더를 못 읽어 확인할 수 없어도 False.
# (예: 2881055의 "Abd With Contrast 2.0 MPR cor"는 IOP가 axial로 기록돼 있어도 ImageType이 DERIVED/LOCALIZER다)
def _is_original_axial(g: dict) -> bool:
    try:
        ds = pydicom.dcmread(g["files"][0], stop_before_pixels=True)
        if {"DERIVED", "LOCALIZER"} & {str(t).upper() for t in ds.ImageType}:
            return False
        iop = [float(x) for x in ds.ImageOrientationPatient]
        return abs(iop[0] * iop[4] - iop[1] * iop[3]) > 0.9   # 슬라이스 법선의 z성분
    except Exception:
        return False


# 시리즈 그룹들 중 best 1개를 고른다. (group, phase) 반환. 후보가 없으면 (None, "").
# 원본 axial이 아닌 시리즈는 후보에서 뺀다 (원본 axial이 하나도 없으면 None → 환자를 건너뜀).
def _pick_best_group(groups: dict):
    sel = _sel()
    best, best_key = None, None
    for g in groups.values():
        if not _is_original_axial(g):
            continue
        phase = sel.classify_phase(g["desc"])
        z_range = round(max(g["z"]) - min(g["z"]), 2) if len(g["z"]) >= 2 else 0.0
        key = (not z_range >= sel.MIN_RANGE_MM, sel.PHASE_RANK[phase], -z_range,
               g["thick"] if g["thick"] is not None else 99.0)
        if best_key is None or key < best_key:
            best, best_key, best_phase = g, key, phase
    return (best, best_phase) if best is not None else (None, "")


# _prognosis_one()을 감싸 한 환자의 예외가 전체 실행을 멈추지 않게 한다(오류는 status에 기록).
def _prognosis_safe(args: tuple) -> tuple:
    try:
        return _prognosis_one(args)
    except Exception as e:
        return args[0], "", "", f"error:{type(e).__name__}:{str(e)[:120]}"


# 환자 1명: npy가 없으면 best series를 골라 저장한다. (pid, 선정 desc, phase, status) 반환. dry면 저장 없이 선정만.
def _prognosis_one(args: tuple) -> tuple:
    pid, patient_dir, dry = args
    out_dir = rf"{NPY_ROOT}\예후데이터\{pid}"
    if any(glob.glob(rf"{glob.escape(d)}\{pid}_*.npy") for d in (out_dir, patient_dir)):
        return pid, "", "", "skip"
    g, phase = _pick_best_group(_scan_groups(patient_dir))
    if g is None:
        return pid, "", "", "no_original_axial"   # 원본 axial 시리즈가 없으면 저장하지 않는다
    if dry:
        return pid, g["desc"], phase, f"dry n={len(g['files'])}"
    slices = []
    for fp in g["files"]:
        try:
            ds = pydicom.dcmread(fp)
            slices.append((int(ds.InstanceNumber), ds))
        except Exception:
            continue
    tag = re.sub(r"[^\w-]", "", (g["desc"].split() or ["series"])[0]) or "series"
    return pid, g["desc"], phase, save_slices(slices, rf"{out_dir}\{pid}_{tag}", g["desc"])


def run_prognosis(dry: int = 0):
    master = pd.read_excel(MASTER_XLSX, sheet_name="Patient_Master", usecols=["patient_id"]).patient_id.astype(int)
    dirs = _sel().build_patient_dir_map(_sel().DICOM_BASE)
    jobs = [(int(p), dirs[int(p)], bool(dry)) for p in sorted(set(master), reverse=True) if int(p) in dirs]
    print(f"Patient_Master {master.nunique()}명 | DICOM 폴더 있음 {len(jobs)}명 (npy 있으면 건너뜀)")
    if dry:
        jobs = jobs[:dry]
    with ProcessPoolExecutor(PROG_WORKERS) as ex:
        res = list(tqdm(ex.map(_prognosis_safe, jobs), total=len(jobs), desc="best series"))
    st = pd.DataFrame(res, columns=["PatientID", "series_desc", "phase", "status"])
    if dry:
        print(st.to_string())
        return
    st.to_csv(PROG_STATUS_CSV, index=False, encoding="utf-8-sig")
    print(st["status"].value_counts().to_dict(), st["phase"].value_counts().to_dict())


if __name__ == "__main__":
    if "--prognosis" in sys.argv:
        run_prognosis(int(sys.argv[sys.argv.index("--dry") + 1]) if "--dry" in sys.argv else 0)
    elif "--missing" in sys.argv:
        for s in SITES:
            run_missing(s) if "--run" in sys.argv else scan_missing(s)
    else:
        for s in (sys.argv[1:] or SITES):
            run_site(s)
