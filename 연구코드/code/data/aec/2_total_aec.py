"""
DICOM_BASE 아래 모든 사이트·환자의 AEC(XRayTubeCurrent) 값을 읽어 사이트별 xlsx로 저장한다.
강남은 Excel(kVp_100_조영제X_중복제거 시트)의 PatientID·Series_Desc를 기반으로 폴더를 특정한다.

[제외 기준]
  1. AEC 값이 모두 NaN
  2. CV < AEC_CV_MIN (0.05)
  3. range < RANGE_MIN (10 mA)
  4. R² ≥ R2_THRESHOLD (0.95) → 거의 직선 신호

[출력] {OUT_DIR}/{site}/aec/{site}_aec_total.xlsx
"""

import os
import pickle
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

# ── 설정 ──────────────────────────────────────────────────────────────────────

SITE = "강남"   # "강남" 또는 "신촌"

DICOM_BASE      = rf"D:/영상제공/{SITE}/{SITE}_axial"
OUT_DIR         = r"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data"
CHECKPOINT_PATH = os.path.join(OUT_DIR, f"aec_total_checkpoint_{SITE}.pkl")

BATCH_SIZE    = 20
AEC_CV_MIN    = 0.05
RANGE_MIN     = 10
R2_THRESHOLD  = 0.95

TEST_N = None   # 정수 지정 시 앞 N명만 처리

# ── 사이트별 Excel 기반 설정 ───────────────────────────────────────────────────
SITE_CONFIGS = {
    "강남": {
        "excel_path":  r"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\강남\metadata\강남_DLO_Results_SMI_kVp100.xlsx",
        "excel_sheet": "kVp_100",
    },
    "신촌": {
        "excel_path":  r"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\신촌\metadata\신촌_DLO_Results_SMI_kVp100.xlsx",
        "excel_sheet": "kVp_100",
    },
}


# ── 폴더 탐색 ─────────────────────────────────────────────────────────────────

def _build_folder_map_excel(site: str, cfg: dict) -> dict[tuple[str, str], str]:
    """Excel 시트의 PatientID·Series_Desc → 시리즈 폴더 절대경로"""
    df = pd.read_excel(cfg["excel_path"], sheet_name=cfg["excel_sheet"])
    folder_map: dict[tuple[str, str], str] = {}

    def _norm(s: str) -> str:
        return " ".join(s.replace("/", "_").split()).lower()

    for _, row in df.iterrows():
        pid = str(int(row["PatientID"]))
        series_desc = str(row["Series_Desc"]).strip()

        patient_dir = os.path.join(DICOM_BASE, pid)
        if not os.path.isdir(patient_dir):
            print(f"[{site}] 환자 폴더 없음: {patient_dir}")
            continue

        if not series_desc or series_desc == "nan":
            print(f"[{site}] Series_Desc 없음: PID={pid}")
            continue

        series_desc = series_desc.replace("...", "").split("(")[0].strip()
        series_norm = _norm(series_desc)

        target = None
        for sub in os.listdir(patient_dir):
            sub_path = os.path.join(patient_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            sub_norm = _norm(sub)
            if series_norm in sub_norm or sub_norm in series_norm:
                target = sub_path
                break

        if target:
            folder_map[(site, pid)] = target
        else:
            print(f"[{site}] Series_Desc '{series_desc}' 미발견: {patient_dir}")

    return folder_map


def build_folder_map(dicom_base: str) -> dict[tuple[str, str], str]:
    """(site, PatientID) → 시리즈 폴더 절대경로"""
    folder_map: dict[tuple[str, str], str] = {}

    for site_name in [SITE]:
        if site_name in SITE_CONFIGS:
            folder_map.update(_build_folder_map_excel(site_name, SITE_CONFIGS[site_name]))
            continue

        axial_dir = dicom_base
        if not os.path.isdir(axial_dir):
            continue

        for pid in os.listdir(axial_dir):
            patient_dir = os.path.join(axial_dir, pid)
            if not os.path.isdir(patient_dir):
                continue

            # DCM 파일이 가장 많은 하위 폴더 선택
            best, best_count = None, 0
            for sub in os.listdir(patient_dir):
                sub_path = os.path.join(patient_dir, sub)
                if not os.path.isdir(sub_path):
                    continue
                n = sum(1 for f in os.listdir(sub_path) if f.endswith(".dcm"))
                if n > best_count:
                    best_count, best = n, sub_path

            if best:
                folder_map[(site_name, pid)] = best

    return folder_map


# ── DICOM 읽기 ────────────────────────────────────────────────────────────────

def read_patient_aec(series_dir: str) -> tuple[dict, list[dict]] | tuple[None, None]:
    """시리즈 폴더에서 AEC 값과 메타데이터를 읽어 z 오름차순으로 반환."""
    rows = []
    series_desc = manufacturer = None

    for fname in os.listdir(series_dir):
        if not fname.endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, fname), stop_before_pixels=True)
            if series_desc is None:
                series_desc  = str(getattr(ds, "SeriesDescription", ""))
                manufacturer = str(getattr(ds, "ManufacturerModelName", ""))
            ipp  = getattr(ds, "ImagePositionPatient", None)
            z    = float(ipp[2]) if ipp is not None else float("nan")
            aec  = getattr(ds, "XRayTubeCurrent", None) or getattr(ds, "TubeCurrent", None)
            rows.append({
                "inst": int(getattr(ds, "InstanceNumber", 0)),
                "z":    z,
                "aec":  float(aec) if aec is not None else float("nan"),
            })
        except Exception:
            continue

    if not rows:
        return None, None

    df = pd.DataFrame(rows).sort_values("inst").reset_index(drop=True)
    meta = {"series_desc": series_desc, "manufacturer": manufacturer or "", "n_slices": len(df)}
    return meta, df.to_dict("records")


# ── 필터 함수 ─────────────────────────────────────────────────────────────────

def _aec_values(r: dict) -> list[float]:
    return [v for k, v in r.items() if k.startswith("aec_") and isinstance(v, (int, float)) and not np.isnan(v)]


def _aec_cv(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = float(np.mean(vals))
    return float(np.std(vals) / mean) if mean != 0 else 0.0


def _is_linear(vals: list[float]) -> bool:
    if not vals or max(vals) - min(vals) < RANGE_MIN:
        return True
    arr = np.array(vals)
    ss_tot = float(np.sum((arr - arr.mean()) ** 2))
    if ss_tot < 1e-10:
        return True
    r = float(np.corrcoef(np.arange(len(arr), dtype=float), arr)[0, 1])
    return r ** 2 >= R2_THRESHOLD


def _apply_filters(results: list[dict]) -> list[dict]:
    def filter_step(data, pred, label):
        kept = [r for r in data if pred(r)]
        removed = len(data) - len(kept)
        if removed:
            print(f"[{label}] {removed}명 제거 → {len(kept)}명 남음")
        return kept

    results = filter_step(
        results,
        lambda r: _aec_cv(_aec_values(r)) >= AEC_CV_MIN,
        f"CV 필터(CV < {AEC_CV_MIN})"
    )
    results = filter_step(
        results,
        lambda r: not _is_linear(_aec_values(r)),
        f"직선 필터(range<{RANGE_MIN}mA 또는 R²≥{R2_THRESHOLD})"
    )
    return results


# ── 출력 ──────────────────────────────────────────────────────────────────────

def _write_output(results: list[dict]) -> None:
    if not results:
        return

    meta_cols = ["site", "PatientID", "n_slices", "series_desc", "manufacturer"]
    df_all = pd.DataFrame(results)

    for site, group in df_all.groupby("site"):
        aec_cols = [f"aec_{i + 1}" for i in range(int(group["n_slices"].max()))]
        df_out = (group
                  .reindex(columns=meta_cols + aec_cols)
                  .sort_values("PatientID")
                  .reset_index(drop=True))
        df_out["n_slices"] = df_out["n_slices"].astype("Int64")

        out_path = os.path.join(OUT_DIR, str(site), "aec", f"{site}_aec_total.xlsx")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df_out.to_excel(writer, sheet_name="aec_total", index=False)


# ── 체크포인트 ────────────────────────────────────────────────────────────────

def _load_checkpoint() -> tuple[set[tuple[str, str]], list[dict]]:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            data = pickle.load(f)
        print(f"[체크포인트] {len(data['processed'])}명 완료 → 이어서 시작")
        return data["processed"], data["results"]
    return set(), []


def _save_checkpoint(processed: set[tuple[str, str]], results: list[dict]) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({"processed": processed, "results": results}, f)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    folder_map = build_folder_map(DICOM_BASE)
    all_keys   = sorted(folder_map, key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else 0))
    if TEST_N is not None:
        all_keys = all_keys[:TEST_N]

    processed, results = _load_checkpoint()
    remaining = [k for k in all_keys if k not in processed]
    print(f"사이트: {sorted({k[0] for k in all_keys})}  |  대상: {len(all_keys)}명  |  미처리: {len(remaining)}명")

    with tqdm(total=len(all_keys), initial=len(processed), desc="AEC 추출") as pbar:
        for site, pid in remaining:
            try:
                meta, rows = read_patient_aec(folder_map[(site, pid)])

                if meta is None or rows is None:
                    tqdm.write(f"  [SKIP] {site} PID {pid}: DICOM 읽기 실패")
                elif all(np.isnan(r["aec"]) for r in rows):
                    tqdm.write(f"  [SKIP] {site} PID {pid}: AEC 데이터 없음")
                else:
                    row = {"site": site, "PatientID": int(pid) if pid.isdigit() else pid, **meta}
                    for i, r in enumerate(rows):
                        row[f"aec_{i + 1}"] = r["aec"]
                    results.append(row)

            except Exception as e:
                tqdm.write(f"  [ERROR] {site} PID {pid}: {type(e).__name__}: {e}")

            processed.add((site, pid))
            pbar.update(1)

            if len(processed) % BATCH_SIZE == 0:
                _save_checkpoint(processed, results)
                _write_output(results)
                tqdm.write(f"  [체크포인트 저장 | 수집 {len(results)}명]")

    results = _apply_filters(results)
    _write_output(results)

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    print(f"\n[완료] 총 {len(results)}명  |  {OUT_DIR}/{{site}}/aec/{{site}}_aec_total.xlsx")


if __name__ == "__main__":
    main()
