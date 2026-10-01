"""
DICOM에서 AEC(XRayTubeCurrent)만 직접 뽑아 저장한다. 세그멘테이션/체성분 계산 없음.

출력: {site}/landmark/{site}_landmark_aec.xlsx
  - 시트 aec_total    : 환자당 1행, 전체 슬라이스 AEC (aec_1 ... aec_n)
  - 시트 aec_landmark : 환자당 1행, {site}_landmarks.xlsx의 anchor별 instance/k/배열위치/AEC
                        + 랜드마크 신뢰도 검증 컬럼
                        {anchor}_k는 body_composition의 anchor_k와 같은 값(같은 물리 슬라이스).
                        {anchor}_idx는 InstanceNumber 오름차순 배열 위치로, k와 축 방향이
                        반대일 수 있다 — 두 파일을 맞출 때는 instance나 k로 join한다.
  - 시트 qc_summary   : 검증 결과 집계

[AEC 배열 방향] 기존 aec_total/aec_cropped와 동일 — aec_1 = InstanceNumber가 가장 작은 쪽
  (간/머리 쪽), aec_n = 두덩뼈/발 쪽.

[랜드마크 검증]  landmark별 AEC는 {site}_landmarks.xlsx의 anchor instance를 그대로 쓰므로,
  랜드마크가 틀리면 AEC도 같이 틀린다. 그래서 다음을 환자별로 같이 기록한다:

  1. volume_det / volume_flipped
     sitk가 읽은 볼륨의 direction 행렬식. det=-1(좌수좌표계)이면 sitk가 쓴 NIfTI affine의
     k축이 실제 DICOM의 z진행과 반대(axcodes가 S가 아니라 I)로 기록돼, TotalSegmentator가
     환자를 머리-발 뒤집힌 상태로 보게 된다. 이 경우 척추 번호 라벨이 뒤죽박죽이 된다.
     (강남 기준 Revolution CT 490명 중 454명(92.7%)이 여기 해당. 타 기종은 0.4~4.8%.)

  2. anchor_order_ok
     척추 anchor(S1~T10)의 k가 아래→위로 단조증가하는지. liver_dome은 해부학적으로 T10
     아래에 올 수 있어(정상) 순서 검사에서 제외하고 liver_below_T10으로 따로 기록한다.

  3. series_match
     이 스크립트가 고른 series가 landmarks.xlsx를 만들 때 쓴 series와 같은지
     (series_description + n_slices 일치). 다르면 instance 번호가 어긋난다.

  4. {anchor}_relpos / {anchor}_relpos_z
     anchor의 스캔 내 상대위치 (0=가장 cranial 슬라이스, 1=가장 caudal). 이를 신뢰 가능한
     환자군의 중앙값과 비교한 robust z-score(MAD 기준)가 relpos_z. |z| > 3.5면
     다른 환자 대비 확연히 어긋난 랜드마크로 보고 {anchor}_outlier에 표시한다.

  landmark_trustworthy = (not volume_flipped) and anchor_order_ok and series_match
"""

import os
import sys

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_liver_pubis_aec import (
    DATA_DIR, SITES, _aec_values, _is_excluded_signal, find_series_folder, read_slices,
)
from check_landmarks import ALL_ANCHORS, CORE_ANCHORS, pick_series_folder

# 아래(caudal) → 위(cranial) 순서. liver_dome은 T10 아래에 올 수 있어 순서 검사에서 뺀다.
SPINE_ORDER = [a for a in CORE_ANCHORS if a != "liver_dome"]
OUTLIER_Z = 3.5   # robust z-score 임계값 (|z| > 이 값이면 다른 환자 대비 이상)
SAVE_EVERY = 100  # 몇 명마다 중간 저장할지


def site_paths(site: str) -> dict:
    return {
        "dicom_base": rf"E:\영상제공\{site}\{site}_axial",
        "dlo_path":   rf"{DATA_DIR}\{site}\metadata\{site}_DLO_Results.xlsx",
        "landmarks":  rf"{DATA_DIR}\{site}\landmark\{site}_landmarks.xlsx",
        "out":        rf"{DATA_DIR}\{site}\landmark\{site}_landmark_aec.xlsx",
    }


# 볼륨 direction 행렬식을 슬라이스 1장만 읽어 구한다(-1이면 NIfTI 상하 반전 → 랜드마크 신뢰 불가).
def _volume_det(series_dir: str) -> float:
    try:
        reader = sitk.ImageSeriesReader()
        ids = reader.GetGDCMSeriesIDs(series_dir)
        if not ids:
            return np.nan
        one = sitk.ReadImage(reader.GetGDCMSeriesFileNames(series_dir, ids[0])[0])
        return float(np.linalg.det(np.array(one.GetDirection()).reshape(3, 3)))
    except Exception:
        return np.nan


# 이 환자에게 쓸 series 폴더를 landmarks.xlsx와 같은 규칙(DLO 힌트 → phase 우선순위)으로 고른다.
def _pick_series(pid: int, patient_dir: str, series_hint: dict) -> str | None:
    subs = [s for s in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, s))]
    if not subs:
        return None
    hint = series_hint.get(pid)
    folder = find_series_folder(patient_dir, hint) if hint is not None else None
    return folder or pick_series_folder(pid, patient_dir, subs)


# 척추 anchor의 k가 caudal→cranial로 단조증가하는지 검사한다.
def _spine_order_ok(lm: pd.Series | None) -> bool | float:
    if lm is None:
        return np.nan
    ks = [lm[f"{a}_k"] for a in SPINE_ORDER if lm.get(f"{a}_status") == "ok"]
    if len(ks) < 2:
        return np.nan
    return all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))


# 환자 1명: 전체 AEC 배열 + anchor별 AEC/검증 정보를 만든다.
def process_patient(pid: int, series_dir: str, folder: str, lm: pd.Series | None) -> tuple[dict, dict]:
    rows = read_slices(series_dir)          # InstanceNumber 오름차순
    if not rows:
        return ({"PatientID": pid, "status": "no_slices"},
                {"PatientID": pid, "status": "no_slices"})

    try:
        hdr = pydicom.dcmread(rows[0]["file"], stop_before_pixels=True)
        series_desc = str(getattr(hdr, "SeriesDescription", ""))
        manufacturer = str(getattr(hdr, "ManufacturerModelName", ""))
        kvp = getattr(hdr, "KVP", None)
    except Exception:
        series_desc = manufacturer = ""
        kvp = None

    aec = [r["aec"] for r in rows]
    n = len(rows)
    # 전량 NaN / CV<0.05 / range<10mA / 거의 직선이면 AEC 변조가 없는 스캔(고정 mA) — 행은
    # 남기되 플래그로 표시한다(기존 aec_total은 이런 환자를 아예 제외했음).
    total = {"PatientID": pid, "status": "ok", "series_desc": series_desc,
             "manufacturer": manufacturer, "kvp": float(kvp) if kvp not in (None, "") else np.nan,
             "n_slices": n, "aec_flat": bool(_is_excluded_signal(_aec_values(rows))),
             "inst_offset": int(rows[0]["inst"]) - 1, "aec_full": aec}

    det = _volume_det(series_dir)
    # landmarks.xlsx를 만들 때와 같은 series인지 — 다르면 instance 번호가 어긋난다.
    if lm is None:
        series_match = np.nan
    else:
        series_match = bool(int(lm.get("n_slices", -1) or -1) == n)

    mark = {"PatientID": pid, "status": "ok", "series_desc": series_desc,
            "manufacturer": manufacturer, "n_slices": n,
            "volume_det": det, "volume_flipped": bool(det < 0) if not np.isnan(det) else np.nan,
            "series_match": series_match,
            "anchor_order_ok": _spine_order_ok(lm),
            "liver_below_T10": (bool(lm["liver_dome_k"] <= lm["T10_center_k"])
                                if lm is not None and lm.get("liver_dome_status") == "ok"
                                and lm.get("T10_center_status") == "ok" else np.nan)}

    _fill_anchors(mark, lm, aec, {r["inst"]: i for i, r in enumerate(rows)}.get)
    return total, mark


# anchor별 instance/배열위치/AEC/상대위치를 mark에 채운다. idx_of: instance → 0-based 배열위치.
def _fill_anchors(mark: dict, lm: pd.Series | None, aec: list, idx_of) -> None:
    n = len(aec)
    for anchor in ALL_ANCHORS:
        inst = lm.get(f"{anchor}_instance") if lm is not None else None
        if inst is None or pd.isna(inst):
            mark[f"{anchor}_instance"] = np.nan
            mark[f"{anchor}_k"] = np.nan
            mark[f"{anchor}_idx"] = pd.NA
            mark[f"{anchor}_aec"] = np.nan
            mark[f"{anchor}_relpos"] = np.nan
            continue
        i = idx_of(int(inst))
        mark[f"{anchor}_instance"] = int(inst)
        # landmarks.xlsx의 k를 그대로 싣는다 — body_composition의 anchor_k와 같은 값이라
        # 두 산출물을 (PatientID, anchor)로 바로 대조할 수 있다. aec_idx는 InstanceNumber
        # 오름차순 배열 위치라서 k와 축 방향이 반대일 수 있다(같은 슬라이스, 다른 번호).
        k = lm.get(f"{anchor}_k")
        mark[f"{anchor}_k"] = int(k) if pd.notna(k) else np.nan
        mark[f"{anchor}_idx"] = (i + 1) if i is not None else pd.NA   # aec_{idx} 컬럼과 대응
        mark[f"{anchor}_aec"] = aec[i] if i is not None else np.nan
        mark[f"{anchor}_relpos"] = (i / (n - 1)) if i is not None and n > 1 else np.nan

    mark["n_anchor_found"] = sum(1 for a in ALL_ANCHORS if not pd.isna(mark[f"{a}_instance"]))
    mark["landmark_trustworthy"] = bool(
        mark["volume_flipped"] is not True
        and mark["anchor_order_ok"] is True
        and mark["series_match"] is not False)


# 신뢰 가능한 환자군의 anchor 상대위치 분포와 비교해 환자별 이상치를 표시한다.
def _add_relpos_outliers(mark_df: pd.DataFrame) -> pd.DataFrame:
    ref = mark_df[mark_df["landmark_trustworthy"] == True]  # noqa: E712
    if len(ref) < 20:
        ref = mark_df
    for anchor in ALL_ANCHORS:
        col = f"{anchor}_relpos"
        vals = ref[col].dropna()
        if len(vals) < 20:
            mark_df[f"{anchor}_relpos_z"] = np.nan
            mark_df[f"{anchor}_outlier"] = pd.NA
            continue
        med = float(vals.median())
        mad = float((vals - med).abs().median())
        scale = mad * 1.4826 if mad > 0 else float(vals.std())
        z = (mark_df[col] - med) / scale if scale > 0 else mark_df[col] * np.nan
        mark_df[f"{anchor}_relpos_z"] = z.round(2)
        mark_df[f"{anchor}_outlier"] = z.abs() > OUTLIER_Z
    out_cols = [f"{a}_outlier" for a in ALL_ANCHORS]
    mark_df["n_outlier_anchor"] = mark_df[out_cols].fillna(False).sum(axis=1).astype(int)
    return mark_df


# anchor별 컬럼을 anchor 단위로 묶어 읽기 좋게 정렬한다.
def _order_mark_cols(df: pd.DataFrame) -> list[str]:
    head = ["PatientID", "status", "series_desc", "manufacturer", "n_slices",
            "landmark_trustworthy", "volume_flipped", "volume_det", "anchor_order_ok",
            "series_match", "liver_below_T10", "n_anchor_found", "n_outlier_anchor"]
    body = []
    for a in ALL_ANCHORS:
        body += [f"{a}_instance", f"{a}_k", f"{a}_idx", f"{a}_aec",
                 f"{a}_relpos", f"{a}_relpos_z", f"{a}_outlier"]
    return [c for c in head + body if c in df.columns]


# aec_1..aec_n 컬럼만 고른다 (aec_flat은 메타 컬럼이라 제외).
def _aec_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("aec_") and c[4:].isdigit()]


# aec_total 시트를 process_patient가 만드는 dict 형태(aec_full 리스트)로 되돌린다.
def _load_totals(out_path: str) -> list[dict]:
    prev = pd.read_excel(out_path, sheet_name="aec_total")
    cols = _aec_cols(prev)
    rows = []
    for _, r in prev.iterrows():
        d = {c: r[c] for c in prev.columns if c not in cols}
        d["aec_full"] = [v for v in r[cols].tolist() if pd.notna(v)]
        rows.append(d)
    return rows


# DICOM을 다시 읽지 않고 갱신된 landmarks.xlsx만 반영해 aec_landmark 시트를 다시 만든다.
# instance 번호는 환자 내에서 연속이므로(검증 완료) 배열위치 = instance - inst_offset - 1.
def relink(site: str) -> None:
    paths = site_paths(site)
    total_rows = _load_totals(paths["out"])
    prev_m = pd.read_excel(paths["out"], sheet_name="aec_landmark").set_index("PatientID")
    lm_df = pd.read_excel(paths["landmarks"])
    lm_by_pid = {int(r["PatientID"]): r for _, r in lm_df.iterrows()}
    print(f"[{site}] relink: aec_total {len(total_rows)}명 × landmarks {len(lm_by_pid)}명")

    mark_rows, skipped = [], 0
    for t in total_rows:
        pid = int(t["PatientID"])
        prev = prev_m.loc[pid] if pid in prev_m.index else None
        off = t.get("inst_offset")
        if t.get("status") != "ok" or prev is None or pd.isna(off):
            # inst_offset이 없는(구버전) 행은 DICOM 없이 재계산할 수 없어 기존 행을 유지한다.
            mark_rows.append({"PatientID": pid, **(dict(prev) if prev is not None else
                                                   {"status": t.get("status", "no_prev")})})
            skipped += 1
            continue
        off, n, aec = int(off), int(t["n_slices"]), t["aec_full"]
        lm = lm_by_pid.get(pid)
        mark = {"PatientID": pid, "status": "ok", "series_desc": t.get("series_desc"),
                "manufacturer": t.get("manufacturer"), "n_slices": n,
                "volume_det": prev.get("volume_det"), "volume_flipped": prev.get("volume_flipped"),
                "series_match": (bool(int(lm.get("n_slices", -1) or -1) == n)
                                 if lm is not None else np.nan),
                "anchor_order_ok": _spine_order_ok(lm),
                "liver_below_T10": (bool(lm["liver_dome_k"] <= lm["T10_center_k"])
                                    if lm is not None and lm.get("liver_dome_status") == "ok"
                                    and lm.get("T10_center_status") == "ok" else np.nan)}
        def idx_of(inst: int, off=off, n=n) -> int | None:
            i = inst - off - 1
            return i if 0 <= i < n else None

        _fill_anchors(mark, lm, aec, idx_of)
        mark_rows.append(mark)

    _write(paths, total_rows, mark_rows)
    print(f"[{site}] relink 완료 (재계산 {len(mark_rows) - skipped}명, 기존 유지 {skipped}명) "
          f"→ {paths['out']}")


def _write(paths: dict, total_rows: list[dict], mark_rows: list[dict]) -> None:
    if not total_rows:
        return
    max_len = max(len(r.get("aec_full", [])) for r in total_rows)
    meta = ["PatientID", "status", "series_desc", "manufacturer", "kvp", "n_slices",
            "aec_flat", "inst_offset"]
    out = []
    for r in total_rows:
        row = {c: r.get(c) for c in meta}
        for i, v in enumerate(r.get("aec_full", [])):
            row[f"aec_{i + 1}"] = v
        out.append(row)
    total_df = pd.DataFrame(out, columns=meta + [f"aec_{i + 1}" for i in range(max_len)])
    total_df = total_df.sort_values("PatientID").reset_index(drop=True)

    mark_df = _add_relpos_outliers(pd.DataFrame(mark_rows).sort_values("PatientID").reset_index(drop=True))
    mark_df = mark_df[_order_mark_cols(mark_df)]

    ok = mark_df[mark_df["status"] == "ok"]
    qc = pd.DataFrame([
        {"항목": "전체 환자", "n": len(mark_df)},
        {"항목": "AEC 추출 성공", "n": int((mark_df["status"] == "ok").sum())},
        {"항목": "landmark 신뢰 가능", "n": int((ok["landmark_trustworthy"] == True).sum())},  # noqa: E712
        {"항목": "볼륨 상하반전(det=-1)", "n": int((ok["volume_flipped"] == True).sum())},  # noqa: E712
        {"항목": "척추 anchor 순서 위반", "n": int((ok["anchor_order_ok"] == False).sum())},  # noqa: E712
        {"항목": "series 불일치", "n": int((ok["series_match"] == False).sum())},  # noqa: E712
        {"항목": "relpos 이상치 anchor 1개 이상", "n": int((ok["n_outlier_anchor"] > 0).sum())},
        {"항목": "AEC 변조 없음(고정 mA)", "n": int(total_df["aec_flat"].fillna(False).sum())},
    ])
    by_model = (ok.groupby("manufacturer")
                  .agg(n=("PatientID", "size"),
                       flipped=("volume_flipped", lambda s: int((s == True).sum())),  # noqa: E712
                       order_bad=("anchor_order_ok", lambda s: int((s == False).sum())),  # noqa: E712
                       trustworthy=("landmark_trustworthy", lambda s: int((s == True).sum())))  # noqa: E712
                  .reset_index().sort_values("n", ascending=False))

    os.makedirs(os.path.dirname(paths["out"]), exist_ok=True)
    tmp = paths["out"] + ".tmp.xlsx"
    with pd.ExcelWriter(tmp, engine="openpyxl") as w:
        total_df.to_excel(w, sheet_name="aec_total", index=False)
        mark_df.to_excel(w, sheet_name="aec_landmark", index=False)
        qc.to_excel(w, sheet_name="qc_summary", index=False)
        by_model.to_excel(w, sheet_name="qc_summary", index=False, startrow=len(qc) + 3)
    os.replace(tmp, paths["out"])


def run_site(site: str, limit: int | None = None, seed: int = 0) -> None:
    paths = site_paths(site)
    print("=" * 78)
    print(f"[{site}] AEC 추출 시작  |  DICOM: {paths['dicom_base']}")

    series_hint: dict = {}
    if os.path.exists(paths["dlo_path"]):
        dlo = pd.read_excel(paths["dlo_path"])[["PatientID", "Series_Desc"]]
        series_hint = dict(zip(dlo["PatientID"].astype(int), dlo["Series_Desc"]))

    lm_by_pid: dict = {}
    if os.path.exists(paths["landmarks"]):
        lm_df = pd.read_excel(paths["landmarks"])
        lm_by_pid = {int(r["PatientID"]): r for _, r in lm_df.iterrows()}
        print(f"[{site}] landmarks.xlsx {len(lm_by_pid)}명 로드")
    else:
        print(f"[{site}] landmarks.xlsx 없음 — aec_landmark 시트는 비어서 나갑니다")

    all_pids = sorted(int(f) for f in os.listdir(paths["dicom_base"])
                      if f.isdigit() and os.path.isdir(os.path.join(paths["dicom_base"], f)))

    # 이미 끝난 환자는 건너뛰고 이어서 처리 (재실행 안전)
    total_rows, mark_rows, done = [], [], set()
    if os.path.exists(paths["out"]):
        total_rows = _load_totals(paths["out"])
        mark_rows = pd.read_excel(paths["out"], sheet_name="aec_landmark").to_dict("records")
        done = {int(r["PatientID"]) for r in total_rows}
        print(f"[{site}] 기존 결과 {len(done)}명 — 건너뜀")

    pending = [p for p in all_pids if p not in done]
    if limit is not None:
        # 샘플 모드: 무작위 N명만 처리하고 본 출력 파일과 분리된 _sample 파일에 쓴다.
        rng = np.random.default_rng(seed)
        pending = sorted(rng.choice(all_pids, size=min(limit, len(all_pids)), replace=False).tolist())
        paths["out"] = paths["out"].replace(".xlsx", "_sample.xlsx")
        total_rows, mark_rows = [], []
        print(f"[{site}] 샘플 모드 {len(pending)}명 (seed={seed}) → {os.path.basename(paths['out'])}")
    print(f"[{site}] 대상 {len(all_pids)}명  |  처리 예정 {len(pending)}명")

    for i, pid in enumerate(tqdm(pending, desc=f"{site} AEC"), 1):
        patient_dir = os.path.join(paths["dicom_base"], str(pid))
        try:
            folder = _pick_series(pid, patient_dir, series_hint)
            if folder is None:
                t = m = {"PatientID": pid, "status": "no_series"}
            else:
                t, m = process_patient(pid, os.path.join(patient_dir, folder),
                                       folder, lm_by_pid.get(pid))
        except Exception as e:
            t = m = {"PatientID": pid, "status": f"error:{type(e).__name__}:{e}"}
        total_rows.append(t)
        mark_rows.append(m)
        if i % SAVE_EVERY == 0 or i == len(pending):
            _write(paths, total_rows, mark_rows)
            tqdm.write(f"  [{site} 저장 {i}/{len(pending)}]")

    if not pending:
        _write(paths, total_rows, mark_rows)
    print(f"[{site}] 완료 → {paths['out']}")


def _self_test() -> None:
    """검증 로직만 빠르게 확인 (DICOM 불필요)."""
    good = pd.Series({f"{a}_k": k for a, k in zip(SPINE_ORDER, range(10, 110, 10))}
                     | {f"{a}_status": "ok" for a in SPINE_ORDER})
    assert _spine_order_ok(good) is True
    bad = good.copy()
    bad["L3_center_k"] = 999                      # L3만 위로 튀면 순서 위반
    assert _spine_order_ok(bad) is False
    assert np.isnan(_spine_order_ok(None))

    # aec_flat이 aec_1..n에 섞이면 재개 시 AEC가 한 칸씩 밀린다 — 컬럼 선택 확인
    t = pd.DataFrame([{"PatientID": 1, "aec_flat": False, "aec_1": 10, "aec_2": 20}])
    assert _aec_cols(t) == ["aec_1", "aec_2"], _aec_cols(t)

    # relink 인덱스: 배열위치 = instance - inst_offset - 1 (instance는 환자 내 연속)
    mk: dict = {"volume_flipped": False, "anchor_order_ok": True, "series_match": True}
    _fill_anchors(mk, pd.Series({"L3_center_instance": 53, "L3_center_k": 90,
                                 "L3_center_status": "ok"}),
                  [0, 1, 2, 3, 4], lambda inst, off=50, n=5: (
                      (inst - off - 1) if 0 <= inst - off - 1 < n else None))
    assert mk["L3_center_idx"] == 3 and mk["L3_center_aec"] == 2, mk
    assert mk["L3_center_k"] == 90, mk          # landmarks의 k가 그대로 실려야 한다

    df = pd.DataFrame({"PatientID": range(60), "landmark_trustworthy": [True] * 60})
    for a in ALL_ANCHORS:
        df[f"{a}_relpos"] = [0.5] * 59 + [0.99]   # 마지막 1명만 동떨어짐
    df = _add_relpos_outliers(df)
    assert df["L3_center_outlier"].iloc[-1] == True, "이상치 미검출"   # noqa: E712
    assert df["L3_center_outlier"].iloc[0] == False                     # noqa: E712
    assert df["n_outlier_anchor"].iloc[-1] == len(ALL_ANCHORS)
    print("self-test OK")


def main() -> None:
    if "--self-test" in sys.argv:
        _self_test()
        return
    sites = SITES
    if "--site" in sys.argv:
        sites = [sys.argv[sys.argv.index("--site") + 1]]
    if "--relink" in sys.argv:
        for site in sites:
            relink(site)
        return
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    for site in sites:
        run_site(site, limit=limit, seed=seed)


if __name__ == "__main__":
    main()
