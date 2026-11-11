r"""landmark_report와 조직(VAT/SAT/TAMA) 이미지 도구. 서브커맨드 셋.

  python tissue_reports.py images 신촌 [--limit N] [--reverse] [--shard i --num-shards n]
      {site}_landmark_filtered.xlsx의 PatientID만 대상으로 환자당 4장(_seg/_vat/_sat/_tama.png)을
      {site}/landmark/landmark_report_seg/{PatientID}/ 에 저장한다(이미 4장 있으면 건너뜀 → 재개 가능).
      --limit N: 앞에서 N명만(테스트), --reverse: PatientID 역순, --shard/--num-shards: pids[i::n]만 처리(병렬 실행용).

  python tissue_reports.py from-reports 신촌 [--limit N]
      이미 만들어 둔 landmark_report/{pid}.png의 오버레이 색에서 VAT/SAT/TAMA 마스크와 CT 원본(조직만)을
      역추정해 {site}/landmark/landmark_report_seg_fromreport/{pid}/{pid}_{seg,vat,sat,tama}.png 로 저장한다(재세그멘테이션 없음).
      오버레이는 0.55*gray + 0.45*색 이므로 색 성분에서 조직 종류, 색이 0인 채널에서 gray가 복원된다.
      한계: 리포트 패널 해상도(~400px, 원본 512)로 저장돼 있어 마스크 경계는 근사값이다.

  python tissue_reports.py merge-tama [--site 강남] [--limit N]
      기존 landmark_report PNG의 NAMA(파랑)/LAMA(청록)/IMATA(초록) 오버레이를 TAMA 한 색(파랑)으로 통일하고
      범례를 SAT/VAT/TAMA 3개로 바꾼다(세그멘테이션 재실행 없음, 여러 번 실행해도 결과 같음).
      오버레이는 0.55*gray + 0.45*색 이므로 R 채널(색 R=0)에서 gray를 복원해 TAMA 색(0,0.4,1)으로 다시 섞는다.
      원본 백업 없이 제자리에서 덮어쓴다.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")  # 동시 실행 프로세스끼리 CPU 경합을 줄이려 프로세스당 스레드 제한(torch import 전)
os.environ.setdefault("MKL_NUM_THREADS", "3")
import argparse
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA_DIR = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data"
WL, WW, THR = 40, 400, 0.22        # from-reports: 창(window) 수준/폭, 색 성분 임계값
TAMA_THR = 0.1                     # merge-tama: 이 이상 R보다 G/B가 크면 청록·초록·파랑 계열(TAMA)로 본다
LEGEND_Y = (58, 82)                # merge-tama: 범례가 있는 행 구간(제목 PatientID 아래, 패널 제목 위)
PANEL_Y0 = 120                     # merge-tama: 이 아래부터 패널 영역


def _cl():
    """무거운 check_landmarks(torch 등)는 필요한 서브커맨드에서만 불러온다."""
    import check_landmarks as cl
    return cl


# ---- images: 세그멘테이션을 다시 돌려 환자당 4장 저장 ----

def run_images(site: str, argv: list[str]) -> None:
    cl = _cl()
    arg = lambda k, d: int(argv[argv.index(k) + 1]) if k in argv else d
    limit = arg("--limit", None)
    cl._save_landmark_report = lambda *a, **k: None   # 기존 landmark_report는 건드리지 않는다
    paths = cl.site_paths(site)
    pids = sorted(pd.read_excel(rf"{cl.DATA_DIR}\{site}\landmark\{cl.SITE_SLUG[site]}_landmark_filtered.xlsx",
                                sheet_name="patient_metadata")["PatientID"].astype(int), reverse="--reverse" in argv)[:limit]
    out_dir = paths["report_dir"] + "_seg"
    dlo = pd.read_excel(paths["dlo_path"])[["PatientID", "Series_Desc"]]
    hint = dict(zip(dlo["PatientID"].astype(int), dlo["Series_Desc"]))

    def series_dir(pid):
        pd_ = os.path.join(paths["dicom_base"], str(pid))
        subs = [s for s in os.listdir(pd_) if os.path.isdir(os.path.join(pd_, s))]
        if not subs:
            return None
        f = subs[0] if len(subs) == 1 else cl.pick_series_folder(pid, pd_, subs)
        if f is None and hint.get(pid) is not None:
            f = cl.find_series_folder(pd_, hint[pid])
        return os.path.join(pd_, f) if f else None

    def done(pid):
        return all(os.path.exists(os.path.join(out_dir, str(pid), f"{pid}_{k}.png"))
                   for k in ("seg", "vat", "sat", "tama"))

    i, n = arg("--shard", 0), arg("--num-shards", 1)   # 병렬 실행: pids[i::n]만 처리(샤드끼리 겹치지 않음)
    todo = [p for p in pids[i::n] if not done(p)]
    print(f"[{site}] filtered {len(pids)}명, 남은 {len(todo)}명 → {out_dir}")
    with tempfile.TemporaryDirectory() as tmp:
        for pid in tqdm(todo, desc=site):
            try:
                sd = series_dir(pid)
                if sd is None:
                    tqdm.write(f"  PID {pid}: series 없음")
                    continue
                cl.process_patient(pid, sd, tmp, report_path=os.path.join(paths["report_dir"], f"{pid}.png"),
                                   want_total=lambda *a: True)
                if not done(pid):
                    tqdm.write(f"  PID {pid}: 이미지 미생성(세그멘테이션 실패)")
            except Exception as e:
                tqdm.write(f"  PID {pid}: {type(e).__name__}: {e}")


# ---- from-reports: 리포트 PNG의 오버레이 색에서 조직 이미지를 역추정 ----

def _extract(cl, report_png: str) -> dict[str, dict]:
    """리포트 PNG → {anchor: {vat_mask, sat_mask, nama_mask(=TAMA), lama_mask, imata_mask, hu_slice}}"""
    from scipy import ndimage
    im = np.asarray(Image.open(report_png).convert("RGB")).astype(float) / 255
    H, W, _ = im.shape
    nonwhite = im.min(axis=2) < 0.97
    # 글자(가는 선)를 지우고 큰 CT 패널만 남긴다
    panels = ndimage.binary_opening(nonwhite, structure=np.ones((25, 25)))
    lab, n = ndimage.label(panels)
    ncols = 4
    nrows = -(-len(cl.ANCHOR_ORDER) // ncols)
    out = {}
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        ys, xs = sl
        if (ys.stop - ys.start) < 100 or (xs.stop - xs.start) < 100:
            continue
        cy, cx = (ys.start + ys.stop) / 2, (xs.start + xs.stop) / 2
        slot = min(int(cy / (H / nrows)), nrows - 1) * ncols + min(int(cx / (W / ncols)), ncols - 1)
        if slot >= len(cl.ANCHOR_ORDER):
            continue
        p = im[sl]
        R, G, B = p[..., 0], p[..., 1], p[..., 2]
        sat = (R - B > THR) & (R - G <= THR)
        vat = (R - G > THR) & (R - B > THR) & ~sat
        tama = (G - R > THR) | (B - R > THR)
        # 마스크 색 성분이 0인 채널로 gray 복원: SAT→B, VAT→G, TAMA→R, 그 외는 그대로
        g = np.where(sat, B / 0.55, np.where(vat, G / 0.55, np.where(tama, R / 0.55, R)))
        g = np.clip(g, 0, 1)
        # _save_tissue_images는 (x, y) 배열을 받아 .T로 그리므로 리포트(이미 표시 방향)를 .T로 넘긴다
        out[cl.ANCHOR_ORDER[slot]] = {
            "seg_status": "ok", "hu_slice": (g * WW + (WL - WW / 2)).T,
            "vat_mask": vat.T, "sat_mask": sat.T, "nama_mask": tama.T,
            "lama_mask": np.zeros_like(tama.T), "imata_mask": np.zeros_like(tama.T)}
    return out


def run_from_reports(site: str, argv: list[str]) -> None:
    cl = _cl()
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    paths = cl.site_paths(site)
    pids = sorted(pd.read_excel(rf"{cl.DATA_DIR}\{site}\landmark\{cl.SITE_SLUG[site]}_landmark_filtered.xlsx",
                                sheet_name="patient_metadata")["PatientID"].astype(int))[:limit]
    out_dir = paths["report_dir"] + "_seg_fromreport"   # 정확본(_seg)과 섞이지 않게 분리
    for pid in tqdm(pids, desc=site):
        rp = os.path.join(paths["report_dir"], f"{pid}.png")
        if not os.path.exists(rp):
            tqdm.write(f"  PID {pid}: 리포트 없음")
            continue
        try:
            cl._save_tissue_images(pid, _extract(cl, rp), out_dir)
        except Exception as e:
            tqdm.write(f"  PID {pid}: {type(e).__name__}: {e}")


# ---- merge-tama: 리포트의 NAMA/LAMA/IMATA 색을 TAMA 한 색으로 통일 ----

def _legend_img() -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(8, 1.0), dpi=110)
    fig.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                        (("SAT", (1, 1, 0)), ("VAT", (1, 0, 0)), ("TAMA", (0, 0.4, 1)))],
               loc="center", ncol=3, frameon=False, fontsize=11, handlelength=1.4, columnspacing=1.8)
    buf = io.BytesIO(); fig.savefig(buf, dpi=110, format="png"); plt.close(fig)
    a = np.array(Image.open(buf).convert("RGB"))
    ys, xs = np.where((a < 250).any(2))
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _convert(img: np.ndarray, legend: np.ndarray) -> np.ndarray:
    out = img.copy()
    p = img[PANEL_Y0:].astype(float) / 255
    R, G, B = p[..., 0], p[..., 1], p[..., 2]
    tama = (G - R > TAMA_THR) | (B - R > TAMA_THR)
    g = R[tama]                          # = 0.55 * gray (색의 R 성분이 0)
    new = np.stack([g, g + 0.18, g + 0.45], axis=-1)
    blk = out[PANEL_Y0:]
    blk[tama] = np.clip(new * 255 + 0.5, 0, 255).astype(np.uint8)
    # 범례: 기존 영역을 지우고 3항목 범례를 가운데에 다시 쓴다
    y0, y1 = LEGEND_Y
    out[y0:y1] = 255
    h, w = legend.shape[:2]
    yt = (y0 + y1) // 2 - h // 2
    xl = img.shape[1] // 2 - w // 2
    out[yt:yt + h, xl:xl + w] = legend
    return out


def run_merge_tama(site: str, limit: int | None, legend: np.ndarray) -> None:
    rep = rf"{DATA_DIR}\{site}\landmark\landmark_report"
    files = sorted(f for f in os.listdir(rep) if f.endswith(".png") and not f.endswith(".tmp.png"))[:limit]
    for f in tqdm(files, desc=site):
        png = os.path.join(rep, f)
        out = _convert(np.array(Image.open(png).convert("RGB")), legend)
        tmp = png + ".tmp.png"; Image.fromarray(out).save(tmp); os.replace(tmp, png)


if __name__ == "__main__":
    cmd, rest = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "images":
        run_images(rest[0], rest)
    elif cmd == "from-reports":
        run_from_reports(rest[0], rest)
    elif cmd == "merge-tama":
        ap = argparse.ArgumentParser()
        ap.add_argument("--site", choices=["강남", "신촌"], default=None)
        ap.add_argument("--limit", type=int, default=None)
        a = ap.parse_args(rest)
        legend = _legend_img()
        for s in ([a.site] if a.site else ["강남", "신촌"]):
            run_merge_tama(s, a.limit, legend)
    else:
        sys.exit(__doc__)
