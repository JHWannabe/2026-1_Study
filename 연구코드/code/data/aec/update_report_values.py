"""
기존 landmark_report PNG에 표시된 체성분 수치(VAT/SAT/TAMA, TAMA=NAMA+LAMA+IMATA)를 landmark.xlsx의 total 시트 값으로 바꾼다.

[왜 이미지를 직접 고치는가] 리포트는 anchor 구간 세그멘테이션으로 만들어져 total 시트(스캔 전체 세그멘테이션) 값과
소수점 이하가 조금 다르다. 세그멘테이션을 다시 돌리지 않고, 이미지 아래 수치 글자만 같은 폰트로 다시 쓴다.
CT 영상과 마스크는 그대로다. total 시트가 있는 환자·anchor만 바꾸며 여러 번 실행해도 결과가 같다(멱등).
원본은 landmark_report_old_values_{YYYYMMDD}/ 에 복사해 둔다.

사용: python update_report_values.py [--site 강남] [--limit N]
"""
import argparse
import io
import os
import shutil
from datetime import date

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data"
SITE_SLUG = {"강남": "gangnam", "신촌": "sinchon"}
ANCHOR_ORDER = ["liver_dome", "T10_center", "T11_center", "T12_center", "L1_center", "L2_center", "L3_center",
                "L4_center", "L5_center", "S1_center", "femoral_head_center", "inferior_pubic_margin"]
TISSUES = ["VAT", "SAT", "TAMA"]   # TAMA = NAMA + LAMA + IMATA


# 리포트와 같은 방식(Agg, 12pt, 110dpi)으로 글자만 그려 타이트하게 자른 RGB 배열을 반환한다.
def _text_img(txt: str) -> np.ndarray:
    fig = plt.figure(figsize=(6, 1.0), dpi=110)
    fig.text(0.5, 0.5, txt, fontsize=12, ha="center", va="center")
    buf = io.BytesIO(); fig.savefig(buf, dpi=110, format="png"); plt.close(fig)
    a = np.array(Image.open(buf).convert("RGB"))
    ys, xs = np.where((a < 250).any(2))
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# 이미지 블록(CT 영역) 행 구간을 찾아 행 번호(0~2)별로 (y0, y1)을 반환한다. 이미지 칸이 적은 행도 찾도록 낮은 기준을 쓰고,
# 행 간격(534px)으로 행 번호를 정한다. 블록이 없는 행은 None. 하나도 못 찾으면 None.
def _block_runs(nw: np.ndarray):
    rows = np.where(nw.sum(1) > 0.15 * nw.shape[1])[0]
    if len(rows) == 0:
        return None
    runs, s, p = [], rows[0], rows[0]
    for y in rows[1:]:
        if y != p + 1:
            runs.append((s, p)); s = y
        p = y
    runs.append((s, p))
    out = [None, None, None]
    for y0, y1 in runs:
        if y1 - y0 < 100:
            continue                                   # 글자 줄 조각은 무시
        k = int(round((y0 - 147) / 534))
        if 0 <= k <= 2:
            out[k] = (y0, y1)
    return out if any(out) else None


# 한 장의 리포트에서 values{anchor: {조직: 값}}의 수치 글자를 바꿔 새 이미지를 반환한다.
def update_image(png: str, values: dict):
    img = np.array(Image.open(png).convert("RGB"))
    nw = (img < 250).any(2); H, W = nw.shape
    runs = _block_runs(nw)
    if runs is None:
        return None, "이미지 블록 행 검출 실패"
    out, n_changed = img.copy(), 0
    for a, v in values.items():
        ri, ci = divmod(ANCHOR_ORDER.index(a), 4)
        if runs[ri] is None:
            continue
        y0, y1 = runs[ri]; xa, xb = ci * W // 4, (ci + 1) * W // 4
        if nw[y0 + 5, xa:xb].sum() < 0.3 * (xb - xa):
            continue                                              # 이 칸은 이미지가 없는 N/A 칸
        ys, xs = np.where(nw[y1 + 1:y1 + 110, xa:xb])   # 이미지 아래 두 줄 수치(다음 행 제목보다 위)
        if len(ys) == 0:
            return None, f"{a} 수치 글자 검출 실패"
        x0, ty0, x1, ty1 = xa + xs.min(), y1 + 1 + ys.min(), xa + xs.max() + 1, y1 + 1 + ys.max() + 1
        if not (25 <= ty1 - ty0 <= 60) or x1 - x0 > (xb - xa):
            return None, f"{a} 수치 글자 영역이 예상과 다름({x1 - x0}x{ty1 - ty0})"
        txt = f"VAT {v['VAT']:.2f}  SAT {v['SAT']:.2f}\nTAMA {v['TAMA']:.2f}"
        nt = _text_img(txt); cx = (x0 + x1) // 2; nx0 = cx - nt.shape[1] // 2
        cur = img[ty0:ty1, x0:x1]
        if cur.shape == nt.shape and abs(nx0 - x0) <= 1 and (cur == nt).all():
            continue                                              # 이미 같은 값
        out[ty0 - 2:ty1 + 2, min(x0, nx0) - 3:max(x1, nx0 + nt.shape[1]) + 3] = 255   # 기존 글자만 지움
        out[ty0:ty0 + nt.shape[0], nx0:nx0 + nt.shape[1]] = nt
        n_changed += 1
    return (out if n_changed else None), (f"{n_changed}칸 변경" if n_changed else "변경 없음")


# 한 사이트의 기존 리포트 수치를 total 시트 값으로 바꾼다(check_landmarks가 total 보충을 끝낸 뒤에도 호출한다).
def run_site(site: str, limit: int | None = None) -> None:
    slug = SITE_SLUG[site]
    x = pd.read_excel(rf"{DATA_DIR}\{site}\landmark\{slug}_landmark.xlsx", sheet_name=None)
    lm = x["landmarks"].set_index("PatientID")
    if f"{ANCHOR_ORDER[0]}_slice" not in lm.columns:             # 이전 형식이면 k/instance에서 계산
        import check_landmarks as cl
        sl = pd.DataFrame([cl._slice_numbers(r) for _, r in lm.reset_index().iterrows()], index=lm.index)
        lm = lm.join(sl)
    tot = {t: x[f"{t}_total"].set_index("PatientID") for t in TISSUES if f"{t}_total" in x}
    if len(tot) < len(TISSUES):
        print(f"[{site}] total 시트 없음 - 건너뜀"); return
    rep = rf"{DATA_DIR}\{site}\landmark\landmark_report"
    bak = rf"{DATA_DIR}\{site}\landmark\landmark_report_old_values_{date.today():%Y%m%d}"
    n = chg = same = fail = 0; fails = []
    for pid in tot["VAT"].index:
        png = os.path.join(rep, f"{pid}.png")
        if not os.path.exists(png) or pid not in lm.index or tot["VAT"].at[pid, "seg_status"] != "ok":
            continue
        values = {}
        for a in ANCHOR_ORDER:
            s = lm.at[pid, f"{a}_slice"]
            if pd.isna(s):
                continue
            vals = {t: tot[t].at[pid, f"{t}_{int(s)}"] for t in TISSUES}
            if all(pd.notna(v) for v in vals.values()):
                values[a] = vals
        if not values:
            continue
        n += 1
        out, msg = update_image(png, values)
        if out is None:
            if msg == "변경 없음": same += 1
            else: fail += 1; fails.append((pid, msg))
            continue
        os.makedirs(bak, exist_ok=True)
        if not os.path.exists(os.path.join(bak, f"{pid}.png")):
            shutil.copy2(png, os.path.join(bak, f"{pid}.png"))
        tmp = png + ".tmp.png"; Image.fromarray(out).save(tmp); os.replace(tmp, png); chg += 1
        if limit and chg >= limit:
            break
    print(f"[{site}] total 시트가 있는 리포트 {n}장 | 변경 {chg} | 이미 같음 {same} | 실패 {fail}")
    for f in fails[:10]:
        print("  실패", f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(SITE_SLUG), default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    for site in ([args.site] if args.site else list(SITE_SLUG)):
        run_site(site, args.limit)


if __name__ == "__main__":
    main()
