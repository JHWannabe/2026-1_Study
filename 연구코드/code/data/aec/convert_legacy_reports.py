"""
total 시트가 없는 환자(AEC flat, metadata 불완전, partial 등)의 이전 형식 리포트를 TAMA 형식으로 바꾼다.

이전 형식 리포트의 둘째 줄은 `NAMA a  LAMA b  IMATA c`이다. 이 숫자를 이미지에서 직접 읽어(리포트와 같은 폰트로 만든
글자 템플릿과 비교) TAMA = a + b + c로 합산하고, 둘째 줄을 `TAMA z`로 다시 쓴다. 첫 줄(VAT/SAT)은 그대로 둔다.
total 시트가 있는 환자는 update_report_values.py가 xlsx 값으로 맞추므로 여기서는 건드리지 않는다.
원본은 landmark_report_old_format_{YYYYMMDD}/ 에 복사해 둔다.

사용: python convert_legacy_reports.py --check   (인식 정확도만 확인, 파일은 바꾸지 않음)
      python convert_legacy_reports.py            (변환)
"""
import argparse, os, re, shutil
from datetime import date
import numpy as np, pandas as pd
from PIL import Image
import update_report_values as u

CHARS = "0123456789.NAMLIT"
_T = {}
def templates():
    if not _T:
        for ch in CHARS:
            t = u._text_img(ch); _T[ch] = (t < 250).any(2)
    return _T

def _segments(mask_cols):
    segs, start = [], None
    for x, v in enumerate(mask_cols):
        if v and start is None: start = x
        if not v and start is not None: segs.append((start, x)); start = None
    if start is not None: segs.append((start, len(mask_cols)))
    return segs

def read_numbers(ink):
    """둘째 줄 ink에서 NAMA/LAMA/IMATA 숫자 3개를 읽는다. 글자(NAMA 등)는 붙어 있어도 되므로 단어 단위로 나눈 뒤
    숫자 단어(2·4·6번째)만 숫자 템플릿으로 읽는다. 읽지 못하면 None."""
    ys = np.where(ink.any(1))[0]
    if len(ys) == 0: return None
    ink = ink[ys.min():ys.max() + 1]
    segs = _segments(ink.any(0)); words = []
    for s in segs:                                   # 글자 사이 간격이 5px 이상이면 단어 경계
        if words and s[0] - words[-1][-1][1] < 5: words[-1].append(s)
        else: words.append([s])
    if len(words) == 2: return "TAMA"                 # 이미 `TAMA z` 형식
    if len(words) != 6: return None
    tpl = {c: t for c, t in templates().items() if c in "0123456789."}
    vals = []
    for w in (words[1], words[3], words[5]):
        txt = ""
        for x0, x1 in w:
            ch = ink[:, x0:x1]; yy = np.where(ch.any(1))[0]; ch = ch[yy.min():yy.max() + 1]
            best, bs = None, 1e9
            for c, t in tpl.items():
                if abs(t.shape[1] - ch.shape[1]) > 1 or abs(t.shape[0] - ch.shape[0]) > 2: continue
                h, ww = min(t.shape[0], ch.shape[0]), min(t.shape[1], ch.shape[1])
                sc = (t[:h, :ww] != ch[:h, :ww]).mean() + 0.05 * (abs(t.shape[0] - ch.shape[0]) + abs(t.shape[1] - ch.shape[1]))
                if sc < bs: best, bs = c, sc
            if best is None or bs > 0.4: return None
            txt += best
        try: vals.append(float(txt))
        except ValueError: return None
    return tuple(vals)


def cells(img):
    """(anchor, 첫줄 영역, 둘째줄 ink 영역 좌표) 목록. 둘째 줄을 못 찾으면 None."""
    nw = (img < 250).any(2); H, W = nw.shape; runs = u._block_runs(nw)
    if runs is None: return None
    out = []
    for ri, run in enumerate(runs):
        if run is None: continue
        y0, y1 = run
        for ci in range(4):
            xa, xb = ci * W // 4, (ci + 1) * W // 4
            if nw[y0 + 5, xa:xb].sum() < 0.3 * (xb - xa): continue
            ys, xs = np.where(nw[y1 + 1:y1 + 110, xa:xb])
            if len(ys) == 0: return None
            top = y1 + 1 + ys.min(); bot = y1 + 1 + ys.max() + 1
            rows = nw[top:bot, xa:xb].any(1); gaps = [i for i in range(1, len(rows) - 1) if not rows[i]]
            if not gaps: return None
            g = gaps[len(gaps) // 2]; y2 = top + g + 1       # 두 줄 사이 빈 행 뒤가 둘째 줄
            out.append((u.ANCHOR_ORDER[ri * 4 + ci], xa, xb, top, y2, bot))
    return out

def parse_image(png):
    img = np.array(Image.open(png).convert("RGB")); cl = cells(img)
    if cl is None: return img, None, "칸 검출 실패"
    res = {}
    for a, xa, xb, top, y2, bot in cl:
        v = read_numbers((img[y2:bot, xa:xb] < 200).any(2))
        if v is None: return img, None, f"{a} 숫자 인식 실패"
        res[a] = v
    if all(v == "TAMA" for v in res.values()): return img, None, "이미 TAMA"
    return img, res, "ok"

def write_tama(img, png, res):
    out = img.copy(); cl = cells(img)
    for a, xa, xb, top, y2, bot in cl:
        n, l, i = res[a]; dec = max(len(str(v).split(".")[1]) for v in (n, l, i))
        txt = f"TAMA {round(n + l + i, dec):.{max(dec, 1)}f}"
        nt = u._text_img(txt)
        cols = np.where((img[y2:bot, xa:xb] < 250).any((0, 2)))[0]; cx = xa + (cols.min() + cols.max()) // 2   # 기존 둘째 줄 중심
        h = bot - y2; out[y2 - 2:bot + 2, xa:xb] = 255
        nx0 = cx - nt.shape[1] // 2; ny0 = y2 + max(0, (h - nt.shape[0]) // 2)
        out[ny0:ny0 + nt.shape[0], nx0:nx0 + nt.shape[1]] = nt
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--site"); a = ap.parse_args()
    for site in ([a.site] if a.site else list(u.SITE_SLUG)):
        slug = u.SITE_SLUG[site]
        x = pd.read_excel(rf"{u.DATA_DIR}\{site}\landmark\{slug}_landmark.xlsx", sheet_name=None); have = set(x["VAT_total"].PatientID)
        rep = rf"{u.DATA_DIR}\{site}\landmark\landmark_report"
        files = [int(f[:-4]) for f in os.listdir(rep) if f.endswith(".png") and f[:-4].isdigit()]
        todo = [p for p in files if p not in have]
        bak = rf"{u.DATA_DIR}\{site}\landmark\landmark_report_old_format_{date.today():%Y%m%d}"
        n_ok = n_done = n_fail = 0; fails = []
        for p in todo:
            png = os.path.join(rep, f"{p}.png"); img, res, msg = parse_image(png)
            if res is None:
                if msg == "이미 TAMA": n_done += 1
                else: n_fail += 1; fails.append((p, msg))
                continue
            if a.check: n_ok += 1; continue
            os.makedirs(bak, exist_ok=True); shutil.copy2(png, os.path.join(bak, f"{p}.png"))
            out = write_tama(img, png, res); tmp = png + ".tmp.png"; Image.fromarray(out).save(tmp); os.replace(tmp, png); n_ok += 1
        print(f"[{site}] 대상 {len(todo)}장 | {'인식 성공' if a.check else '변환'} {n_ok} | 이미 TAMA {n_done} | 실패 {n_fail}")
        for f in fails[:8]: print("  실패", f)

if __name__ == "__main__":
    main()
