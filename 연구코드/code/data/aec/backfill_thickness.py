"""기존 landmark 파일들의 landmarks 시트에 slice_thickness/mm_range를 채운다(DICOM 헤더만 읽음, 세그멘테이션 없음).
시리즈는 check_landmarks와 같은 규칙(pick_series_folder)으로 고른다."""
import os, sys
import numpy as np
import pandas as pd
import check_landmarks as cl
import merge_metadata as mm


def fill(site):
    slug = mm.SITE_SLUG[site]
    p = cl.site_paths(site)
    rng = cl.N_SLICES_RANGE[site]
    # 파일마다 읽지 않도록 main(전체 환자) 기준으로 한 번 계산한다
    lm = cl._read_sheets(p["out"])["landmarks"]
    vals = {}
    for n, pid in enumerate(lm.PatientID.astype(int), 1):
        if n % 100 == 0:
            print(f"  {site} {n}/{len(lm)}", flush=True)
        d = os.path.join(p["dicom_base"], str(pid))
        subs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
        f = subs[0] if len(subs) == 1 else cl.pick_series_folder(pid, d, subs, rng)
        if f is None:
            continue
        sd = os.path.join(d, f)
        vals[pid] = (cl._slice_thickness(sd), cl._mm_range(cl.read_slices(sd)))
    print(site, "계산", len(vals), flush=True)
    t = pd.DataFrame.from_dict(vals, orient="index", columns=["slice_thickness", "mm_range"])
    base = rf"{mm.DATA_DIR}\{site}\landmark\{slug}_landmark"
    for path in (p["out"], base + "_full.xlsx", base + "_filtered.xlsx"):
        x = pd.read_excel(path, sheet_name=None)
        l = x["landmarks"].drop(columns=["slice_thickness", "mm_range"], errors="ignore")
        i = list(l.columns).index("n_slices") + 1
        for k, c in enumerate(["slice_thickness", "mm_range"]):
            l.insert(i + k, c, l.PatientID.map(t[c]))
        x["landmarks"] = l
        tmp = path + ".tmp.xlsx"
        with pd.ExcelWriter(tmp, engine="xlsxwriter") as w:
            for k, d in x.items():
                d.to_excel(w, sheet_name=k, index=False)
        cl._replace_retry(tmp, path)
        print("저장", path, flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        fill(s)
