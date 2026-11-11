import os, sys, pydicom, pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8')
SP = os.path.dirname(os.path.abspath(__file__))
D = r"C:\Users\jhjun\Desktop\2026-1_Study\연구코드\data"

def hdr(a):
    site, pid = a
    pdir = rf"E:\영상제공\{site}\{site}_axial\{pid}"
    dates = set()
    try:
        for sub in sorted(os.listdir(pdir)):
            sd = os.path.join(pdir, sub)
            if not os.path.isdir(sd): continue
            for f in sorted(os.listdir(sd)):
                fp = os.path.join(sd, f)
                if os.path.isfile(fp):
                    ds = pydicom.dcmread(fp, stop_before_pixels=True)
                    for t in ("StudyDate", "SeriesDate", "AcquisitionDate"):
                        v = str(getattr(ds, t, "") or "")
                        if len(v) == 8: dates.add(v); break
                    break  # 시리즈당 파일 1개
    except Exception: pass
    return site, pid, ",".join(sorted(dates))

jobs = []
for site, sl in [("강남", "gangnam"), ("신촌", "sinchon")]:
    ids = pd.read_excel(rf"{D}\{site}\landmark\{sl}_landmark_full.xlsx", sheet_name="patient_metadata").PatientID
    jobs += [(site, int(p)) for p in ids]
print(len(jobs), "patients", flush=True)
with ThreadPoolExecutor(8) as ex:
    res = list(tqdm(ex.map(hdr, jobs), total=len(jobs), desc='CT date'))
out = pd.DataFrame(res, columns=["site", "PatientID", "study_dates"])
out.to_csv(os.path.join(SP, "ct_dates.csv"), index=False, encoding="utf-8-sig")
print("done", (out.study_dates != "").sum(), "with date", flush=True)
