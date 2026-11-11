import sys,os
sys.path.insert(0,'.')
import pandas as pd, check_landmarks as cl
rv=pd.read_excel('../../series_reselect_review.xlsx')
rv=rv[rv.changed]
for site in ['강남','신촌']:
    pids=set(rv[rv.site==site].PatientID.astype(int))
    pids.discard(9430117)
    print(site,'재처리',len(pids),'명',flush=True)
    if not pids: continue
    p=cl.site_paths(site)
    cl._drop_pids(p,pids)
    att=cl._load_checkpoint(p['checkpoint'])-pids
    if site=='강남': att.add(9430117)   # 제거 확정된 환자
    cl._save_checkpoint(p['checkpoint'],att)
    cl.run_site(site,retry_failed=False)
print('완료')
