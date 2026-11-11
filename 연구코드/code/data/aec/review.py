import sys,os
sys.path.insert(0,'.')
import pandas as pd, check_landmarks as cl
rows=[]
for site,slug in [('강남','gangnam'),('신촌','sinchon')]:
    lo,hi=cl.N_SLICES_RANGE[site]
    d=pd.read_excel(rf'../../../data/{site}/landmark/{slug}_landmark_filtered.xlsx',sheet_name='landmarks')
    o=d[(d.n_slices<lo)|(d.n_slices>hi)]
    base=rf'E:\영상제공\{site}\{site}_axial'
    for _,r in o.iterrows():
        pid=int(r.PatientID); p=os.path.join(base,str(pid))
        subs=[s for s in os.listdir(p) if os.path.isdir(os.path.join(p,s))]
        if len(subs)<2: continue
        a=cl.pick_series_folder(pid,p,subs); b=cl.pick_series_folder(pid,p,subs,(lo,hi))
        info=[]
        for s in subs:
            sd=os.path.join(p,s); n=len(cl.read_slices(sd) or [])
            info.append(f"{s} [{n}]")
        rows.append(dict(site=site,PatientID=pid,current_n=r.n_slices,current=a,proposed=b,changed=a!=b,candidates=" | ".join(info)))
        print(site,pid,'CHANGE' if a!=b else 'same',a,'->',b,flush=True)
out=pd.DataFrame(rows); out['decision']=''
out.to_excel('../../../data/series_reselect_review.xlsx',index=False)
print('saved; changed',out.changed.sum(),'of',len(out))
