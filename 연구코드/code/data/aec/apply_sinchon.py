import sys
sys.path.insert(0,'.')
import pandas as pd
import check_landmarks as cl


def main():
    site='신촌'
    rv=pd.read_excel('../../../data/series_reselect_review.xlsx')
    pids=set(rv[(rv.site==site)&rv.changed].PatientID.astype(int))
    print(site,'재처리',len(pids),'명',flush=True)
    p=cl.site_paths(site)
    cl._drop_pids(p,pids)
    cl._save_checkpoint(p['checkpoint'],cl._load_checkpoint(p['checkpoint'])-pids)
    cl.run_site(site,retry_failed=False)
    print('완료')


if __name__ == "__main__":   # Windows spawn 재import 시 재실행 방지
    main()
