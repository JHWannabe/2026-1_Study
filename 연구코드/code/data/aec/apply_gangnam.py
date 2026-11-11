import sys
sys.path.insert(0,'.')
import check_landmarks as cl


def main():
    site='강남'
    pids={int(x) for x in open('gangnam_pids.txt').read().split(',')}-{9430117}
    print(site,'재처리',len(pids),'명',flush=True)
    p=cl.site_paths(site)
    cl._drop_pids(p,pids)
    att=cl._load_checkpoint(p['checkpoint'])-pids
    att.add(9430117)   # 제거 확정된 환자
    cl._save_checkpoint(p['checkpoint'],att)
    cl.run_site(site,retry_failed=False)
    print('완료')


if __name__ == "__main__":   # Windows spawn이 이 파일을 다시 import해도 재실행되지 않게 한다
    main()
