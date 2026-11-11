r"""landmark_report/partial/{pid}/{pid}.png 의 오버레이에서 {pid}_{seg,vat,sat,tama}.png 를 같은 폴더에 역추정 저장한다.
이미 4장 있으면 건너뜀(재개 가능). 사용: python partial_tissue_images.py [강남 신촌]
"""
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tissue_reports as tr

KINDS = ("seg", "vat", "sat", "tama")


def split(site):
    """landmark_report/{pid}/ 를 파일 5개면 full/, 아니면 partial/ 로 이동(재개 가능)."""
    rep = rf"{tr.DATA_DIR}\{site}\landmark\landmark_report"
    for k in ("full", "partial"):
        os.makedirs(os.path.join(rep, k), exist_ok=True)
    for pid in tqdm([p for p in os.listdir(rep) if p not in ("full", "partial")], desc=f"{site} split"):
        d = os.path.join(rep, pid)
        os.rename(d, os.path.join(rep, "full" if len(os.listdir(d)) == 5 else "partial", pid))


def main(sites):
    cl = tr._cl()
    for site in sites:
        split(site)
        root = rf"{tr.DATA_DIR}\{site}\landmark\landmark_report\partial"
        for pid in tqdm(sorted(os.listdir(root)), desc=site):
            d = os.path.join(root, pid)
            if not all(os.path.exists(os.path.join(d, f"{pid}_{k}.png")) for k in KINDS):
                gen(cl, pid, d, root)
            if len(os.listdir(d)) == 5:   # 5장 채워지면 full로 승격
                os.rename(d, os.path.join(os.path.dirname(root), "full", pid))


def gen(cl, pid, d, root):
    rp = os.path.join(d, f"{pid}.png")
    if not os.path.exists(rp):
        tqdm.write(f"  PID {pid}: 리포트 없음")
        return
    try:
        cl._save_tissue_images(pid, tr._extract(cl, rp), root)
    except Exception as e:
        tqdm.write(f"  PID {pid}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["강남", "신촌"])
