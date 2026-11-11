"""기존 full 파일의 시트로 write_full_and_filtered만 다시 돌려 metadata 칼럼을 갱신한다(세그멘테이션 없음)."""
import sys
import pandas as pd
import merge_metadata as mm


def main(site):
    slug = mm.SITE_SLUG[site]
    x = pd.read_excel(rf"{mm.DATA_DIR}\{site}\landmark\{slug}_landmark_full.xlsx", sheet_name=None)
    sheets = {k: x[k] for k in ("landmarks", "aec_total")}
    mm.write_full_and_filtered(site, slug, sheets, filtered_extra={k: v for k, v in x.items() if k.endswith("_total") and k != "aec_total"})
    f = pd.read_excel(rf"{mm.DATA_DIR}\{site}\landmark\{slug}_landmark_filtered.xlsx", sheet_name="patient_metadata")
    print(site, f.shape, list(f.columns))


if __name__ == "__main__":
    main(sys.argv[1])
