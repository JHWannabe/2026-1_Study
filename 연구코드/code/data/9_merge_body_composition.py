"""
[목적] extract_liver_pubis_aec_new10000.py가 만든 체성분·AEC 산출물(new10000_z_bounds.xlsx,
new10000_body_composition.xlsx[body_composition 시트 + body_comp_cropped의 12개 조직 시트],
new10000_aec_cropped.xlsx[aec_cropped/aec_128 시트 + aec_total 시트])을 new10000_final_dataset.xlsx에
병합해 gangnam_final_dataset.xlsx와 완전히 동일한 구조(16개 시트, metadata의 질환 플래그 이름만
이 코호트 것으로 다름)로 완성한다.

[처리 흐름]
  1. new10000_final_dataset.xlsx의 기존 metadata 시트를 읽고, 신장/체중이 결측인 환자는
     제외한다(BMI·SMI 계산 불가).
  2. new10000_body_composition.xlsx(seg_status=='ok'만)에서 VAT/SAT/IMATA/NAMA/LAMA_sum_cm2를
     metadata에 병합하고, TAMA=NAMA+LAMA, SMI=TAMA/(Height[cm]/100)^2(소수 둘째 자리)를 계산한다
     (5_merged_features.py의 gangnam 계산식과 동일). 컬럼 순서도 gangnam_final_dataset.xlsx의
     metadata와 동일하게 맞춘다.
  3. aec_cropped/aec_128은 new10000_aec_cropped.xlsx — 체성분과 동일한 간→두덩뼈 크롭으로
     계산된 버전 — 에서 가져온다. aec_i와 TAMA_i 등이 항상 같은 물리적 슬라이스를 가리키도록
     보장하기 위함(6_new10000_features.py 참고).
  4. aec_total은 new10000_aec_total.xlsx(크롭 전 전체 슬라이스 곡선)에서 가져오며, gangnam과
     동일하게 컬럼명을 Series_Desc/Manufacturer/n_slices_total로 통일한다.
  5. new10000_body_comp_cropped.xlsx의 {조직}_raw/{조직}_interp 12개 시트를 gangnam과 동일한
     이름({조직}_cropped/{조직}_128)으로 리네임해 옮긴다.
  6. metadata는 전체 코호트 명단을 그대로 유지하고, 나머지 15개 상세 시트만 공통
     PatientID로 맞춰(세그멘테이션 미완료/실패, AEC 신호 변동성 미달 등으로 일부 시트에
     없는 환자는 상세 시트에서만 제외) new10000_final_dataset.xlsx에 저장한다.
  7. 그와 별도로, metadata까지 그 공통 PatientID로 inner join해 16개 시트 전부가 정확히
     같은 환자 집합을 공유하는(gangnam_final_dataset.xlsx와 동일한 형태) 버전을
     "C:\\Users\\jhjun\\Desktop\\Clinical_Study\\data\\new10000_final_dataset.xlsx"에 저장한다.
"""

import os
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "새 데이터 10000명"

FINAL_PATH = DATA_DIR / "new10000_final_dataset.xlsx"
BODY_COMP_PATH = DATA_DIR / "new10000_body_composition.xlsx"
BODY_COMP_CROPPED_PATH = BODY_COMP_PATH  # body_composition과 같은 파일의 시트로 저장됨
AEC_CROPPED_PATH = DATA_DIR / "new10000_aec_cropped.xlsx"
AEC_TOTAL_PATH = AEC_CROPPED_PATH  # aec_cropped와 같은 파일의 시트로 저장됨
# metadata까지 상세 시트 공통 PatientID로 inner join한 버전(16개 시트 전부 동일 환자 집합)을
# 별도 위치에 저장 — FINAL_PATH의 metadata는 전체 코호트 유지 목적이라 건드리지 않는다.
INNER_JOIN_PATH = Path(r"C:\Users\jhjun\Desktop\Clinical_Study\data\new10000_final_dataset.xlsx")

TISSUE_KEYS = ["SFA", "VFA", "TAMA", "LAMA", "NAMA", "IMATA"]


def _write_excel_atomic(path: Path, sheets: dict) -> None:
    """임시 파일에 다 쓴 뒤 os.replace로 교체한다 — 3단계 파이프라인이 재시작 시/50명마다
    같은 파일에 병합을 다시 실행할 수 있어, 두 실행이 겹치면 direct write는 zip 구조가
    깨질 수 있다(경험한 BadZipFile 손상)."""
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    os.replace(tmp_path, path)


def main():
    if not BODY_COMP_PATH.exists():
        print(f"체성분 산출물이 아직 없습니다: {BODY_COMP_PATH} (extract_liver_pubis_aec_new10000.py 먼저 실행)")
        return

    existing = pd.read_excel(FINAL_PATH, sheet_name=None)
    df_meta = existing["metadata"]
    # 이 스크립트는 3단계가 더 진행될 때마다 재실행되는 걸 전제로 한다 — 이전 실행에서
    # 이미 병합해둔 체성분 컬럼이 metadata에 남아있으면 merge()가 VAT_x/VAT_y처럼
    # 겹치는 컬럼에 접미사를 붙여 df_meta["NAMA"] 같은 접근이 KeyError로 깨진다.
    # 매번 깨끗한 상태에서 다시 병합하도록 미리 지운다.
    BODY_COMP_COLS = ["VAT", "SAT", "IMATA", "NAMA", "LAMA", "TAMA", "SMI"]
    df_meta = df_meta.drop(columns=[c for c in BODY_COMP_COLS if c in df_meta.columns])

    # 신장/체중이 없으면 BMI·SMI를 못 구해 분석에 못 쓰므로 제외한다. valid_ids가
    # df_meta에서 나오므로 상세 시트에서도 자동으로 같이 빠진다.
    n_before = len(df_meta)
    df_meta = df_meta[df_meta["Height"].notna() & df_meta["Weight"].notna()].reset_index(drop=True)
    print(f"Height/Weight 결측 {n_before - len(df_meta)}명 제외 → metadata {len(df_meta)}명")

    bc = pd.read_excel(BODY_COMP_PATH, sheet_name="body_composition")
    bc_ok = bc[bc["seg_status"] == "ok"][["PatientID", "VAT_sum_cm2", "SAT_sum_cm2",
                                           "IMATA_sum_cm2", "NAMA_sum_cm2", "LAMA_sum_cm2"]]
    bc_ok = bc_ok.rename(columns={"VAT_sum_cm2": "VAT", "SAT_sum_cm2": "SAT",
                                   "IMATA_sum_cm2": "IMATA", "NAMA_sum_cm2": "NAMA", "LAMA_sum_cm2": "LAMA"})

    df_meta = df_meta.merge(bc_ok, on="PatientID", how="left")
    df_meta["TAMA"] = df_meta["NAMA"] + df_meta["LAMA"]
    df_meta["SMI"] = (df_meta["TAMA"] / (df_meta["Height"] / 100) ** 2).round(2)

    # gangnam_final_dataset.xlsx의 metadata 컬럼 순서와 동일하게 맞춘다. 질환명만 이
    # 코호트 것(DM/HTN/DL/Osteoporosis/MI/Stroke)으로 다르고 나머지는 동일한 순서.
    disease_cols = ["DM", "HTN", "DL", "Osteoporosis", "MI", "Stroke"]
    front = ["PatientID", "Series_Desc", "Manufacturer", "kVp", "PatientAge", "PatientSex",
             "Height", "Weight", "BMI", "SMI", "VAT", "SAT", "IMATA", "NAMA", "LAMA", "TAMA"]
    df_meta = df_meta[front + disease_cols]
    n_with_bodycomp = df_meta["VAT"].notna().sum()
    print(f"metadata에 체성분 병합: {n_with_bodycomp}/{len(df_meta)}명 (나머지는 세그멘테이션 미완료/실패로 NaN)")

    out_sheets = {"metadata": df_meta}
    valid_ids = set(df_meta["PatientID"])

    if AEC_CROPPED_PATH.exists():
        for sheet in ("aec_cropped", "aec_128"):
            df_aec = pd.read_excel(AEC_CROPPED_PATH, sheet_name=sheet)
            df_aec = df_aec[df_aec["PatientID"].isin(valid_ids)].reset_index(drop=True)
            out_sheets[sheet] = df_aec
            print(f"  [{sheet}] {len(df_aec)}행")
    else:
        print(f"  [aec_cropped/aec_128] 아직 없음 — 건너뜀 ({AEC_CROPPED_PATH})")

    if AEC_TOTAL_PATH.exists():
        df_aec_total = pd.read_excel(AEC_TOTAL_PATH, sheet_name="aec_total")
        df_aec_total = df_aec_total.rename(columns={
            "series_desc": "Series_Desc", "manufacturer": "Manufacturer", "n_slices": "n_slices_total",
        })
        df_aec_total = df_aec_total[df_aec_total["PatientID"].isin(valid_ids)].reset_index(drop=True)
        aec_i_cols = [c for c in df_aec_total.columns if c.startswith("aec_")]
        df_aec_total = df_aec_total[["PatientID", "Series_Desc", "Manufacturer", "n_slices_total"] + aec_i_cols]
        out_sheets["aec_total"] = df_aec_total
        print(f"  [aec_total] {len(df_aec_total)}행")
    else:
        print(f"  [aec_total] 아직 없음 — 건너뜀 ({AEC_TOTAL_PATH})")

    for tk in TISSUE_KEYS:
        for src_suffix, out_suffix in (("raw", "cropped"), ("interp", "128")):
            sheet_name = f"{tk}_{src_suffix}"
            try:
                df = pd.read_excel(BODY_COMP_CROPPED_PATH, sheet_name=sheet_name)
            except (FileNotFoundError, ValueError):
                print(f"  [{sheet_name}] 아직 없음 — 건너뜀")
                continue
            df = df[df["PatientID"].isin(valid_ids)].reset_index(drop=True)
            out_sheets[f"{tk}_{out_suffix}"] = df
            print(f"  [{tk}_{out_suffix}] {len(df)}행")

    # metadata는 코호트 전체 인구통계 명단이라 그대로 유지한다(3단계가 아직 진행 중이면
    # 대부분 체성분/AEC가 NaN인 게 정상). 나머지 15개 상세 시트만 서로 공통 PatientID로
    # 맞춰 한 시트에서라도 빠지는(세그멘테이션 미완료/실패, AEC 신호 변동성 미달 등) 환자는
    # 상세 시트들에서 제거한다. metadata까지 필터링하면 아직 3단계가 안 끝난 환자의
    # 인구통계 행이 영구히 사라지고(다음 실행이 이미 줄어든 metadata를 다시 읽으므로)
    # 복구할 방법이 없다.
    detail_sheets = {name: df for name, df in out_sheets.items() if name != "metadata"}
    common_ids = set.intersection(*(set(df["PatientID"]) for df in detail_sheets.values()))
    print(f"상세 시트 공통 PatientID: {len(common_ids)}명 (metadata는 전체 {len(df_meta)}명 유지, 상세 시트에서만 제거)")
    for name in detail_sheets:
        out_sheets[name] = out_sheets[name][out_sheets[name]["PatientID"].isin(common_ids)].reset_index(drop=True)

    _write_excel_atomic(FINAL_PATH, out_sheets)
    print(f"저장 완료: {FINAL_PATH}  (시트: {list(out_sheets.keys())}, metadata {len(df_meta)}명, 상세 공통 {len(common_ids)}명)")

    # metadata도 common_ids로 inner join한, 16개 시트 전부 같은 환자 집합인 버전을 별도 저장.
    # 이 파일은 Excel로 열려있거나 동기화 중이면 잠길 수 있는 부가 산출물이라, 여기서
    # 실패해도(PermissionError 등) FINAL_PATH 저장은 이미 끝났으니 3단계 전체를 죽이지
    # 않고 경고만 남긴다.
    try:
        inner_join_sheets = dict(out_sheets)
        inner_join_sheets["metadata"] = df_meta[df_meta["PatientID"].isin(common_ids)].reset_index(drop=True)
        INNER_JOIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _write_excel_atomic(INNER_JOIN_PATH, inner_join_sheets)
        print(f"저장 완료: {INNER_JOIN_PATH}  (16개 시트 전부 공통 {len(common_ids)}명으로 inner join)")
    except OSError as e:
        print(f"경고: {INNER_JOIN_PATH} 저장 실패(파일이 열려있거나 동기화 중일 수 있음) — {e}")


if __name__ == "__main__":
    main()
