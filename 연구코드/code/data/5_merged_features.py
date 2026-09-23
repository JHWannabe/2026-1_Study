"""
[전체 목적]
환자 메타데이터({SITE}_DLO_Results.xlsx), 체성분 요약({SITE}_body_composition.xlsx),
AEC 곡선({SITE}_aec_cropped.xlsx, {SITE}_aec_total.xlsx), 체성분 슬라이스 곡선
({SITE}_body_comp_cropped.xlsx) 5개 원본을 PatientID로 join해 모델 학습에 사용할
최종 피처 파일({gangnam|sinchon}_final_dataset.xlsx)을 생성한다.
강남 → 신촌 순서로 순차 처리하며, 두 사이트 처리가 끝나면 단계별 제외 인원 흐름을
data/data_flowchart.xlsx로 정리해 저장한다 (제외 인원 합계가 total-final_rows와
일치하도록 FLOWCHART_STEPS/save_data_flowchart에서 검증).

[처리 흐름]
  STEP 1. 메타데이터 결측치·이상치 제거
           ① Height/Weight 결측 제거
           ② 20세 미만 제외
           ③ Height/Weight/BMI IQR 이상치 제거 (Q1-1.5IQR ~ Q3+1.5IQR 범위 밖)
           ④ 나머지 전체 칼럼 결측치 제거 (예: PatientAge 등 위 규칙에 안 걸리는 결측)
  STEP 2. AEC Crop 변동성 없음 제거
           AEC 값이 모두 NaN / CV < 0.05 / range < 10mA / R² ≥ 0.95(거의 직선)인 환자 제외
  STEP 3. 5개 원본 전체(필터링된 메타데이터, aec_cropped, aec_128, aec_total,
           body_comp_cropped 12시트, body_composition)에 공통으로 존재하는
           PatientID 교집합 계산
  STEP 4. 교집합으로 필터링 후 metadata 시트에 합칠 수 있는 스칼라 값과
           별도 시트로 저장해야 하는 곡선(다중 컬럼) 데이터를 구분해 저장

[metadata 시트에 병합 vs 별도 시트로 저장하는 기준]
  - metadata 병합 대상: 환자 1명당 값이 1개뿐인 스칼라 컬럼
      · {SITE}_DLO_Results.xlsx  : PatientID, Manufacturer, Series_Desc, PatientAge,
                                    PatientSex, kVp, Height, Weight, BMI,
                                    VAT(내장지방)_SUM→VAT, SAT(피하지방)_SUM→SAT,
                                    HTN, DM, CKD
      · {SITE}_body_composition.xlsx : IMATA_sum_cm2→IMATA, NAMA_sum_cm2→NAMA,
                                        LAMA_sum_cm2→LAMA (metadata 컬럼명 간소화를 위해
                                        _sum_cm2 접미사 제거)
                                        (series_description/manufacturer_model/kVp/
                                        seg_status/n_slices_range는 DLO_Results와
                                        중복이라 metadata에 옮기지 않음.
                                        VAT/SAT는 DLO_Results의
                                        VAT(내장지방)_SUM/SAT(피하지방)_SUM과 값이 완전히
                                        동일한 중복 컬럼이라 body_composition.xlsx
                                        쪽에서는 가져오지 않음)
      · TAMA = NAMA + LAMA (총 복부 근육 단면적, 크롭 구간 전체 슬라이스 합산값 그대로 유지)
      · SMI = TAMA / (Height[cm]/100)^2, 소수 둘째 자리 반올림 (골격근지수, 향후
        근감소증 분류에 사용. 참고: 문헌상 SMI는 L3 단면 SMA/Height²로 정의되며 정상
        cutoff가 대략 34~52 cm²/m² 수준이나, 여기서는 TAMA를 슬라이스 합산값 그대로
        써서 문헌상의 단일 슬라이스 스케일과는 다름)
  - metadata 컬럼 순서: HTN/DM/CKD는 향후 라벨/공변량으로 다루기 쉽도록 맨 뒤로 배치
  - 별도 시트 저장 대상: 환자 1명당 값이 여러 개(슬라이스·시점별 곡선)라 metadata의
    한 행에 담을 수 없는 컬럼
      · aec_cropped / aec_128 시트 (aec_cropped.xlsx)
      · aec_total 시트 (aec_total.xlsx)
      · {조직}_cropped / _128 시트 (body_comp_cropped.xlsx의 {조직}_raw/_interp 시트를
        aec_cropped/aec_128과 동일한 명명 규칙으로 리네임, 조직당 2개×6조직=12개)

[공통 컬럼명 통일]
  원본 5개 파일은 같은 값을 담은 컬럼도 시트마다 이름/대소문자가 달라(예: series_desc /
  series_description, manufacturer_model, n_slices_range) 이를 COLUMN_RENAME_MAP으로
  아래처럼 통일해 출력한다. 컬럼 순서도 CANONICAL_COLUMN_ORDER 기준으로 맞춘다.
    · series_desc / series_description        → Series_Desc
    · manufacturer / manufacturer_model        → Manufacturer
    · n_slices_range                           → n_slices_cropped (aec_cropped의 값과 동일)
    · n_slices (aec_total, 크롭 전 전체 슬라이스 수라 값이 달라 별도 유지) → n_slices_total

[출력 파일: {gangnam|sinchon}_final_dataset.xlsx]
  metadata 시트    : 위 스칼라 컬럼 병합 결과 (PatientID 기준 join)
                    컬럼 순서: ..., BMI, SMI, ..., TAMA, ..., HTN, DM, CKD (맨 뒤)
  aec_cropped 시트 : PatientID, Series_Desc, Manufacturer, kVp, n_slices_cropped, z_range, aec_1 ~ aec_N
  aec_128 시트     : PatientID, Series_Desc, Manufacturer, kVp, n_slices_cropped, z_range, aec_1 ~ aec_128
  aec_total 시트   : PatientID, Series_Desc, Manufacturer, n_slices_total, aec_1 ~ aec_N (크롭 전 전체 슬라이스 곡선)
  {조직}_cropped 시트 : {site}_body_comp_cropped.xlsx의 {조직}_raw 시트를 리네임해 옮김
                       (PatientID, Series_Desc, Manufacturer, kVp, n_slices_cropped,
                        z_range, {조직}_1 ~ _N — 슬라이스별 면적 cm2, 간→두덩뼈 방향.
                        kVp/z_range는 원본에 없어 같은 환자·같은 크롭 구간인
                        aec_cropped 시트에서 가져와 채움)
  {조직}_128 시트    : {조직}_interp 시트를 리네임. 128포인트로 보간한 {조직}_1 ~ _128
                       (조직 = SFA/VFA/TAMA/LAMA/NAMA/IMATA, 총 12개 시트.
                        kVp/n_slices_cropped/z_range는 위와 동일하게 채움)
"""

import json
import pandas as pd
from pathlib import Path

SITE_EN = {"강남": "gangnam", "신촌": "sinchon"}

AEC_CV_MIN, RANGE_MIN, R2_THRESHOLD = 0.05, 10, 0.95

BODY_COMP_SCALAR_COLS = [
    "PatientID", "IMATA", "NAMA", "LAMA", "TAMA",
]

# 원본 5개 파일마다 동일한 값을 담고 있으면서 이름/대소문자만 다른 컬럼을 통일한다.
# (예: series_desc / series_description → Series_Desc, manufacturer_model → Manufacturer,
#  n_slices_range → n_slices_cropped(값이 aec_cropped의 n_slices_cropped와 동일),
#  aec_total의 n_slices는 크롭 전 전체 슬라이스 수로 값 자체가 달라 n_slices_total로 구분,
#  VAT(내장지방)_SUM/SAT(피하지방)_SUM → VAT/SAT로 통일. 값이
#  body_composition.xlsx의 VAT/SAT와 완전히 동일해 중복 병합하지 않음.
#  IMATA/NAMA/LAMA는 metadata 스칼라 컬럼명 간소화를 위해
#  _sum_cm2 접미사를 제거)
COLUMN_RENAME_MAP = {
    "series_desc": "Series_Desc",
    "series_description": "Series_Desc",
    "manufacturer": "Manufacturer",
    "manufacturer_model": "Manufacturer",
    "n_slices_range": "n_slices_cropped",
    "n_slices": "n_slices_total",
    "VAT(내장지방)_SUM": "VAT",
    "SAT(피하지방)_SUM": "SAT",
    "IMATA_sum_cm2": "IMATA",
    "NAMA_sum_cm2": "NAMA",
    "LAMA_sum_cm2": "LAMA",
}

# 여러 시트에 공통으로 등장하는 컬럼의 출력 순서. 목록에 없는 컬럼(아이디 값 컬럼,
# aec_1~N/조직_1~N 등)은 원래 순서 그대로 뒤에 붙는다.
CANONICAL_COLUMN_ORDER = [
    "PatientID", "Series_Desc", "Manufacturer", "kVp",
    "PatientAge", "PatientSex", "Height", "Weight", "BMI", "SMI",
    "n_slices_total", "n_slices_cropped", "z_range", "seg_status",
]

# metadata 시트에서 맨 뒤로 보낼 컬럼 (향후 라벨/공변량으로 다루기 쉽도록 분리)
CANONICAL_COLUMN_BACK = ["HTN", "DM", "CKD"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_RENAME_MAP)


def _reorder_columns(df: pd.DataFrame, back: list[str] | None = None) -> pd.DataFrame:
    back = back or []
    front = [c for c in CANONICAL_COLUMN_ORDER if c in df.columns]
    back_cols = [c for c in back if c in df.columns]
    rest = [c for c in df.columns if c not in front and c not in back_cols]
    return df[front + rest + back_cols]


def _is_excluded_signal(vals: pd.Series) -> bool:
    vals = vals.dropna().astype(float)
    if vals.empty:
        return True
    mean = vals.mean()
    cv = vals.std() / mean if mean != 0 else 0.0
    if cv < AEC_CV_MIN:
        return True
    if vals.max() - vals.min() < RANGE_MIN:
        return True
    if vals.var() < 1e-10:
        return True
    r = vals.corr(pd.Series(range(len(vals)), index=vals.index).astype(float))
    return bool(pd.notna(r) and r ** 2 >= R2_THRESHOLD)


def process_site(site: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  SITE: {site}")
    print(f"{'='*60}")

    ROOT     = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data" / site

    meta_path       = DATA_DIR / "metadata" / f"{site}_DLO_Results.xlsx"
    aec_path        = DATA_DIR / "aec" / f"{site}_aec_cropped.xlsx"
    aec_total_path  = DATA_DIR / "aec" / f"{site}_aec_total.xlsx"
    bc_path         = DATA_DIR / "aec" / f"{site}_body_comp_cropped.xlsx"
    body_comp_path  = DATA_DIR / "aec" / f"{site}_body_composition.xlsx"
    out_path        = DATA_DIR / f"{SITE_EN[site]}_final_dataset.xlsx"
    stats_path      = DATA_DIR / f"{SITE_EN[site]}_final_dataset_stats.json"

    stats: dict = {}

    # ── STEP 1: 메타데이터 결측치·이상치 제거 ────────────────────────────────

    df_smi = pd.read_excel(meta_path)
    stats["total"] = int(len(df_smi))
    print(f"메타데이터 전체: {len(df_smi)}명")

    # PatientAge가 숫자가 아닌 값(과거 fill_dlo_results.py 컬럼 오기입 버그의 잔재 등)은
    # NaN으로 변환 — 이후 결측치 제거 단계에서 자연스럽게 제외되도록 한다.
    df_smi["PatientAge"] = pd.to_numeric(df_smi["PatientAge"], errors="coerce")

    # ① Height/Weight 결측 제거
    no_clinical = df_smi["Height"].isna() | df_smi["Weight"].isna()
    stats["no_height_weight"] = int(no_clinical.sum())
    if no_clinical.any():
        print(f"  임상 데이터(Height/Weight) 없음 {no_clinical.sum()}명 제외")
        df_smi = df_smi[~no_clinical].copy()

    # ② 20세 미만 제외
    under_20 = df_smi["PatientAge"] < 20
    stats["under_20"] = int(under_20.sum())
    if under_20.any():
        print(f"  나이<20세 {under_20.sum()}명 제외")
        df_smi = df_smi[~under_20].copy()

    # ③ Height/Weight/BMI IQR 이상치 제거
    outlier_mask = pd.Series(False, index=df_smi.index)
    for col in ("Height", "Weight", "BMI"):
        q1, q3 = df_smi[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        col_outlier = ~df_smi[col].between(lower, upper)
        stats[f"{col.lower()}_outlier"] = int(col_outlier.sum())
        print(f"  {col} IQR 이상치 {col_outlier.sum()}명 (범위: [{lower:.1f}, {upper:.1f}])")
        outlier_mask |= col_outlier
    stats["clinical_outlier_total"] = int(outlier_mask.sum())
    if outlier_mask.any():
        print(f"  Height/Weight/BMI 이상치 합계 {outlier_mask.sum()}명 제외")
        df_smi = df_smi[~outlier_mask].copy()

    # ④ 나머지 전체 칼럼 결측치 제거
    remaining_na = df_smi.isna().any(axis=1)
    stats["remaining_na"] = int(remaining_na.sum())
    if remaining_na.any():
        print(f"  기타 칼럼 결측치 {remaining_na.sum()}명 제외")
        df_smi = df_smi[~remaining_na].copy()

    stats["after_metadata_filter"] = int(len(df_smi))
    print(f"메타데이터 필터링 후: {len(df_smi)}명")

    # ── 원본 로드 ────────────────────────────────────────────────────────────
    # sheet_name=None → 모든 시트를 {시트명: DataFrame} 딕셔너리로 로드
    aec_sheets = pd.read_excel(aec_path, sheet_name=None)          # aec_cropped, aec_128
    df_aec_total = pd.read_excel(aec_total_path)                    # 크롭 전 전체 슬라이스 AEC 곡선
    bc_sheets = pd.read_excel(bc_path, sheet_name=None) if bc_path.exists() else {}
    df_body_comp = pd.read_excel(body_comp_path) if body_comp_path.exists() else None

    # 시트마다 다르게 표기된 공통 컬럼명(대소문자/명칭)을 통일
    df_smi = _normalize_columns(df_smi)
    aec_sheets = {name: _normalize_columns(df) for name, df in aec_sheets.items()}
    df_aec_total = _normalize_columns(df_aec_total)
    bc_sheets = {name: _normalize_columns(df) for name, df in bc_sheets.items()}
    if df_body_comp is not None:
        df_body_comp = _normalize_columns(df_body_comp)
        # TAMA(총 복부 근육 단면적) = 정상 감쇠 근육(NAMA) + 저감쇠 근육(LAMA) 합산값 그대로 유지
        df_body_comp["TAMA"] = df_body_comp["NAMA"] + df_body_comp["LAMA"]

    # ── STEP 2: AEC Crop 변동성 없음 제거 ────────────────────────────────────
    df_aec_cropped = aec_sheets["aec_cropped"]
    aec_cols = [c for c in df_aec_cropped.columns if c.startswith("aec_")]
    excluded_signal = df_aec_cropped[aec_cols].apply(_is_excluded_signal, axis=1)
    stats["aec_no_variability"] = int(excluded_signal.sum())
    stats["aec_cropped_total"] = int(len(df_aec_cropped))
    print(f"AEC Crop 변동성 없음 {excluded_signal.sum()}명 제외")
    variable_aec_ids = set(df_aec_cropped.loc[~excluded_signal, "PatientID"])

    # 체성분 슬라이스 곡선 시트(body_comp_cropped.xlsx) 원본에는 kVp/z_range가 없어
    # 같은 환자·같은 크롭 구간인 aec_cropped 시트에서 값을 가져와 채운다.
    if bc_sheets:
        aec_meta = df_aec_cropped[["PatientID", "kVp", "z_range"]]
        bc_sheets = {name: df.merge(aec_meta, on="PatientID", how="left") for name, df in bc_sheets.items()}

    # ── STEP 3: 5개 원본 전체의 PatientID 교집합 계산 ────────────────────────
    common_ids = set(df_smi["PatientID"]) & variable_aec_ids
    for df_aec in aec_sheets.values():
        common_ids &= set(df_aec["PatientID"])
    common_ids &= set(df_aec_total["PatientID"])
    for df_bc in bc_sheets.values():
        common_ids &= set(df_bc["PatientID"])
    if df_body_comp is not None:
        common_ids &= set(df_body_comp["PatientID"])

    stats["common_patient_ids"] = int(len(common_ids))
    print(f"5개 원본 공통 PatientID: {len(common_ids)}명")

    # ── STEP 4: 최종 필터링 및 저장 ──────────────────────────────────────────

    def in_common(df):
        return df[df["PatientID"].isin(common_ids)].copy()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # ── metadata 시트: 스칼라 값만 PatientID 기준으로 병합 ──────────────
        df_meta = in_common(df_smi)
        if df_body_comp is not None:
            df_body_comp_scalar = in_common(df_body_comp)[BODY_COMP_SCALAR_COLS]
            df_meta = df_meta.merge(df_body_comp_scalar, on="PatientID", how="left")
            # SMI(골격근지수) = TAMA(cm2) / 신장(m)^2 — Height는 cm 단위로 저장돼 있음
            df_meta["SMI"] = (df_meta["TAMA"] / (df_meta["Height"] / 100) ** 2).round(2)
        df_meta = _reorder_columns(df_meta, back=CANONICAL_COLUMN_BACK)
        df_meta.to_excel(writer, sheet_name="metadata", index=False)
        print(f"  [metadata] {len(df_meta)}행")

        # ── AEC 곡선 시트들 (aec_cropped, aec_128) ──────────────────────────
        for sheet_name, df_aec in aec_sheets.items():
            df_aec_filtered = _reorder_columns(in_common(df_aec))
            df_aec_filtered.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  [{sheet_name}] {len(df_aec_filtered)}행")

        # ── aec_total 시트 (크롭 전 전체 슬라이스 AEC 곡선) ─────────────────
        df_aec_total_filtered = _reorder_columns(in_common(df_aec_total))
        df_aec_total_filtered.to_excel(writer, sheet_name="aec_total", index=False)
        print(f"  [aec_total] {len(df_aec_total_filtered)}행")

        # ── 체성분 슬라이스 곡선 시트 ({조직}_cropped / _128, 12개) ──────────
        # 원본 시트명은 {조직}_raw/{조직}_interp이지만, aec_cropped/aec_128과 동일한
        # 명명 규칙(크롭 구간 원본=cropped, 128포인트 보간=128)으로 맞춰 출력한다.
        for sheet_name, df_bc in bc_sheets.items():
            out_sheet = sheet_name.replace("_raw", "_cropped").replace("_interp", "_128")
            df_bc_filtered = _reorder_columns(in_common(df_bc))
            df_bc_filtered.to_excel(writer, sheet_name=out_sheet, index=False)
            print(f"  [{out_sheet}] {len(df_bc_filtered)}행")

    stats["final_rows"] = int(len(df_meta))
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {out_path}")
    return stats


# ── 데이터 흐름(제외 인원) 정리 ────────────────────────────────────────────────

# (라벨, stats 키) 순서. 키가 None인 행은 제외 0명인 체크포인트(구간 소계) 행.
# "_intersection_excluded"는 stats에 직접 없는 합성 키로, 메타데이터 필터링 후 인원 대비
# 5개 원본 PatientID 교집합(AEC 변동성 없음 제외 포함)에서 추가로 빠진 인원을 뜻한다.
FLOWCHART_STEPS = [
    ("전체 환자", None),
    ("Height/Weight 결측 제외", "no_height_weight"),
    ("20세 미만 제외", "under_20"),
    ("Height/Weight/BMI IQR 이상치 제외", "clinical_outlier_total"),
    ("기타 컬럼 결측치 제외", "remaining_na"),
    ("[메타데이터 필터링 후]", None),
    ("5개 원본 PatientID 교집합 미달 제외 (AEC Crop 변동성 없음 포함)", "_intersection_excluded"),
    ("[최종 데이터셋]", None),
]


def save_data_flowchart(stats_by_site: dict, out_path: Path) -> None:
    remaining = {site: stats["total"] for site, stats in stats_by_site.items()}
    rows = []
    for label, key in FLOWCHART_STEPS:
        row = {"단계": label}
        for site, stats in stats_by_site.items():
            if key is None:
                excluded = 0
            elif key == "_intersection_excluded":
                excluded = stats["after_metadata_filter"] - stats["common_patient_ids"]
            else:
                excluded = stats[key]
            remaining[site] -= excluded
            row[f"{site}_제외"] = excluded
            row[f"{site}_잔여"] = remaining[site]
        if key == "_intersection_excluded":
            row["비고"] = "; ".join(
                f"{site}: AEC 변동성 없음 {s['aec_no_variability']}명 "
                f"(AEC Crop 원본 {s['aec_cropped_total']}명 기준)"
                for site, s in stats_by_site.items()
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)

    for site, stats in stats_by_site.items():
        assert remaining[site] == stats["final_rows"], (
            f"{site}: 단계별 제외 합계가 최종 인원과 맞지 않음 "
            f"(계산값 {remaining[site]} vs final_rows {stats['final_rows']})"
        )
    print(f"데이터 흐름 저장 완료: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    stats_by_site = {SITE_EN[site]: process_site(site) for site in ["강남", "신촌"]}
    save_data_flowchart(stats_by_site, ROOT / "data" / "data_flowchart.xlsx")
