"""
[전체 목적]
aec_cropped 시트의 AEC 신호에서 "거의 직선에 가까운" 환자를 제거한다.

[직선 판정 기준]
  환자별 유효 AEC 값(aec_1 ~ aec_{n_slices_cropped})에 선형 회귀를 적용하고
  결정계수 R² 가 R2_THRESHOLD 이상이면 직선으로 판정하여 제거한다.

  R² = 1.0 → 완벽한 직선 (단조증가/감소 포함)
  R² ≥ 0.95 → 거의 직선 (기본 임계값)

  추가로 신호 변동폭이 매우 작은 경우(최대-최소 < RANGE_MIN)도 제거한다.
  (완전 고정값은 4_crop_aec.py에서 이미 제거됨)

[출력]
  입력 파일에 새 시트를 덮어쓰지 않고 별도 파일로 저장한다.
  제거된 환자 목록은 콘솔 및 제거_목록 시트로 저장된다.
"""

import numpy as np
import pandas as pd
from scipy import stats

# ── 설정 ──────────────────────────────────────────────────────────────────────

SITE     = "강남"
DATA_DIR = rf"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\{SITE}"

IN_PATH  = rf"{DATA_DIR}\{SITE}_liver_merged_features_ok.xlsx"
OUT_PATH = rf"{DATA_DIR}\{SITE}_liver_merged_features_ok_filtered.xlsx"

R2_THRESHOLD = 0.95   # R² 이 이 값 이상이면 "거의 직선"으로 판정
RANGE_MIN    = 10     # AEC 최대-최소 차이가 이 값 미만이면 변동 없음으로 판정 (mA 단위)

AEC_COLS_START = "aec_1"   # AEC 값이 시작되는 첫 번째 컬럼명


# ── 핵심 함수 ─────────────────────────────────────────────────────────────────

def compute_r2(vals: np.ndarray) -> float:
    """
    1D 배열에 선형 회귀를 적용하고 결정계수 R² 를 반환한다.

    R² = 1 - SS_res / SS_tot
      SS_res : 잔차 제곱합 (실제값 - 예측값)²의 합
      SS_tot : 전체 제곱합 (실제값 - 평균)²의 합

    배열 길이가 2 미만이거나 SS_tot ≈ 0(완전 고정값)이면 R²=1 을 반환한다.
    (완전 고정값도 '직선'으로 판정해 제거한다.)
    """
    if len(vals) < 2:
        return 1.0
    ss_tot = float(np.sum((vals - vals.mean()) ** 2))
    if ss_tot < 1e-10:
        return 1.0
    x = np.arange(len(vals), dtype=float)
    slope, intercept, r, *_ = stats.linregress(x, vals)
    return float(r ** 2)


def is_linear(vals: np.ndarray, r2_thr: float, range_min: int) -> tuple[bool, float]:
    """
    AEC 신호가 '거의 직선'인지 판정한다.

    반환: (직선 여부, R² 값)
    """
    aec_range = float(vals.max() - vals.min())
    if aec_range < range_min:
        return True, compute_r2(vals)    # 변동폭 자체가 너무 작음

    r2 = compute_r2(vals)
    return r2 >= r2_thr, r2


def extract_aec_values(row: pd.Series, aec_cols: list[str]) -> np.ndarray:
    """
    행에서 유효한 AEC 값만 추출한다.
    n_slices_cropped 까지만 사용하고 NaN은 제외한다.
    """
    n = int(row["n_slices_cropped"])
    cols_to_use = aec_cols[:n]
    vals = row[cols_to_use].values.astype(float)
    return vals[~np.isnan(vals)]


# ── 메인 처리 ─────────────────────────────────────────────────────────────────

def filter_linear_patients():
    print(f"[입력] {IN_PATH}")

    # ── aec_cropped 시트 로드 ─────────────────────────────────────────────────
    df = pd.read_excel(IN_PATH, sheet_name="aec_cropped")
    print(f"  aec_cropped 원본: {len(df)}명")

    aec_cols = [c for c in df.columns if c.startswith("aec_")]

    # ── 환자별 직선 판정 ─────────────────────────────────────────────────────
    r2_values   = []
    linear_mask = []

    for _, row in df.iterrows():
        vals = extract_aec_values(row, aec_cols)
        linear, r2 = is_linear(vals, R2_THRESHOLD, RANGE_MIN)
        r2_values.append(round(r2, 4))
        linear_mask.append(linear)

    df["r2_linear"] = r2_values
    df["is_linear"] = linear_mask

    # ── 통계 출력 ────────────────────────────────────────────────────────────
    removed_df = df[df["is_linear"]].copy()
    kept_df    = df[~df["is_linear"]].copy()

    print(f"\n[직선 판정 결과]  R²≥{R2_THRESHOLD}  또는  범위<{RANGE_MIN} mA")
    print(f"  제거: {len(removed_df)}명")
    print(f"  유지: {len(kept_df)}명")
    if not removed_df.empty:
        print(f"\n  [제거된 환자 목록]")
        for _, row in removed_df.iterrows():
            n    = int(row["n_slices_cropped"])
            vals = extract_aec_values(row, aec_cols)
            aec_range = float(vals.max() - vals.min()) if len(vals) > 0 else 0.0
            print(f"    PatientID={int(row['PatientID']):>8d}  "
                  f"n={n:>3d}  range={aec_range:>6.1f} mA  R²={row['r2_linear']:.4f}")

    # ── 필터된 aec_cropped 정리 ───────────────────────────────────────────────
    drop_cols = ["r2_linear", "is_linear"]
    kept_clean = kept_df.drop(columns=drop_cols).reset_index(drop=True)

    # 컬럼을 유효 범위까지만 유지 (모든 환자가 NaN인 aec 컬럼 제거)
    aec_col_mask = kept_clean[aec_cols].notna().any(axis=0)
    valid_aec_cols = [c for c in aec_cols if aec_col_mask[c]]
    meta_cols  = ["PatientID", "n_slices_cropped", "z_range"]
    kept_clean = kept_clean[meta_cols + valid_aec_cols]

    # ── 다른 시트도 동일 PatientID 기준으로 필터 ────────────────────────────
    kept_pids = set(kept_clean["PatientID"].astype(int).tolist())
    xl        = pd.ExcelFile(IN_PATH)

    print(f"\n[저장] {OUT_PATH}")
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        for sheet in xl.sheet_names:
            if sheet == "aec_cropped":
                kept_clean.to_excel(writer, sheet_name="aec_cropped", index=False)
                print(f"  aec_cropped : {len(kept_clean)}명 저장")
            else:
                other = pd.read_excel(IN_PATH, sheet_name=sheet)
                if "PatientID" in other.columns:
                    filtered = other[other["PatientID"].astype(int).isin(kept_pids)].reset_index(drop=True)
                    filtered.to_excel(writer, sheet_name=sheet, index=False)
                    print(f"  {sheet:15s}: {len(filtered)}명 저장 (원본 {len(other)}명)")
                else:
                    other.to_excel(writer, sheet_name=sheet, index=False)
                    print(f"  {sheet:15s}: PatientID 없음 → 그대로 저장")

        # 제거 목록 시트 추가
        removed_summary = removed_df[["PatientID", "n_slices_cropped", "z_range", "r2_linear"]].copy()
        removed_summary["aec_range_ma"] = [
            round(float(np.ptp(extract_aec_values(row, aec_cols))), 1)
            if len(extract_aec_values(row, aec_cols)) > 0 else 0.0
            for _, row in removed_df.iterrows()
        ]
        removed_summary.to_excel(writer, sheet_name="제거_직선환자", index=False)
        print(f"  제거_직선환자: {len(removed_summary)}명")

    print("\n완료.")


if __name__ == "__main__":
    filter_linear_patients()
