"""
[전체 목적]
SMI 메타데이터(강남_DLO_Results_SMI.xlsx)와
AEC crop 데이터(강남_aec_cropped_ok.xlsx)를 통합해
모델 학습에 사용할 최종 피처 파일(강남_liver_merged_features_ok.xlsx)을 생성한다.

[처리 흐름]
  STEP 1. 데이터 로드 (SMI 메타데이터 + AEC 전체 시트)
  STEP 2. 공통 PatientID 교집합 계산
  STEP 3. 환자 필터링
           ① kVp != 100 제거 (CT 프로토콜 표준화)
           ② 나이 20세 미만 제거 (소아 제외)
           ③ ManufacturerModelName 비율 1% 미만 제거 (소수 장비 통제)
           ④ n_slices_cropped < 100 제거 (AEC 구간 너무 짧음)
           ⑤ z_range IQR 이상치 제거 (Q1-1.5IQR ~ Q3+1.5IQR 범위 밖)
  STEP 4. 최종 교집합으로 필터링 후 시트별 엑셀 저장

[출력 파일: 강남_liver_merged_features_ok.xlsx]
  metadata 시트    : PatientID, 나이, 성별, kVp, mAs, 장비모델, TAMA, IMATA, BMI, Height, Weight, SMI
  aec_cropped 시트 : PatientID, n_slices_cropped, z_range, aec_1 ~ aec_N (원본 crop)
  aec_128 시트     : PatientID, n_slices_cropped, z_range, aec_1 ~ aec_128 (128포인트 보간)
  aec_256 시트     : PatientID, n_slices_cropped, z_range, aec_1 ~ aec_256 (256포인트 보간)

[필터링 기준 설명]
  kVp == 100 유지 이유:
    kVp(관전압)에 따라 AEC 신호의 절대적 스케일이 달라진다.
    동일 kVp 환자만 모아야 AEC 신호가 서로 비교 가능한 단위를 갖는다.

  ManufacturerModelName 1% 미만 제거 이유:
    CT 장비 모델이 다르면 AEC 알고리즘·하드웨어 특성이 달라
    신호 패턴에 체계적 편향(bias)이 생길 수 있다.
    단, 전체에서 1% 미만인 소수 모델은 통계적으로 대표성이 없고
    노이즈 또는 입력 오류일 가능성이 있어 제거한다.
"""

import shutil

import pandas as pd
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────

# __file__ 기준 두 단계 위(연구코드/) 폴더를 루트로 설정
ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "강남"

smi_path = DATA_DIR / "metadata" / "강남_DLO_Results_SMI.xlsx"
aec_path = Path(r"C:\Users\jhjun\OneDrive\Desktop\대학원\data\강남\aec\liver_pubis\강남_aec_cropped_ok.xlsx")
out_path = DATA_DIR / "강남_liver_merged_features_ok.xlsx"


# ── STEP 1: 데이터 로드 ───────────────────────────────────────────────────────

df_smi = pd.read_excel(smi_path)

# sheet_name=None → 모든 시트를 {시트명: DataFrame} 딕셔너리로 로드
# aec_cropped, aec_128, aec_256 시트가 모두 포함된다
aec_sheets = pd.read_excel(aec_path, sheet_name=None)


# ── STEP 2: 공통 PatientID 교집합 계산 ───────────────────────────────────────
# SMI 메타데이터와 AEC 데이터 모두에 존재하는 환자만 사용한다.
# 어느 한 쪽에만 있는 환자는 입력 피처가 불완전하므로 제외한다.

common_ids = set(df_smi["PatientID"])           # SMI 데이터의 PatientID 집합

# 각 AEC 시트의 PatientID와 교집합을 구한다 (모든 시트에 공통인 ID만 유지)
for df_aec in aec_sheets.values():
    common_ids &= set(df_aec["PatientID"])


# ── STEP 3: 환자 필터링 ───────────────────────────────────────────────────────
# ── ① kVp 필터링 ──────────────────────────────────────────────────────────────
# kVp(관전압) 설정이 다르면 AEC(관전류) 값의 절대 스케일이 달라지므로
# 동일 kVp(100kV) 환자만 유지한다.

df_common   = df_smi[df_smi["PatientID"].isin(common_ids)]
kvp_removed = (df_common["kVp"] != 100).sum()       # 제거할 환자 수 카운트
common_ids  = set(df_common.loc[df_common["kVp"] == 100, "PatientID"])

print(f"공통 PatientID: {len(common_ids) + kvp_removed}명 "
      f"→ kVp != 100 제거 {kvp_removed}명 → 최종 {len(common_ids)}명")


# ── ② 나이 20세 미만 제거 ─────────────────────────────────────────────────────

df_common    = df_smi[df_smi["PatientID"].isin(common_ids)]
age_removed  = (df_common["PatientAge"] < 20).sum()
common_ids   = set(df_common.loc[df_common["PatientAge"] >= 20, "PatientID"])

print(f"나이 20세 미만 제거 {age_removed}명 → 최종 {len(common_ids)}명")


# ── ③ ManufacturerModelName 희귀 모델 제거 ────────────────────────────────────
# 전체에서 1% 미만의 비율을 차지하는 CT 장비 모델을 가진 환자를 제거한다.
# 소수 장비의 AEC 특성이 모델 학습에 노이즈로 작용하는 것을 방지한다.

df_common    = df_smi[df_smi["PatientID"].isin(common_ids)]
# normalize=True: 각 모델의 비율(0~1)을 계산
model_ratio  = df_common["ManufacturerModelName"].value_counts(normalize=True)
valid_models = model_ratio[model_ratio >= 0.01].index    # 1% 이상 모델만 유효

removed    = (~df_common["ManufacturerModelName"].isin(valid_models)).sum()
common_ids = set(df_common.loc[df_common["ManufacturerModelName"].isin(valid_models), "PatientID"])

print(f"ManufacturerModelName 1% 미만 제거 {removed}명 → 최종 {len(common_ids)}명")
print(f"  제거된 모델: {model_ratio[model_ratio < 0.01].to_dict()}")

# 유지된 모델 비율 출력 (검수용)
valid_ratio = model_ratio[model_ratio >= 0.01].sort_values(ascending=False)
print("  유지된 모델 비율:")
for model, ratio in valid_ratio.items():
    n = int(round(ratio * (len(common_ids) + removed)))
    print(f"    {model}: {ratio*100:.1f}% ({n}명)")


# ── ④ n_slices_cropped < 100 제거 ─────────────────────────────────────────────
# AEC crop 구간이 100슬라이스 미만이면 보간 품질이 떨어지고
# 간~두덩뼈 구간이 정상적으로 캡처되지 않은 것으로 간주해 제외한다.

df_aec_cropped  = aec_sheets["aec_cropped"]
df_aec_common   = df_aec_cropped[df_aec_cropped["PatientID"].isin(common_ids)]
slices_removed  = (df_aec_common["n_slices_cropped"] < 100).sum()
common_ids      = set(df_aec_common.loc[df_aec_common["n_slices_cropped"] >= 100, "PatientID"])

print(f"n_slices_cropped < 100 제거 {slices_removed}명 → 최종 {len(common_ids)}명")


# ── ⑤ z_range 이상치 제거 ──────────────────────────────────────────────────────
# IQR 방식으로 z_range 이상치를 제거한다.
# Q1 - 1.5*IQR 미만 또는 Q3 + 1.5*IQR 초과인 환자를 제외한다.

df_aec_common   = df_aec_cropped[df_aec_cropped["PatientID"].isin(common_ids)]
q1, q3          = df_aec_common["z_range"].quantile([0.25, 0.75])
iqr             = q3 - q1
lower, upper    = q1 - 1.5 * iqr, q3 + 1.5 * iqr
range_mask      = df_aec_common["z_range"].between(lower, upper)
range_removed   = (~range_mask).sum()
common_ids      = set(df_aec_common.loc[range_mask, "PatientID"])

print(f"z_range 이상치 제거 {range_removed}명 (범위: [{lower:.1f}, {upper:.1f}]) → 최종 {len(common_ids)}명")


# ── STEP 4: 최종 필터링 및 저장 ───────────────────────────────────────────────

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

    # ── metadata 시트 ────────────────────────────────────────────────────────
    # SRC_Report 컬럼은 HYPERLINK 경로로 모델 학습에 불필요하므로 제거
    df_smi_filtered = (
        df_smi[df_smi["PatientID"].isin(common_ids)]
        .drop(columns=["SRC_Report"])
        .copy()
    )
    df_smi_filtered.to_excel(writer, sheet_name="metadata", index=False)
    print(f"  [metadata] {len(df_smi_filtered)}행")

    # ── AEC 시트들 (aec_cropped, aec_128, aec_256) ───────────────────────────
    for sheet_name, df_aec in aec_sheets.items():
        # 공통 PatientID만 필터링
        df_aec = df_aec[df_aec["PatientID"].isin(common_ids)]
        df_aec.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  [{sheet_name}] {len(df_aec)}행")

print(f"저장 완료: {out_path}")


# ── STEP 7: 최종 PID에 해당하는 PNG 복사 ─────────────────────────────────────
# ok/bottom, ok/upper 폴더에서 common_ids에 속하는 환자의 PNG만
# ok_filtered/bottom, ok_filtered/upper 폴더로 복사한다.
# 파일명 형식: {PatientID}_{슬라이스번호}.png

OK_DIR       = Path(r"C:\Users\jhjun\OneDrive\Desktop\대학원\data\강남\aec\liver_pubis\ok")
FILTERED_DIR = OK_DIR.parent / "ok_filtered"

for sub in ("bottom", "upper"):
    src_dir = OK_DIR / sub
    dst_dir = FILTERED_DIR / sub
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for png in src_dir.glob("*.png"):
        pid = png.stem.split("_")[0]          # '1003064_163' → '1003064'
        if pid in {str(p) for p in common_ids}:
            shutil.copy2(png, dst_dir / png.name)
            copied += 1

    total = len(list(src_dir.glob("*.png")))
    print(f"  [{sub}] {total}개 중 {copied}개 복사 → {dst_dir}")

print(f"PNG 복사 완료: {FILTERED_DIR}")
