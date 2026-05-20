import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # 연구코드/
DATA_DIR = ROOT / "data" / "강남"

smi_path = DATA_DIR / "metadata" / "강남_DLO_Results_SMI.xlsx"
aec_path = DATA_DIR / "aec" / "강남_aec_cropped.xlsx"
out_path = DATA_DIR / "강남_merged_features.xlsx"

df_smi = pd.read_excel(smi_path)
aec_sheets = pd.read_excel(aec_path, sheet_name=None)  # 모든 시트 로드

# 세 데이터셋 모두에 존재하는 PatientID 교집합
common_ids = set(df_smi["PatientID"])
for df_aec in aec_sheets.values():
    common_ids &= set(df_aec["PatientID"])

# ManufacturerModelName 비율 1% 미만 제거
df_common = df_smi[df_smi["PatientID"].isin(common_ids)]
model_ratio = df_common["ManufacturerModelName"].value_counts(normalize=True)
valid_models = model_ratio[model_ratio >= 0.01].index
removed = (~df_common["ManufacturerModelName"].isin(valid_models)).sum()
common_ids = set(df_common.loc[df_common["ManufacturerModelName"].isin(valid_models), "PatientID"])

print(f"공통 PatientID: {len(common_ids) + removed}명 → ManufacturerModelName 1% 미만 제거 {removed}명 → 최종 {len(common_ids)}명")
print(f"  제거된 모델: {model_ratio[model_ratio < 0.01].to_dict()}")

# kVp == 100만 유지
df_common2 = df_smi[df_smi["PatientID"].isin(common_ids)]
kvp_removed = (df_common2["kVp"] != 100).sum()
common_ids = set(df_common2.loc[df_common2["kVp"] == 100, "PatientID"])
print(f"kVp != 100 제거 {kvp_removed}명 → 최종 {len(common_ids)}명")

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df_smi_filtered = df_smi[df_smi["PatientID"].isin(common_ids)].drop(columns=["SRC_Report"])
    df_smi_filtered.to_excel(writer, sheet_name="metadata", index=False)
    print(f"  [metadata] {len(df_smi_filtered)}행")

    for sheet_name, df_aec in aec_sheets.items():
        df_aec = df_aec.drop(columns=["No"])
        df_aec = df_aec[df_aec["PatientID"].isin(common_ids)]
        df_aec.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  [{sheet_name}] {len(df_aec)}행")

print(f"저장 완료: {out_path}")
