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

print(f"공통 PatientID: {len(common_ids)}명")

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df_smi_filtered = df_smi[df_smi["PatientID"].isin(common_ids)]
    df_smi_filtered.to_excel(writer, sheet_name="metadata", index=False)
    print(f"  [metadata] {len(df_smi_filtered)}행")

    for sheet_name, df_aec in aec_sheets.items():
        df_aec = df_aec.drop(columns=["No"])
        merged = pd.merge(df_smi_filtered, df_aec, on="PatientID", how="inner")
        merged.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  [{sheet_name}] {len(merged)}행")

print(f"저장 완료: {out_path}")
